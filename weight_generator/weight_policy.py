import itertools

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.linear import LinearExpr
from weight_input import WeightingData
from weight_var import WeightVariable
import itertools
from tqdm import tqdm


class PolicyDefinition:
    """
    Policy class for handling policy objective function and constraint definition
    using input data.

    Parameters
    ---------------
    :param weight_data: Input data gathered from excel file, used in weight generation.
    :type weight_data:
    :param lp_solver:
    :type lp_solver:
    """

    def __init__(self, weight_data: WeightingData, lp_solver: Model):
        self.data = weight_data
        self.solver = lp_solver
        # self.gb_demand = weight_data.ll_ss - weight_data.demands
        # self.prod_demand = weight_data.ul_ss - weight_data.ll_ss
        self.weights = WeightVariable(weight_data, lp_solver)
        self._add_strategic_priority()
        self._add_product_priority()
        self._add_objective_func()

    def _calc_sl_matrix(self) -> pd.DataFrame:
        """
        Calculating the batch_box/Demand ratio to minimum sales loss limit.
        :return sl_coefficient: The amount of sales loss that will occur
        if we produce EXACTLY ONE batch less than enough.
        """
        sales_demands = self.data.demands
        sales_demands[self.data.solve_window_dates[-1]] = self.data.ftr_demand_mat
        # TODO: ^^^ Not a clean solution, should be fixed ^^^
        batchbox = self.data.product_data.filter(["SKU_Code", "Approved_Batch_Size_MIP"])
        sl_coefficient = np.minimum(
            np.divide(batchbox["Approved_Batch_Size_MIP"].to_numpy()[:, np.newaxis], sales_demands,
                      where=(sales_demands > 0.5)), 1)
        return sl_coefficient

    def _get_relative_weight(self, inter_code):
        """
        Function for getting the relative weight of two same-tier products
        based on resource clash and strategic weight
        :return:
        """
        # Getting coefficient of material consumption
        product_resources = self.data.contested_resources.loc[inter_code]
        con_res_array = self.data.contested_resources.to_numpy()
        material_coef = np.ones(con_res_array.shape[0])
        if con_res_array.any():
            con_res_coef = np.divide(product_resources.to_numpy()[np.newaxis, :],
                                     con_res_array,
                                     out=np.zeros_like(con_res_array),
                                     where=con_res_array != 0)
            material_coef = np.max(con_res_coef, axis=1)
        # Getting coefficient of cycle times
        product_ct = self.data.bottle_neck_ct.loc[inter_code]
        con_ct_array = self.data.bottle_neck_ct.to_numpy().copy()
        con_ct_coef = np.divide(product_ct.to_numpy()[np.newaxis, :],
                                con_ct_array,
                                out=np.zeros_like(con_ct_array),
                                where=con_ct_array != 0)
        cycletime_coef = np.max(con_ct_coef, axis=1)
        resource_coef = np.max(np.vstack([material_coef, cycletime_coef]), axis=0)
        product_strategic_weight = self.data.product_data[self.data.product_data["inter_code"] == inter_code]["Strategic_Weight"]
        strategic_coef = np.divide(product_strategic_weight.to_numpy(), self.data.product_data["Strategic_Weight"])
        return resource_coef * strategic_coef

    def _add_strategic_priority(self):
        """
        Adder function used for calculating strategic priority for each product
        as a constraint and adding it to the optimization model
        """

        sales_loss_threshold = self._calc_sl_matrix()
        tier_indices_dict = self.data.tier_indices_dict
        for rank, tier in enumerate(list(tier_indices_dict.keys())[:-1]):
            tier_rows = tier_indices_dict[tier]
            next_tier_rows = tier_indices_dict[list(tier_indices_dict.keys())[rank + 1]]
            for index in itertools.product(tier_rows, range(self.data.run_duration)):
                if sales_loss_threshold.iloc[index] > 0.01:
                    self.solver.add_constraint(self.weights.sl_var[index] * sales_loss_threshold.iloc[index] >=
                                               np.sum(self.weights.sl_var[next_tier_rows, index[1]]) * 0.5)

    def _add_monthly_priority(self):
        """
        Adder function used for calculating monthly priority for each product
        as a constraint and adding it to the optimization model
        """

        pass

    def _add_product_priority(self):
        """
        Adder function used for calculating same-tier product priority for each product
        as a constraint and adding it to the optimization model
        """
        tier_indices_dict = self.data.tier_indices_dict
        for key, value in tier_indices_dict.items():
            same_tier_products = value.copy()
            for index in tqdm(value):
                target_product = self.data.product_data.iloc[index, :]
                product_relative_weight = self._get_relative_weight(target_product["inter_code"])
                same_tier_products.remove(index)
                for cont_index in itertools.product(same_tier_products[:5], range(self.data.run_duration)):
                    self.solver.add_constraint(
                                               self.weights.sl_var[index, cont_index[1]] >=
                                               self.weights.sl_var[cont_index]
                                               * product_relative_weight[cont_index[0]]
                                               * 0.6
                                               )

    def _add_objective_func(self):
        """Adding a Minimization objective to prevent over inflation of weights."""

        self.solver.minimize(self.solver.sum(self.weights.sl_var))
