import numpy as np
import pandas as pd
from docplex.mp.model import Model
from docplex.mp.linear import LinearExpr
from MIP_Pro.LP.LPData import LPData
from MIP_Pro.Variables.BatchVariable import BatchVariable
from MIP_Pro.Variables.SlackVariable import SlackVariable
from MIP_Pro.Variables.TimeVariable import TimeVariable


class LPProblem:
    """Problem class for handling objective function definition and constraint definition
     using problem input data."""

    @staticmethod
    def lin_exp_check(inst):
        if isinstance(inst, LinearExpr):
            return True
        else:
            return False

    def __init__(self,
                 lp_data: LPData,
                 lp_model: Model):
        self.data = lp_data
        self.solver = lp_model
        self.batch = BatchVariable(lp_data, lp_model)
        self.prod_loss = SlackVariable("Prod_loss", lp_data, lp_model)
        self.expiry = SlackVariable("expiry", lp_data, lp_model)
        self.salesloss = SlackVariable("sales_loss", lp_data, lp_model)
        self.salesloss2 = SlackVariable("sales_loss2", lp_data, lp_model)
        self.salesloss3 = SlackVariable("sales_loss3", lp_data, lp_model)
        self.salesloss4 = SlackVariable("sales_loss4", lp_data, lp_model)
        self.gb_loss = SlackVariable("gb_loss", lp_data, lp_model)
        self.overstock = SlackVariable("over_stock", lp_data, lp_model)
        self.gb_loss_ftr = SlackVariable("future_gb_loss", lp_data, lp_model)
        self.overstock_ftr = SlackVariable("future_over_stock", lp_data, lp_model)
        self.closing_salable, self.future_closing, \
            self.box_output, self.pure_box_output = self._generate_closing_stocks()
        self.mps_times = self._add_mps_time_constraint()
        self._add_holding_prereq_constraint()
        try:
            self.rm_closing, self.rm_material_codes = self._add_raw_material_constraint()
            self.pm_closing, self.pm_material_codes = self._add_package_material_constraint()
        except AttributeError:
            pass
        try:
            self.franchise_time = TimeVariable(lp_data, lp_model)
            self._add_campaign_constraint()
        except AttributeError:
            pass
        self._add_first_period_holding_constraint()
        self._add_salesloss_constraint()
        self._add_productionloss_constraint()
        self._add_gbloss_constraint()
        self._add_overstock_constraint()
        self._add_ftr_gbloss_constraint()
        self._add_ftr_overstock_constraint()
        self._define_objective_function()

    def _generate_closing_stocks(self):
        """Calculating the closing salable stock of each period also Calculating future closing stock
        to see how much of the "next-first-period" demand we can satisfy;
        Future-closing = total_production + opening - total_demand - future_sales"""

        available_wip = self.data.wip_tensor
        total_box_output, pure_box_output = self.batch.get_output_box(available_wip)
        period_count = len(self.data.solve_window_dates)
        batch_size = self.data.product_data.drop_duplicates(subset="bom_sku")["Batch_Box"].to_numpy()
        first_opening = self.data.get_prod_weight("Opening_Stock")
        # total_box_output = np.multiply(output_batch, batch_size[:, np.newaxis])
        # Calculating the closing salable stock of each period
        cumu_box_output = np.cumsum(total_box_output, axis=1)
        unique_skus = self.data.product_data["SKU_Code"].unique()
        skus = self.data.product_data["SKU_Code"].drop(labels=self.data.mask).to_numpy()
        sku_cumu_output = np.empty(shape=(unique_skus.shape[0], period_count), dtype=np.object)
        sku_box_output = np.empty(shape=(unique_skus.shape[0], period_count), dtype=np.object)
        for i in range(unique_skus.shape[0]):
            sku = unique_skus[i]
            sku_cumu_output[i] = np.sum(cumu_box_output[np.where(skus == sku)[0]], axis=0)
            sku_box_output[i] = np.sum(total_box_output[np.where(skus == sku)[0]], axis=0)
        cumulative_production = sku_cumu_output  # + Other things like CMO
        opening_stock = np.repeat(first_opening[:, np.newaxis], period_count, axis=1)
        cumulative_demand = np.cumsum(self.data.demands.groupby(by='SKU_Code', sort=False).mean().to_numpy(), axis=1)
        closing_salable = cumulative_production - cumulative_demand + opening_stock

        # Calculating future closing stock
        future_demand = self.data.ftr_demand_mat.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        future_closing = closing_salable[:, -1] - future_demand
        return closing_salable, future_closing, sku_box_output, pure_box_output

    def _add_mps_time_constraint(self):
        timing_data = self.data.timing_data
        cycle_times_all = np.multiply(timing_data.to_numpy()[:, :, np.newaxis], self.batch.x)
        mps_time_all = np.sum(cycle_times_all, axis=0, initial=0)
        mps_time_constraint = \
            self.data.machine_available_time[list(map
                                                  (str,
                                                   self.data.solve_window_dates))].to_numpy() - mps_time_all
        for member in mps_time_constraint.flatten()[list(map(self.lin_exp_check, mps_time_constraint.flatten()))]:
            self.solver.add_constraint(member >= 0)
        return cycle_times_all

    def _add_campaign_constraint(self):
        campaign_dict = self.data.campaign_data
        for key, data in campaign_dict.items():
            product_df = data[0]
            machine_df = data[1]
            franchise_list = product_df["Franchise"].unique().tolist()
            available_times = self.data.machine_available_time[
                self.data.machine_available_time["Machine_Code"].isin(machine_df["Machine_Code"].to_list())]
            available_times.set_index(keys="Machine_Code", inplace=True)
            for franchise in franchise_list:
                machines = machine_df["Machine_Code"]
                products = product_df[product_df["Franchise"] == franchise]
                fran_index = franchise_list.index(franchise)

                for machine in machines.to_list():
                    allocations = self.data.allocations[
                        (self.data.allocations["with_site"].isin(products["with_site"].to_list())) &
                        (self.data.allocations["Machine_Code"] == machine)
                    ]
                    machine_index = machines.to_list().index(machine)
                    mps_time_index = self.data.timing_data.columns.get_loc(machine)
                    if not allocations.empty:
                        for period in range(self.data.run_duration):
                            period_column = list(map(str, self.data.solve_window_dates))[period]
                            product_indices = [self.data.timing_data.index.get_loc(with_code)
                                               for with_code
                                               in products["with_site"].to_list()
                                               ]
                            self.solver.add_constraint(
                                self.franchise_time.var[key][fran_index, machine_index, period] >=
                                np.sum(self.mps_times[product_indices, mps_time_index, period])
                            )
                            self.solver.add_constraint(
                                self.franchise_time.var[key][fran_index, machine_index, period] <=
                                available_times.loc[machine, period_column] / 2
                            )
                            self.solver.add_constraint(
                                np.sum(self.franchise_time.var[key], axis=0)[machine_index, period] <=
                                available_times.loc[machine, period_column]
                            )

    def _add_holding_prereq_constraint(self):
        product_count = len(self.data.product_data.index)
        available_wip = self.data.wip_tensor
        cumulative_batch = self.batch.generate_cumulative_batch_variables(available_wip)
        holding_bigm = self.data.generate_holding_bigm()
        opc_mat = self.data.opc_data.iloc[:, -8:].to_numpy()
        holding_mat = self.data.holding_data.iloc[:, -7:].to_numpy()
        for p in range(product_count):
            op_indices = np.where(opc_mat[p] == 1)[0]
            period_indices = np.where(self.data.prod_active_window[p])[0]
            if len(period_indices) == 0:
                continue
            holding_offset = np.repeat(holding_mat[p][np.newaxis, :], axis=0, repeats=len(period_indices))
            holding_offset[-1, :] = 0
            if len(period_indices) > 1:
                holding_offset[-2, :][holding_offset[-2, :] > 0] = 1
            if len(period_indices) > 2:
                holding_offset[-3, :][holding_offset[-3, :] > 1] = 2
            holding_offset = np.hstack([holding_offset, np.zeros(shape=len(period_indices))[:, np.newaxis]]).astype(int)
            for period in range(len(period_indices)):
                for operation in range(1, len(op_indices)):
                    op = op_indices[operation - 1]
                    next_op = op_indices[operation]
                    per = period_indices[period]
                    offset_per = per + holding_offset[period, op]
                    self.solver.add_constraint(cumulative_batch[p, next_op, per] <= cumulative_batch[p, op, per])
                    self.solver.add_constraint(
                        (cumulative_batch[p, next_op, offset_per] +
                         holding_bigm[p, op, offset_per] +
                         self.expiry.var[p, offset_per]) >= cumulative_batch[p, op, per]
                    )

    def _add_first_period_holding_constraint(self):
        pass

    def _add_raw_material_constraint(self):
        # generating raw material matrix
        period_count = self.data.run_duration
        bom_rm = self.data.bom_data[self.data.bom_data["Machine_Allocation"].isna()]
        bom_rm_mat = bom_rm.pivot_table(index="bom_sku",
                                        columns="Material_Code",
                                        values="Consumption_Factor_(Wastage)",
                                        aggfunc="mean")
        bom_rm_mat = self.data.product_data.filter(["bom_sku"]).merge(bom_rm_mat, how="left", on="bom_sku")
        bom_rm_mat = bom_rm_mat.set_index(keys="bom_sku", inplace=False, drop=True).fillna(np.PZERO)
        rm_consumption = np.repeat(bom_rm_mat.to_numpy()[:, :, np.newaxis], repeats=period_count, axis=2)
        material_codes = pd.DataFrame(data=bom_rm_mat.columns.to_list(), columns=["code"])

        material_open = material_codes.merge(self.data.material_opening,
                                           how="left",
                                           left_on="code",
                                           right_index=True).fillna(np.PZERO)
        material_open.drop_duplicates(subset="code", inplace=True)
        material_open.set_index(keys="code", inplace=True, drop=True)
        # First period opening stock of raw materials
        opening_vec = material_open["Opening"].to_numpy().flatten()
        rm_order_df = material_codes.merge(self.data.material_arrival,
                                           how="left",
                                           left_on="code",
                                           right_index=True).fillna(np.PZERO)
        rm_order_df.drop_duplicates(subset="code", inplace=True)
        # rm_order_df = rm_order_df.merge(bom_rm[["Material_Code", "Production_Support", "LT"]].drop_duplicates(),
        #                                 how="left", left_on="code", right_on="Material_Code")
        # New order and purchase amounts for each material for each month of the run
        # rm_order = rm_order_df.filter(list(map(str, self.data.solve_window_dates))).to_numpy()
        rm_order_df.set_index(keys="code", drop=True, inplace=True)
        production_support = np.zeros_like(rm_order_df).astype(bool)
        production_support[(material_open["Production_Support"] != 'Final')] = True
        num_active_dates = int(material_open["LT"].unique().max())
        active_dates = list(self.data.problem_periods.values())[:num_active_dates]
        if set(self.data.solve_window_dates).intersection(set(active_dates)):
            for duration in material_open["LT"].unique():
                active_dates = list(self.data.problem_periods.values())[:int(duration)]
                active_dates = list(set(self.data.solve_window_dates).intersection(set(active_dates)))
                active_mask = pd.DataFrame(rm_order_df.columns.map(lambda x: int(x)).to_numpy()).isin(active_dates).T
                active_mask = pd.DataFrame(np.repeat(active_mask.to_numpy(),
                                           axis=0,
                                           repeats=len(rm_order_df.index)))
                product_mask = ((material_open["Production_Support"] == 'Final') &
                                (material_open["LT"] == duration)).to_numpy()[:, np.newaxis]
                production_support += product_mask * active_mask.to_numpy()

        # Generating raw material consumption constraint
        input_batch = self.batch.get_input_batch()
        period_wise_rm_consumption = np.multiply(rm_consumption, input_batch[:, np.newaxis, :],
                                                 where=(rm_consumption > 0))
        period_wise_rm_consumption[np.where(rm_consumption == 0)] = 0
        cumu_rm_consumption = np.cumsum(np.sum(period_wise_rm_consumption, axis=0,
                                               initial=0),
                                        axis=1)

        cumulative_rm_open = np.repeat(opening_vec[:, np.newaxis], repeats=period_count, axis=1)
        cumulative_rm_order = np.cumsum(rm_order_df.to_numpy(), axis=1)
        rm_closing = cumulative_rm_open + cumulative_rm_order - cumu_rm_consumption
        for member in rm_closing.flatten()[np.array(list(map(self.lin_exp_check, rm_closing.flatten()))) *
                                           production_support.flatten()]:
            self.solver.add_constraint(member >= 0)
        return rm_closing, material_codes

    def _add_package_material_constraint(self):
        # Generating package material matrices
        period_count = self.data.run_duration
        pm_bom_df = self.data.bom_data[~self.data.bom_data["Machine_Allocation"].isna()]
        pm_machines = pm_bom_df["Machine_Allocation"].unique()
        pm_consumption_list = []
        pm_materials_list = []
        for m in pm_machines:
            # Column index of the machine in our x matrix
            machine_index = \
                self.data.machine_available_time[self.data.machine_available_time["Machine_Code"] == m].index[0]
            # Batches produced on the machine "m"
            machine_batch = self.batch.x[:, machine_index, :]
            bom_pm = pm_bom_df[pm_bom_df["Machine_Allocation"] == m]
            bom_pm_mat = bom_pm.pivot_table(index="bom_sku",
                                            columns="Material_Code",
                                            values="Consumption_Factor_(Wastage)",
                                            aggfunc="mean")
            # Sorting the PM_BOM rows to match our products
            bom_pm_mat = self.data.product_data["bom_sku"].to_frame().merge(bom_pm_mat,
                                                                               how="left",
                                                                               left_on="bom_sku",
                                                                               right_index=True).fillna(0)
            bom_pm_mat.set_index(keys="bom_sku", inplace=True, drop=True)
            pm_materials_list.append(bom_pm_mat.columns.to_numpy())
            pm_consumption = np.repeat(bom_pm_mat.to_numpy()[:, :, np.newaxis], repeats=period_count, axis=2)
            period_wise_pm_consumption = np.multiply(pm_consumption, machine_batch[:, np.newaxis, :],
                                                     where=(pm_consumption > 0))
            period_wise_pm_consumption[np.where(pm_consumption == 0)] = 0
            cumu_pm_consumption = np.cumsum(np.sum(period_wise_pm_consumption, axis=0), axis=1)
            pm_consumption_list.append(cumu_pm_consumption)

        pm_consumption_array = np.vstack(pm_consumption_list)
        pm_material_codes = np.unique(np.hstack(pm_materials_list))
        cumu_pm_consumption = np.empty(shape=(pm_material_codes.shape[0], period_count), dtype=np.object)
        for i in range(pm_material_codes.shape[0]):
            code = pm_material_codes[i]
            pm_code_mask = np.hstack(pm_materials_list) == code
            cumu_pm_consumption[i, :] = np.sum(pm_consumption_array[pm_code_mask, :], axis=0)

        avail_material_df = pd.DataFrame({"code": pm_material_codes})
        pm_order_df = avail_material_df.merge(self.data.material_arrival,
                                              how="left",
                                              left_on="code",
                                              right_index=True).fillna(np.PZERO)
        pm_order_df.set_index(keys="code", inplace=True, drop=True)
        pm_order = pm_order_df.to_numpy()
        # Generating production support matrix for relaxing PM constraint for some materials in some periods
        material_open = avail_material_df.merge(self.data.material_opening,
                                                how="left",
                                                left_on="code",
                                                right_index=True).fillna(np.PZERO)
        opening_vec = material_open.filter(["Opening"], axis=1).to_numpy()
        opening_vec = opening_vec.flatten()
        production_support = np.zeros_like(pm_order).astype(bool)
        production_support[(material_open["Production_Support"] != 'Final')] = True
        num_active_dates = int(material_open["LT"].unique().max())
        active_dates = list(self.data.problem_periods.values())[:num_active_dates]
        if set(self.data.solve_window_dates).intersection(set(active_dates)):
            for duration in material_open["LT"].unique():
                active_dates = list(self.data.problem_periods.values())[:int(duration)]
                active_dates = list(set(self.data.solve_window_dates).intersection(set(active_dates)))
                active_mask = pd.DataFrame(pm_order_df.columns.map(lambda x: int(x)).to_numpy()).isin(active_dates).T
                active_mask = pd.DataFrame(np.repeat(active_mask.to_numpy(),
                                                     axis=0,
                                                     repeats=len(pm_order_df.index)))
                product_mask = ((material_open["Production_Support"] == 'Final') &
                                (material_open["LT"] == duration)).to_numpy()[:, np.newaxis]
                production_support += product_mask * active_mask.to_numpy()

        cumulative_pm_order = np.cumsum(pm_order, axis=1)
        cumulative_pm_open = np.repeat(opening_vec[:, np.newaxis], repeats=period_count, axis=1)
        pm_closing = cumulative_pm_open + cumulative_pm_order - cumu_pm_consumption
        for member in pm_closing.flatten()[list(map(self.lin_exp_check, pm_closing.flatten())) *
                                           production_support.flatten()]:
            self.solver.add_constraint(member >= 0)
        return pm_closing, pm_material_codes

    def _add_salesloss_constraint(self):
        salesloss_limit = self.data.sales_loss_lim.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        salesloss_constraint = np.divide(self.closing_salable, salesloss_limit,
                                         where=(salesloss_limit != 0),
                                         out=np.zeros_like(salesloss_limit, dtype=np.object)) \
                               + self.salesloss.var - salesloss_limit.astype(bool).astype(int)
        salesloss_constraint2 = np.divide(self.closing_salable, salesloss_limit,
                                          where=(salesloss_limit != 0),
                                          out=np.zeros_like(salesloss_limit, dtype=np.object)) \
                                + self.salesloss2.var - salesloss_limit.astype(bool).astype(int) * 0.75
        salesloss_constraint3 = np.divide(self.closing_salable, salesloss_limit,
                                          where=(salesloss_limit != 0),
                                          out=np.zeros_like(salesloss_limit, dtype=np.object)) \
                                + self.salesloss3.var - salesloss_limit.astype(bool).astype(int) * 0.5
        salesloss_constraint4 = np.divide(self.closing_salable, salesloss_limit,
                                          where=(salesloss_limit != 0),
                                          out=np.zeros_like(salesloss_limit, dtype=np.object)) \
                                + self.salesloss4.var - salesloss_limit.astype(bool).astype(int) * 0.25
        for member in salesloss_constraint.flatten()[list(map(self.lin_exp_check, salesloss_constraint.flatten()))]:
            self.solver.add_constraint(member >= 0)
        for member in salesloss_constraint2.flatten()[list(map(self.lin_exp_check, salesloss_constraint2.flatten()))]:
            self.solver.add_constraint(member >= 0)
        for member in salesloss_constraint3.flatten()[list(map(self.lin_exp_check, salesloss_constraint3.flatten()))]:
            self.solver.add_constraint(member >= 0)
        for member in salesloss_constraint4.flatten()[list(map(self.lin_exp_check, salesloss_constraint4.flatten()))]:
            self.solver.add_constraint(member >= 0)

    def _add_productionloss_constraint(self):
        ul_ss = self.data.ul_ss.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        prodloss_constraint = np.divide(self.closing_salable, ul_ss,
                                        where=(ul_ss != 0),
                                        out=np.zeros_like(ul_ss, dtype=np.object)) \
                              + self.prod_loss.var - ul_ss.astype(bool).astype(int)
        for member in prodloss_constraint.flatten()[list(map(self.lin_exp_check, prodloss_constraint.flatten()))]:
            self.solver.add_constraint(member >= 0)

    def _add_gbloss_constraint(self):
        ll_ss = self.data.ll_ss.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        gbloss_constraint = np.divide(self.closing_salable, ll_ss,
                                      where=(ll_ss != 0),
                                      out=np.zeros_like(ll_ss, dtype=np.object)
                                      ) + self.gb_loss.var - ll_ss.astype(bool).astype(int)
        for member in gbloss_constraint.flatten()[list(map(self.lin_exp_check, gbloss_constraint.flatten()))]:
            self.solver.add_constraint(member >= 0)

    def _add_overstock_constraint(self):
        ul_ss = self.data.ul_ss.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        overstock_constraint = np.divide(self.closing_salable, ul_ss,
                                         where=(ul_ss != 0),
                                         out=np.zeros_like(ul_ss, dtype=np.object)) \
                               - self.overstock.var - ul_ss.astype(bool).astype(int)
        # overstock_constraint = self.closing_salable - self.overstock.var - ul_ss
        for member in overstock_constraint.flatten()[list(map(self.lin_exp_check, overstock_constraint.flatten()))]:
            self.solver.add_constraint(member <= 0)

    def _add_ftr_gbloss_constraint(self):
        future_ll_ss = self.data.future_ll_ss.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        future_gbloss_constraint = np.divide(self.future_closing, future_ll_ss,
                                             where=(future_ll_ss != 0),
                                             out=np.zeros_like(future_ll_ss, dtype=np.object)) \
                                   + self.gb_loss_ftr.var[:, 0] - future_ll_ss.astype(bool).astype(int)
        for member in future_gbloss_constraint.flatten()[
            list(map(self.lin_exp_check, future_gbloss_constraint.flatten()))]:
            self.solver.add_constraint(member >= 0)

    def _add_ftr_overstock_constraint(self):
        future_ul_ss = self.data.future_ul_ss.groupby(by='SKU_Code', sort=False).mean().to_numpy()
        future_overstock_constraint = np.divide(self.future_closing, future_ul_ss,
                                                where=(future_ul_ss != 0),
                                                out=np.zeros_like(future_ul_ss, dtype=np.object)) \
                                      - self.overstock_ftr.var[:, 0] - future_ul_ss.astype(bool).astype(int)
        for member in future_overstock_constraint.flatten()[
            list(map(self.lin_exp_check, future_overstock_constraint.flatten()))]:
            self.solver.add_constraint(member <= 0)

    def _define_objective_function(self):
        def duplicate_sku_calc():
            """Since numpy.unique automatically sorts its outputs, we have to reorder the outputs to match
               our input data."""
            all_skus = self.data.product_data["SKU_Code"].to_numpy()
            unique_skus, sku_counts = np.unique(all_skus, return_counts=True)
            sorted_df = pd.DataFrame(data=np.vstack([unique_skus, sku_counts]).T, columns=["SKU", "Counts"])
            ordered_df = pd.DataFrame(data=pd.unique(all_skus), columns=["SKU"])
            ordered_df = pd.merge(ordered_df, sorted_df, how="left", on="SKU")
            unique_skus, sku_counts = ordered_df["SKU"].to_numpy(), ordered_df["Counts"].to_numpy()
            duplicated_skus = unique_skus[sku_counts >= 2]
            dupl_sku_index = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]
            return dupl_sku_index, sku_counts

        farvardin = self.data.farvardin
        farvardin_f = self.data.farvardin_f
        strategic_tier = self.data.strategic_tier
        monthly_weight = self.data.monthly
        future_monthly = self.data.monthly_f
        salseloss_weight = self.data.get_prod_weight("Sales_Loss_Weight")
        strategic_weight = self.data.get_prod_weight("Strategic_Weight")
        strategic_weight_f = self.data.get_prod_weight("Strategic_Weight", future=True)
        gb_loss_weight = self.data.get_prod_weight("FG_GB_Loss_Weight")
        over_stock_weight = self.data.get_prod_weight("FG_Over_Stock_Weight")
        future_gb_loss_weight = self.data.get_prod_weight("FG_Over_Stock_Weight", future=True) / 10
        future_over_stock_weight = self.data.get_prod_weight("FG_Over_Stock_Weight", future=True) / 100
        in_gb_weight = self.data.get_prod_weight("IN_GB_Weight")
        in_gb_weight_f = self.data.get_prod_weight("IN_GB_Weight", future=True)
        op_batch_output = self.batch.generate_op_batch_variables(self.data.wip_tensor)
        all_skus = self.data.product_data["SKU_Code"].to_numpy()
        dupl_sku_index, sku_count = duplicate_sku_calc()
        site_4_mask = self.data.product_data["with_site"].map(lambda x: x[-1]) == "4"
        inter_code_weight = np.repeat(strategic_weight, sku_count, axis=0)[:, np.newaxis] * np.repeat(monthly_weight,
                                                                                                      sku_count, axis=0)

        objective = (self.solver.sum(self.salesloss.var
                                     * monthly_weight
                                     * strategic_weight[:, np.newaxis]
                                     * salseloss_weight[:, np.newaxis]
                                     * farvardin) / 10 ** 22
                     + self.solver.sum(self.salesloss2.var
                                       * monthly_weight
                                       * strategic_weight[:, np.newaxis]
                                       * salseloss_weight[:, np.newaxis]
                                       * farvardin
                                       * 0.5) / 10 ** 22
                     + self.solver.sum(self.salesloss3.var
                                       * monthly_weight
                                       * strategic_weight[:, np.newaxis]
                                       * salseloss_weight[:, np.newaxis]
                                       * farvardin
                                       * 0.25) / 10 ** 22
                     + self.solver.sum(self.salesloss4.var
                                       * monthly_weight
                                       * strategic_weight[:, np.newaxis]
                                       * salseloss_weight[:, np.newaxis]
                                       * farvardin
                                       * 0.125) / 10 ** 22
                     + self.solver.sum(self.prod_loss.var
                                       * monthly_weight
                                       * strategic_weight[:, np.newaxis]
                                       * in_gb_weight[:, np.newaxis]
                                       * farvardin) / 10 ** 22
                     + self.solver.sum(self.gb_loss.var
                                       * monthly_weight
                                       * strategic_weight[:, np.newaxis]
                                       * gb_loss_weight[:, np.newaxis]
                                       * farvardin) / 10 ** 22
                     # + self.solver.sum(self.overstock.var
                     #                   * monthly_weight
                     #                   * strategic_weight[:, np.newaxis]
                     #                   * over_stock_weight[:, np.newaxis]
                     #                   * farvardin) / 10 ** 23
                     + self.solver.sum(self.gb_loss_ftr.var
                                       * future_gb_loss_weight[:, np.newaxis]
                                       * future_monthly[:, np.newaxis]
                                       * strategic_weight_f[:, np.newaxis]
                                       * farvardin_f[:, np.newaxis]) / 10 ** 22
                     + self.solver.sum(self.overstock_ftr.var
                                       * future_monthly[:, np.newaxis]
                                       * strategic_weight_f[:, np.newaxis]
                                       * future_over_stock_weight[:, np.newaxis]
                                       * farvardin_f[:, np.newaxis]) / 10 ** 22
                     - self.solver.sum(op_batch_output[:, :-2, :]
                                       * inter_code_weight[:, np.newaxis, :]) / 10 ** 20
                     + self.solver.sum(self.expiry.var * 10 ** 27) / 10 ** 22
                     # + self.solver.sum(self.franchise_time.var[2])
                     # - self.solver.sum(self.pure_box_output[site_4_mask]) * 10 ** 20
                     )

        self.solver.minimize(objective)
