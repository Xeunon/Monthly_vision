import numpy as np
import pandas as pd
from MIP_Pro.LP.LPProblem import LPProblem
from MIP_Pro.LP.LPData import LPData
from docplex.mp.dvar import Var


def lin_expr_check(inst):
    if isinstance(inst, Var):
        return True
    else:
        return False


def generate_production_report(lp_data: LPData, lp_problem: LPProblem):
    period_count = lp_data.run_duration
    product_count = len(lp_data.product_data.index)
    machine_count = len(lp_data.machine_available_time.index)
    cols = lp_data.machine_available_time["Machine_Code"].to_list()
    index = np.repeat(lp_data.product_data["inter_code"].to_numpy(), repeats=period_count)
    x = np.zeros(shape=(product_count * period_count, machine_count), dtype=np.float32)
    for p in range(product_count):
        prod, machine = np.where(lp_problem.batch.x[p, :, :].T)
        for var in zip(prod, machine):
            from_row = p * period_count
            to_row = p * period_count + period_count
            x[from_row: to_row, :][var] = lp_problem.batch.x[p, :, :].T[var].solution_value
    batch_box = np.repeat(lp_data.product_data["Approved_Batch_Size_MIP"].to_numpy(), repeats=period_count)
    batch_unit = batch_box * np.repeat(lp_data.product_data["Num_in_Box"].to_numpy(), repeats=period_count)
    production_df = pd.DataFrame(data=x, index=index, columns=cols)
    production_box_df = pd.DataFrame(data=np.multiply(x, batch_box[:, np.newaxis]), index=index, columns=cols)
    production_unit_df = pd.DataFrame(data=np.multiply(x, batch_unit[:, np.newaxis]), index=index, columns=cols)
    total_production = np.zeros_like(lp_problem.box_output)
    a, b = np.where(lp_problem.box_output)
    for var in zip(a, b):
        total_production[var] = lp_problem.box_output[var].solution_value

    all_skus = lp_data.product_data["SKU_Code"].to_numpy()
    unique_skus, sku_count = np.unique(all_skus, return_counts=True)
    duplicated_skus = unique_skus[sku_count >= 2]
    dupl_sku_indices = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]

    def _slack_converter(slack_var, expiry=False):
        if not expiry:
            reshaped_slack = np.repeat(slack_var, sku_count, axis=0)
            reshaped_slack[dupl_sku_indices, :] = 0
        else:
            reshaped_slack = slack_var
        slack_out = np.zeros_like(reshaped_slack)
        a, b = np.where(reshaped_slack)
        for var in zip(a, b):
            slack_out[var] = reshaped_slack[var].solution_value
        return slack_out.flatten()

    demands = lp_data.demands.to_numpy()
    demands[dupl_sku_indices, :] = 0
    product_names = lp_data.product_data["SKU_name"].to_numpy()
    dataframes = [production_df, production_box_df, production_unit_df]

    production_df.insert(loc=0, column="Total_Box_Output", value=total_production.flatten())
    production_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates, product_count))
    production_box_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates, product_count))
    production_unit_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates, product_count))
    production_df.insert(loc=0, column="SKU_name", value=np.repeat(product_names, period_count))
    production_df.insert(loc=3, column="Sales_loss", value=_slack_converter(lp_problem.salesloss.var))
    production_df.insert(loc=3, column="Over_stock", value=_slack_converter(lp_problem.overstock.var))
    production_df.insert(loc=3, column="Expiry", value=_slack_converter(lp_problem.expiry.var, expiry=True))
    production_df.insert(loc=3, column="Total_demand", value=demands.flatten())

    return production_df, production_box_df, production_unit_df


def generate_op_based_report(lp_data: LPData, lp_problem: LPProblem):
    period_count = lp_data.run_duration
    product_count = len(lp_data.product_data.index)
    product_names = lp_data.product_data["SKU_name"].to_numpy()
    cols = lp_data.opc_data.iloc[:, -8:].columns.to_list()
    index = np.repeat(lp_data.product_data["SKU_Code"].to_numpy(), repeats=period_count)
    cumu_batch_output = np.zeros(shape=(product_count * period_count, len(cols)))
    # total_batch = lp_production.generate_op_batch_variables(0)
    available_wip = lp_data.wip_tensor
    cumulative_batch = lp_problem.batch.generate_cumulative_batch_variables(available_wip)
    for p in range(product_count):
        prod, operation = np.where(cumulative_batch[p, :, :].T)
        for var in zip(prod, operation):
            from_row = p * period_count
            to_row = p * period_count + period_count
            try:
                cumu_batch_output[from_row: to_row, :][var] = cumulative_batch[p, :, :].T[var].solution_value
            except AttributeError:
                cumu_batch_output[from_row: to_row, :][var] = cumulative_batch[p, :, :].T[var]
    batch_df = pd.DataFrame(data=cumu_batch_output, index=index, columns=cols)

    all_skus = lp_data.product_data["SKU_Code"].to_numpy()
    unique_skus, sku_count = np.unique(all_skus, return_counts=True)
    duplicated_skus = unique_skus[sku_count >= 2]
    dupl_sku_indices = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]

    def _slack_converter(slack_var, expiry=False):
        if not expiry:
            reshaped_slack = np.repeat(slack_var, sku_count, axis=0)
            reshaped_slack[dupl_sku_indices, :] = 0
        else:
            reshaped_slack = slack_var
        slack_out = np.zeros_like(reshaped_slack)
        a, b = np.where(reshaped_slack)
        for var in zip(a, b):
            slack_out[var] = reshaped_slack[var].solution_value
        return slack_out.flatten()

    batch_df.insert(loc=0, column="Expiry", value=_slack_converter(lp_problem.expiry.var, expiry=True))
    batch_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates, product_count))
    batch_df.insert(loc=0, column="SKU_name", value=np.repeat(product_names, period_count))
    return batch_df


def generate_machine_report(lp_data: LPData, lp_problem: LPProblem):
    period_count = lp_data.run_duration
    product_count = len(lp_data.product_data.index)
    machine_count = len(lp_data.machine_available_time.index)
    _, production_box_df, production_unit_df = generate_production_report(lp_data, lp_problem)
    monthly_box_output = production_box_df.groupby(by="Months").sum()
    monthly_unit_output = production_unit_df.groupby(by="Months").sum()
    mps_time_tensor = lp_problem.mps_times
    mps_time = np.zeros(shape=(product_count * period_count, machine_count), dtype=np.int16)
    for p in range(product_count):
        prod, machine = np.where(mps_time_tensor[p, :, :].T)
        for var in zip(prod, machine):
            from_row = p * period_count
            to_row = p * period_count + period_count
            mps_time[from_row: to_row, :][var] = mps_time_tensor[p, :, :].T[var].solution_value
    cols = lp_data.machine_available_time["Machine_Code"].to_list()
    index = np.repeat(lp_data.product_data["inter_code"].to_numpy(), repeats=period_count)
    mps_tims_df = pd.DataFrame(data=mps_time, index=index, columns=cols)
    mps_tims_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates, product_count))
    monthly_mps_time = mps_tims_df.groupby("Months").sum()
    available_times = lp_data.machine_available_time.loc[:, list(map(str, lp_data.solve_window_dates))].T.to_numpy()
    monthly_machine_reversed_time = np.divide(1, available_times, where=(available_times != 0))
    monthly_saturation = monthly_mps_time * monthly_machine_reversed_time
    return monthly_mps_time, monthly_saturation, monthly_box_output, monthly_unit_output


def generate_raw_material_report(lp_data: LPData, lp_problem: LPProblem):
    bom_rm = lp_data.bom_data[lp_data.bom_data["Machine_Allocation"].isna()]
    rm_matrix = lp_problem.rm_closing.copy()
    a, b = np.where(lp_problem.rm_closing)
    for var in zip(a, b):
        try:
            rm_matrix[var] = lp_problem.rm_closing[var].solution_value
        except AttributeError:
            rm_matrix[var] = lp_problem.rm_closing[var]
    material_codes = lp_problem.rm_codes
    opening_vec = material_codes.merge(lp_data.material_opening,
                                       how="left",
                                       left_on="code",
                                       right_index=True).fillna(np.PZERO)
    opening_vec.drop_duplicates(subset="code", inplace=True)
    opening_vec.set_index(keys="code", inplace=True, drop=True)
    # First period opening stock of raw materials
    opening_vec = opening_vec.to_numpy().flatten()
    rm_order_df = material_codes.merge(lp_data.material_arrival,
                                       how="left",
                                       left_on="code",
                                       right_index=True).fillna(np.PZERO)
    rm_order_df.drop_duplicates(subset="code", inplace=True)
    rm_order_df = rm_order_df.merge(bom_rm[["Material_Code", "Production_Support", "LT"]].drop_duplicates(),
                                    how="left", left_on="code", right_on="Material_Code")
    # New order and purchase amounts for each material for each month of the run
    rm_order = rm_order_df.filter(list(map(str, lp_data.solve_window_dates))).to_numpy()
    production_support = np.ones_like(rm_order) * 10 ** 6
    production_support[(rm_order_df["Production_Support"] != 'Final')] = 0
    for duration in rm_order_df["LT"].unique():
        production_support[:, :int(duration)][
            (rm_order_df["Production_Support"] == 'Final') & (rm_order_df["LT"] == duration)] = 0

    cumulative_support = np.cumsum(production_support, axis=1)
    cumulative_order = np.cumsum(rm_order, axis=1)
    cumulative_open = np.repeat(opening_vec[:, np.newaxis], repeats=lp_data.run_duration, axis=1)
    cols = lp_data.solve_window_dates
    # rm_df = pd.DataFrame(data=rm_matrix, columns=cols, index=material_codes.to_numpy().flatten())
    pure_rm_df = pd.DataFrame(data=rm_matrix - cumulative_support, columns=cols,
                              index=material_codes.to_numpy().flatten())
    rm_consumption_df = pd.DataFrame(data=-rm_matrix + cumulative_order + cumulative_support + cumulative_open,
                                     columns=cols,
                                     index=material_codes.to_numpy().flatten()).astype(float).round(2)
    for column_index in reversed(range(1, len(rm_consumption_df.columns))):
        rm_consumption_df.iloc[:, column_index] -= rm_consumption_df.iloc[:, column_index-1]
    return pure_rm_df, rm_consumption_df


def generate_package_material_report(lp_data: LPData, lp_problem: LPProblem):
    material_codes = pd.DataFrame(lp_problem.pm_codes, columns=["code"])
    pm_bom_df = lp_data.bom_data[~lp_data.bom_data["Machine_Allocation"].isna()]
    pm_order_df = material_codes.merge(lp_data.material_arrival,
                                       how="left",
                                       left_on="code",
                                       right_index=True).fillna(np.PZERO)
    pm_order_df.set_index(keys="code", inplace=True, drop=True)
    pm_order = pm_order_df.to_numpy()
    # Generating production support matrix for relaxing PM constraint for some materials in some periods
    pm_order_df = pm_order_df.merge(pm_bom_df[["Material_Code", "Production_Support", "LT"]].drop_duplicates(),
                                    how="left", left_on="code", right_on="Material_Code")
    production_support = np.ones_like(pm_order) * 10 ** 6
    production_support[(pm_order_df["Production_Support"] != 'Final')] = 0
    for duration in pm_order_df["LT"].unique():
        production_support[:, :int(duration)][(pm_order_df["Production_Support"] == 'Final') &
                                              (pm_order_df["LT"] == duration)] = 0
    pm_matrix = lp_problem.pm_closing.copy()
    a, b = np.where(lp_problem.pm_closing)
    for var in zip(a, b):
        try:
            pm_matrix[var] = lp_problem.pm_closing[var].solution_value
        except AttributeError:
            pm_matrix[var] = lp_problem.pm_closing[var]
    opening_vec = material_codes.merge(lp_data.material_opening,
                                       how="left",
                                       left_on="code",
                                       right_index=True).fillna(np.PZERO)
    opening_vec = opening_vec.drop(["code"], axis=1).to_numpy()
    opening_vec = opening_vec.flatten()
    cumulative_open = np.repeat(opening_vec[:, np.newaxis], repeats=lp_data.run_duration, axis=1)
    cumulative_support = np.cumsum(production_support, axis=1)
    cumulative_order = np.cumsum(pm_order, axis=1)
    cols = lp_data.solve_window_dates
    pure_pm_df = pd.DataFrame(data=pm_matrix - cumulative_support, columns=cols,
                              index=material_codes.to_numpy().flatten())
    pm_consumption_df = pd.DataFrame(data=-pm_matrix + cumulative_order + cumulative_support + cumulative_open,
                                     columns=cols,
                                     index=material_codes.to_numpy().flatten()).astype(float).round(2)
    for column_index in reversed(range(1, len(pm_consumption_df.columns))):
        pm_consumption_df.iloc[:, column_index] -= pm_consumption_df.iloc[:, column_index-1]
    return pure_pm_df, pm_consumption_df
