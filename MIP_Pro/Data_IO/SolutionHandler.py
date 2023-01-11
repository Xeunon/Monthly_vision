import numpy as np
import pandas as pd
from MIP_Pro.LP.LPProblem import LPProblem
from MIP_Pro.LP.LPData import LPData
from docplex.mp.dvar import Var
from datetime import datetime
from sqlalchemy import create_engine
import pyodbc
import urllib
import os
import time


class SolutionHandler:
    """Responsible for converting model outputs into reportable forms and also uploading the data to report servers."""

    def __init__(self,
                 scenario: str,
                 sync: bool = False,
                 ):
        self.MAX_TRIES = 10
        self.SCENARIO = scenario
        self.ENGINE = self.connect_to_server() if sync else None
        self.SYNC = sync
        self.save_dir = self.create_save_dir()

    def connect_to_server(self):
        params = urllib.parse.quote_plus("DRIVER={SQL Server};"
                                         "SERVER=PBI-DEVELOP;"
                                         "DATABASE=OPT;"
                                         "UID=OPT;"
                                         "PWD=]rMw&{n;"
                                         )
        engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
        # Cleaning up previous sync operations
        conn = pyodbc.connect("Driver={SQL Server};"
                              "Server=pbi-develop;"
                              "Database=OPT;"
                              "UID=OPT;"
                              "PWD=]rMw&{n;"
                              "Trusted_Connection=no;")
        cur = conn.cursor()
        cur.execute(f"DELETE FROM dbo.LPExportMonth where Scenario = '{self.SCENARIO}'")
        conn.commit()
        cur.execute(f"DELETE FROM dbo.LPExportVision where Scenario = '{self.SCENARIO}'")
        conn.commit()
        cur.execute(f"DELETE FROM dbo.LPMachineMonth where Scenario = '{self.SCENARIO}'")
        conn.commit()
        cur.execute(f"DELETE FROM dbo.LPMachineVision where Scenario = '{self.SCENARIO}'")
        conn.commit()
        cur.execute(f"DELETE FROM dbo.LPMachineDetailMonth where Scenario = '{self.SCENARIO}'")
        conn.commit()
        cur.execute(f"DELETE FROM dbo.LPMachineDetail where Scenario = '{self.SCENARIO}'")
        conn.commit()
        conn.close()
        return engine

    @staticmethod
    def create_save_dir():
        datetimeobj = datetime.now()
        timestamp = datetimeobj.strftime("%b-%d-%Y(%H-%M)")
        save_dir = "../data/MIP Output"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        folder_name = os.path.join(save_dir, timestamp)
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        return folder_name

    @staticmethod
    def load_csv_data(name, index_col=None):
        """Loading model input data from csv files placed in the CSV_DATA folder"""
        data_directory = "..\\data\\CSV_Data"
        return pd.read_csv(os.path.join(data_directory, name), index_col=index_col)

    def _aggregate_gb_data(self, gb_dataframe: pd.DataFrame, name: str, lpdata):
        allocation_df = self.load_csv_data("allocation.csv")
        sku_numbox = allocation_df.filter(["SKU_Code", "Num_in_Box"]).drop_duplicates(subset="SKU_Code")
        gb_data = gb_dataframe.merge(sku_numbox, how="left", on="SKU_Code")
        inter_codes = lpdata.product_data.filter(["SKU_Code",
                                                "Product_Code",
                                                "Product_Counter"]).drop_duplicates(subset=["SKU_Code", "Product_Code"])
        inter_codes = pd.merge(inter_codes, sku_numbox,
                               how="left", on="SKU_Code")
        limit_data = gb_data.filter(["Product_Code", "Month", name, "Num_in_Box"])
        limit_data["unit"] = limit_data[name] * limit_data["Num_in_Box"]
        aggregate = limit_data.groupby(by=["Product_Code", "Month"], as_index=False, sort=False).sum()
        aggregate.drop(labels=["Num_in_Box"], axis=1, inplace=True)
        prod_limit_data = aggregate.merge(inter_codes, how="right", on="Product_Code")
        prod_limit_data[name] = (prod_limit_data["unit"] *
                                 prod_limit_data["Product_Counter"] /
                                 prod_limit_data["Num_in_Box"]).astype(int)
        prod_limit_matrix = pd.pivot_table(prod_limit_data, values=name, index="SKU_Code", columns="Month")
        prod_limit_matrix = prod_limit_matrix.merge(inter_codes["SKU_Code"],
                                                    how="right",
                                                    right_on="SKU_Code",
                                                    left_index=True).set_index(keys="SKU_Code")
        return prod_limit_matrix

    @staticmethod
    def generate_production_report(lp_data: LPData, lp_problem: LPProblem):
        def duplicate_sku_calc():
            """Since numpy.unique automatically sorts its outputs, we have to reorder the outputs to match
               our input data."""
            all_skus = lp_data.product_data["SKU_Code"].to_numpy()
            unique_skus, sku_counts = np.unique(all_skus, return_counts=True)
            sorted_df = pd.DataFrame(data=np.vstack([unique_skus, sku_counts]).T, columns=["SKU", "Counts"])
            ordered_df = pd.DataFrame(data=pd.unique(all_skus), columns=["SKU"])
            ordered_df = pd.merge(ordered_df, sorted_df, how="left", on="SKU")
            unique_skus, sku_counts = ordered_df["SKU"].to_numpy(), ordered_df["Counts"].to_numpy()
            duplicated_skus = unique_skus[sku_counts >= 2]
            dupl_sku_index = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]
            return dupl_sku_index, sku_counts

        def get_2d_var_value(variable):
            var_value = np.zeros_like(variable)
            a, b = np.where(variable)
            for var in zip(a, b):
                try:
                    var_value[var] = variable[var].solution_value
                except AttributeError:
                    var_value[var] = variable[var]
            return var_value

        def _slack_converter(slack_var, expiry=False):
            if expiry:
                reshaped_slack = np.split(slack_var, np.cumsum(sku_count)[:-1])
                reshaped_slack = np.array([arr.sum(axis=0) for arr in reshaped_slack])
            else:
                reshaped_slack = slack_var
            slack_out = np.zeros_like(reshaped_slack)
            a, b = np.where(reshaped_slack)
            for var in zip(a, b):
                slack_out[var] = reshaped_slack[var].solution_value
            return slack_out

        period_count = len(lp_data.solve_window_dates)
        product_count = len(lp_data.product_data.index)
        machine_count = len(lp_data.machine_available_time.index)
        cols = lp_data.machine_available_time["Machine_Code"].to_list()
        index = np.repeat(lp_data.product_data["with_site"].to_numpy(), repeats=period_count)
        x = np.zeros(shape=(product_count * period_count, machine_count), dtype=np.float32)
        for p in range(product_count):
            prod, machine = np.where(lp_problem.batch.x[p, :, :].T)
            for var in zip(prod, machine):
                from_row = p * period_count
                to_row = p * period_count + period_count
                x[from_row: to_row, :][var] = lp_problem.batch.x[p, :, :].T[var].solution_value
        # batch_box = np.repeat(lp_data.product_data["Batch_Box"].to_numpy(), repeats=period_count)
        # batch_unit = batch_box * np.repeat(lp_data.product_data["Num_in_Box"].to_numpy(), repeats=period_count)
        production_df = pd.DataFrame(data=x, index=index, columns=cols)
        production_box_df = pd.merge(production_df, lp_data.product_data.filter(["SKU_Code", "with_site"]),
                                     how="left", right_on="with_site", left_index=True)
        production_box_df = production_box_df.drop(labels="with_site", axis=1)
        production_box_df.insert(loc=0, column="Month", value=np.tile(lp_data.solve_window_dates, product_count))
        production_box_df = production_box_df.groupby(by=["SKU_Code", "Month"], as_index=False).sum()
        production_box_df = production_box_df.merge(
            lp_data.product_data.filter(["SKU_Code", "SKU_name"]).drop_duplicates(subset="SKU_Code"),
            how="right", on="SKU_Code")
        production_box_df.insert(loc=1, column="SKU_name", value=production_box_df.pop("SKU_name"))

        # production_unit_df = pd.DataFrame(data=np.multiply(x, batch_unit[:, np.newaxis]), index=index, columns=cols)

        dupl_sku_indices, sku_count = duplicate_sku_calc()
        demands = lp_data.demands.to_numpy()
        # demands = np.repeat(demands, sku_count, axis=0)
        # demands[dupl_sku_indices] = 0
        product_names = lp_data.product_data["SKU_name"].to_numpy()
        # dataframes = [production_df, production_box_df, production_unit_df]
        total_output = get_2d_var_value(lp_problem.box_output)
        # total_output = np.insert(total_output, lp_data.mask, 0.0, axis=0)
        opening_stock = np.zeros_like(total_output)
        sales_loss = np.zeros_like(total_output)
        opening_stock[:, 0] = lp_data.get_prod_weight("Opening_Stock")
        sales_loss[:, 0] = np.maximum(demands[:, 0] - opening_stock[:, 0], 0)
        for period in range(1, period_count):
            opening_stock[:, period] = np.maximum(opening_stock[:, period - 1] - demands[:, period - 1], 0) \
                                       + total_output[:, period - 1]
            sales_loss[:, period] = np.maximum(demands[:, period] - opening_stock[:, period], 0)
        closing_stock = np.hstack([opening_stock, (np.maximum(opening_stock[:, -1] - demands[:, -1], 0) \
                                                   + total_output[:, -1])[:, np.newaxis]])[:, :-1]
        production_box_df.insert(loc=3, column="Box_Output", value=total_output.flatten())
        # production_box_df.insert(loc=4, column="Sales_loss", value=_slack_converter(lp_problem.salesloss.var).flatten())
        production_box_df.insert(loc=3, column="expiry", value=_slack_converter(lp_problem.expiry.var, expiry=True).flatten())
        production_box_df.insert(loc=4, column="Over_stock", value=_slack_converter(lp_problem.overstock.var).flatten())
        production_box_df.insert(loc=5, column="Total_demand", value=demands.flatten())
        production_box_df.insert(loc=6, column="Closing_stock", value=closing_stock.flatten())
        sl_lim = lp_data.dm_offset.to_numpy()
        # production_box_df["Sales_loss"] = production_box_df["Sales_loss"].to_numpy() * sl_lim.flatten()
        production_box_df.insert(loc=7, column="sl_lim", value=sl_lim.flatten())
        production_box_df.insert(loc=7, column="Sales_loss", value=sales_loss.flatten())
        production_box_df.insert(loc=8, column="Coverage",
                                 value=np.nan_to_num(np.divide(production_box_df["Closing_stock"].to_numpy(),
                                                               sl_lim.flatten(),
                                                               where=sl_lim.flatten() != 0).astype(float)))

        return production_df, production_box_df

    def generate_op_based_report(self, lp_data: LPData, lp_problem: LPProblem):
        period_count = len(lp_data.solve_window_dates)
        product_count = len(lp_data.product_data.index)
        product_names = lp_data.product_data["SKU_name"].to_numpy()
        cols = lp_data.opc_data.iloc[:, -8:].columns.to_list()
        index = np.repeat(lp_data.product_data["with_site"].to_numpy(), repeats=period_count)
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
        last_month = lp_data.solve_window_dates[-1]
        wip_df = batch_df[batch_df["Months"] == last_month].iloc[:, 3:]
        last_op_indices = np.maximum(lp_data.opc_data.iloc[:, -8:].shape[1] -
                                     np.flip(lp_data.opc_data.iloc[:, -8:].to_numpy(), axis=1).argmax(axis=1) - 1, 6)
        for i in range(product_count):
            wip = wip_df.iloc[i, :] - wip_df.iloc[i, last_op_indices[i]]
            wip_df.iloc[i, :] = np.maximum(wip.to_numpy(), 0)
        wip_df.insert(loc=0, column="Months", value=np.tile(lp_data.solve_window_dates[-1], product_count))
        if self.SYNC:
            pass
        return batch_df, wip_df

    def generate_machine_report(self, lp_data: LPData, lp_problem: LPProblem):
        period_count = len(lp_data.solve_window_dates)
        product_count = len(lp_data.product_data.index)
        machine_count = len(lp_data.machine_available_time.index)
        _, production_box_df = self.generate_production_report(lp_data, lp_problem)
        monthly_box_output = production_box_df.groupby(by="Month").sum()
        # monthly_unit_output = production_unit_df.groupby(by="Month").sum()
        mps_time_tensor = lp_problem.mps_times
        mps_time = np.zeros(shape=(product_count * period_count, machine_count), dtype=np.int16)
        for p in range(product_count):
            prod, machine = np.where(mps_time_tensor[p, :, :].T)
            for var in zip(prod, machine):
                from_row = p * period_count
                to_row = p * period_count + period_count
                mps_time[from_row: to_row, :][var] = mps_time_tensor[p, :, :].T[var].solution_value
        cols = lp_data.machine_available_time["Machine_Code"].to_list()
        index = np.repeat(lp_data.product_data["bom_sku"].to_numpy(), repeats=period_count)
        mps_tims_df = pd.DataFrame(data=mps_time, index=index, columns=cols)
        mps_tims_df.insert(loc=0, column="Month", value=np.tile(lp_data.solve_window_dates, product_count))
        monthly_mps_time = mps_tims_df.groupby("Month").sum()
        available_times = lp_data.machine_available_time.loc[:, list(map(str, lp_data.solve_window_dates))].T.to_numpy()
        monthly_machine_reversed_time = np.divide(1, available_times, where=(available_times != 0))
        monthly_saturation = monthly_mps_time * monthly_machine_reversed_time
        if self.SYNC:
            pass
        return monthly_mps_time, monthly_saturation, monthly_box_output

    @staticmethod
    def generate_raw_material_report(lp_data: LPData, lp_problem: LPProblem):
        rm_matrix = lp_problem.rm_closing.copy()
        a, b = np.where(lp_problem.rm_closing)
        for var in zip(a, b):
            try:
                rm_matrix[var] = lp_problem.rm_closing[var].solution_value
            except AttributeError:
                rm_matrix[var] = lp_problem.rm_closing[var]
        material_codes = lp_problem.rm_material_codes
        opening_vec = material_codes.merge(lp_data.material_opening,
                                           how="left",
                                           left_on="code",
                                           right_index=True).fillna(np.PZERO)
        opening_vec.drop_duplicates(subset="code", inplace=True)
        opening_vec.set_index(keys="code", inplace=True, drop=True)
        opening_vec = opening_vec.filter(items=["Opening"])
        # First period opening stock of raw materials
        opening_vec = opening_vec.to_numpy().flatten()
        rm_order_df = material_codes.merge(lp_data.material_arrival,
                                           how="left",
                                           left_on="code",
                                           right_index=True).fillna(np.PZERO)
        rm_order_df.drop_duplicates(subset="code", inplace=True)
        # New order and purchase amounts for each material for each month of the run
        rm_order = rm_order_df.filter(list(map(str, lp_data.solve_window_dates))).to_numpy()
        cumulative_order = np.cumsum(rm_order, axis=1)
        cumulative_open = np.repeat(opening_vec[:, np.newaxis], repeats=lp_data.run_duration, axis=1)
        cols = lp_data.solve_window_dates
        # rm_df = pd.DataFrame(data=rm_matrix, columns=cols, index=material_codes.to_numpy().flatten())
        pure_rm_df = pd.DataFrame(data=rm_matrix, columns=cols,
                                  index=material_codes.to_numpy().flatten())
        rm_consumption_df = pd.DataFrame(data= -rm_matrix + cumulative_order + cumulative_open,
                                         columns=cols,
                                         index=material_codes.to_numpy().flatten()).astype(float).round(2)
        for column_index in reversed(range(1, len(rm_consumption_df.columns))):
            rm_consumption_df.iloc[:, column_index] -= rm_consumption_df.iloc[:, column_index - 1]
        return pure_rm_df, rm_consumption_df

    @staticmethod
    def generate_package_material_report(lp_data: LPData, lp_problem: LPProblem):
        material_codes = pd.DataFrame(lp_problem.pm_material_codes, columns=["code"])
        pm_order_df = material_codes.merge(lp_data.material_arrival,
                                           how="left",
                                           left_on="code",
                                           right_index=True).fillna(np.PZERO)
        pm_order_df.set_index(keys="code", inplace=True, drop=True)
        pm_order = pm_order_df.to_numpy()
        # Generating production support matrix for relaxing PM constraint for some materials in some periods
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
        opening_vec = opening_vec.drop(["code"], axis=1)
        opening_vec = opening_vec.filter(items=["Opening"])
        opening_vec = opening_vec.to_numpy().flatten()
        cumulative_open = np.repeat(opening_vec[:, np.newaxis], repeats=lp_data.run_duration, axis=1)
        cumulative_order = np.cumsum(pm_order, axis=1)
        cols = lp_data.solve_window_dates
        pure_pm_df = pd.DataFrame(data=pm_matrix, columns=cols,
                                  index=material_codes.to_numpy().flatten())
        pm_consumption_df = pd.DataFrame(data= -pm_matrix + cumulative_order + cumulative_open,
                                         columns=cols,
                                         index=material_codes.to_numpy().flatten()).astype(float).round(2)
        for column_index in reversed(range(1, len(pm_consumption_df.columns))):
            pm_consumption_df.iloc[:, column_index] -= pm_consumption_df.iloc[:, column_index - 1]
        return pure_pm_df, pm_consumption_df

    def output_dataframe(self, dataframe: pd.DataFrame, name: str, has_index: bool = True):
        output_path = os.path.join(self.save_dir, name + ".csv")
        dataframe.to_csv(output_path, index=has_index)
        return

    def upload_to_server(self, dataframe: pd.DataFrame, table_name: str, index: str):
        tries = 0
        while True:
            try:
                dataframe.to_sql(name=table_name, con=self.ENGINE, index=False, if_exists="append", index_label=index)
                break
            except:
                print(f"Report Sync failed trying again... tries left:{self.MAX_TRIES - tries}")
                time.sleep(2)
            tries += 1
            if tries >= self.MAX_TRIES:
                dataframe.to_sql(name=table_name, con=self.ENGINE, index=False, if_exists="append",
                                 index_label=index)
                break
        return

    def upload_production_data(self, production_df, lp_data: LPData):
        columns = ['Tab/Cap', 'WG/DG', 'SKU_Code', 'ProductCode', 'Desc', 'ActiveProducts',
                   'StrategicTier', 'Wet Granulation', 'Syrup Mixing', 'Drying', 'Blending', 'Roller Compactor',
                   'Press', 'Capsule Filling', 'Coating', 'Blistering', 'Counter', 'Tube Filling', 'Syrup Filling',
                   'Manual Packaging', 'Standard_Batch_Size_Box', 'Standard_Batch_Size_Kilo', 'Demand',
                   'Production_Batch', 'Allocated_Batch', 'Allocated_Box', 'No_Allocated_Batch', 'No_Allocated_Box',
                   'Allocated_Kilo', 'No_Allocated_Kilo', 'Opening_Stock', 'Sales_Demand', 'Needed_Stock',
                   'Production_Demand', 'Production_Allocation', 'Production_Loss', 'Sales_Achievement', 'Sales_Loss',
                   'Xproject_Production_Allocation', 'Closing_Stock', 'Production_Loss_Total', 'Year', 'Site',
                   'Scenario']
        process_dict = {'Wet Granulation': 'Wet Granulation',
                        'Syrup Mixing': 'Syrup Mixing',
                        'Drying': 'Drying',
                        'Blending': 'Blending',
                        'Roller Compactor': 'Roller Compactor',
                        'Compression': 'Press',
                        'CAP Filling': 'Capsule Filling',
                        'Coating': 'Coating',
                        'Blistering': 'Blistering',
                        'Counter': 'Counter',
                        'Tube Filling': 'Tube Filling',
                        'Syrup Filling': 'Syrup Filling',
                        'Manual Packaging': 'Manual Packaging', }
        report_df = pd.DataFrame(data=production_df.loc[:, "SKU_Code"].drop_duplicates().values, index=None,
                                 columns=["SKU_Code"])
        temp_df = pd.DataFrame(data=production_df.loc[:, "SKU_Code"].drop_duplicates().values,
                               index=production_df.loc[:, "SKU_Code"].drop_duplicates().values,
                               columns=["SKU_Code"])
        temp_df = pd.merge(temp_df, lp_data.product_data,
                           how="left", on="SKU_Code").drop_duplicates(subset=["SKU_Code"])
        per_site_production_df = []
        for site in range(1, 5):
            site_machines = list(lp_data.machine_available_time[
                                     lp_data.machine_available_time["Site"] == site]["Machine_Code"])
            site_machines.extend(["SKU_Code", "SKU_name", "Month"])
            site_production_df = production_df.loc[:, site_machines]
            site_production_df["Site"] = site
            per_site_production_df.append(site_production_df)

        for site_production_df in per_site_production_df:
            site_production_df.reset_index(drop=True, inplace=True)
            site = site_production_df.loc[0, "Site"]
            if site == 1 or site == 2:
                site_production_df[f"Standard_Batch_Size_Box"] = (site_production_df.filter(["SKU_Code"])).merge(
                    (lp_data.allocations[
                        lp_data.allocations["Type"].isin([1, 2])]).drop_duplicates(
                        subset=["SKU_Code"]).filter(
                        items=["SKU_Code", "Batch_Box"]), how="left", on="SKU_Code").fillna(0)["Batch_Box"].values
                site_production_df[f"Standard_Batch_Size_Kilo"] = (site_production_df.filter(["SKU_Code"])).merge(
                    (lp_data.allocations[
                        lp_data.allocations["Type"].isin([1, 2])]).drop_duplicates(
                        subset=["SKU_Code"]).filter(
                        items=["SKU_Code", "Batch_Kilo"]), how="left", on="SKU_Code").fillna(0)["Batch_Kilo"].values
            else:
                site_production_df[f"Standard_Batch_Size_Box"] = (site_production_df.filter(["SKU_Code"])).merge(
                    (lp_data.allocations[
                        lp_data.allocations["Type"] == (site_production_df.loc[0, "Site"])]).drop_duplicates(
                        subset=["SKU_Code"]).filter(
                        items=["SKU_Code", "Batch_Box"]), how="left", on="SKU_Code").fillna(0)["Batch_Box"].values
                site_production_df[f"Standard_Batch_Size_Kilo"] = (site_production_df.filter(["SKU_Code"])).merge(
                    (lp_data.allocations[
                        lp_data.allocations["Type"] == (site_production_df.loc[0, "Site"])]).drop_duplicates(
                        subset=["SKU_Code"]).filter(
                        items=["SKU_Code", "Batch_Kilo"]), how="left", on="SKU_Code").fillna(0)["Batch_Kilo"].values

        for process in list(lp_data.allocations["Sub_Process"].drop_duplicates()):
            process_machines = lp_data.allocations[
                lp_data.allocations["Sub_Process"] == process]["Machine_Code"].drop_duplicates()
            for site_production_df in per_site_production_df:
                process_machines = set(process_machines).intersection(site_production_df.columns)
                site_production_df[process_dict[process]] = site_production_df.loc[:, process_machines].sum(axis=1) * \
                                                            site_production_df[f"Standard_Batch_Size_Box"].to_numpy()

        output_machines = lp_data.machine_available_time[
            lp_data.machine_available_time["Process"].isin(
                ["Blistering_Counter_Syrup_Filling", "Manual_Packaging"])]["Machine_Code"]
        for site_production_df in per_site_production_df:
            process_machines = set(output_machines).intersection(site_production_df.columns)
            site_production_df["Allocated_Box"] = site_production_df.loc[:, process_machines].sum(axis=1) * \
                                                  site_production_df[f"Standard_Batch_Size_Box"].to_numpy()
        report_df = []
        for site_production_df in per_site_production_df:
            report_df.append(site_production_df.iloc[:, -20:])
        machine_box = []
        for site_production_df in per_site_production_df:
            site_production_df.iloc[:, :-20] = site_production_df.iloc[:, :-20] * site_production_df[
                                                                                      "Standard_Batch_Size_Box"].to_numpy()[
                                                                                  :, np.newaxis]
            machine_box.append(site_production_df)
        machine_box = pd.concat(machine_box, axis=1)
        report_df = pd.concat(report_df)
        report_df.rename({"SKU_name": "Desc", "Month": "Year"}, inplace=True, axis=1)
        report_df = report_df.merge(lp_data.product_data.filter(["SKU_Code", "TabCap", "Method", "Product_Code",
                                                                 "Product_Counter", "Package_Type",
                                                                 "Strategic_Group",
                                                                 "PressType"]).drop_duplicates(subset="SKU_Code"),
                                    how="left", on="SKU_Code")
        report_df.rename({"Method": "WG/DG", "Product_Code": "ProductCode",
                          "Product_Counter": "ActiveProducts",
                          "Strategic_Group": "StrategicTier",
                          "TabCap": "Tab/Cap"}, inplace=True, axis=1)
        all_gb_data = self.load_csv_data("gb_df.csv")
        gb_data = all_gb_data[(all_gb_data["Month"].isin(report_df["Year"].drop_duplicates())) &
                              (all_gb_data["SKU_Code"].isin(report_df["SKU_Code"].drop_duplicates()))]
        salesloss_mat = self._aggregate_gb_data(gb_data, "DM_OFFSET", lp_data)
        ul_ss = self._aggregate_gb_data(gb_data, "FG_UL_SS", lp_data)
        sl_lim = pd.melt(salesloss_mat.reset_index(), id_vars=["SKU_Code"],
                         value_name="Demand", var_name="Year")
        ul_ss = pd.melt(ul_ss.reset_index(), id_vars=["SKU_Code"],
                         value_name="Production_Demand", var_name="Year")
        report_df = report_df.merge(sl_lim, on=["SKU_Code", "Year"], how="left")
        report_df = report_df.merge(ul_ss, on=["SKU_Code", "Year"], how="left")
        report_df["Production_Batch"] = np.divide(report_df["Production_Demand"].to_numpy(),
                                                  report_df["Standard_Batch_Size_Box"].to_numpy(),
                                                  where=(report_df["Standard_Batch_Size_Box"] > 0.5))
        report_df["Allocated_Batch"] = np.divide(report_df["Allocated_Box"].to_numpy(),
                                                 report_df["Standard_Batch_Size_Box"].to_numpy(),
                                                 where=(report_df["Standard_Batch_Size_Box"] > 0.5))
        report_df["No_Allocated_Batch"] = np.maximum(report_df["Production_Batch"].to_numpy() -
                                                     report_df["Allocated_Batch"].to_numpy(), 0)
        report_df["No_Allocated_Box"] = np.maximum(report_df["Demand"].to_numpy() -
                                                   report_df["Allocated_Box"].to_numpy(), 0)
        report_df["Allocated_Kilo"] = report_df["Allocated_Batch"] * report_df["Standard_Batch_Size_Kilo"]
        report_df["No_Allocated_Kilo"] = report_df["No_Allocated_Batch"] * report_df["Standard_Batch_Size_Kilo"] - \
                                         report_df["Allocated_Kilo"]
        report_df["Closing_Stock"] = \
            report_df.merge(
                production_df.filter(["SKU_Code", "Closing_stock", "Month"]).rename({"Month": "Year"}, axis=1),
                on=["SKU_Code", "Year"], how="left")["Closing_stock"]
        report_df["Sales_Demand"] = report_df["Demand"]
        report_df["Needed_Stock"] = report_df["Production_Demand"] - report_df["Demand"]
        report_df["Production_Allocation"] = report_df["No_Allocated_Box"]
        report_df["Production_Loss"] = np.maximum((report_df["Production_Demand"] -
                                                   report_df["Production_Allocation"]).to_numpy(), 0)
        report_df["Sales_Loss"] = \
            report_df.merge(production_df.filter(["SKU_Code", "Sales_loss", "Month"]).rename({"Month": "Year"}, axis=1),
                            on=["SKU_Code", "Year"], how="left")["Sales_loss"].astype(float)
        report_df["Sales_Achievement"] = (report_df["Sales_Demand"] - report_df["Sales_Loss"]).astype(float)
        report_df["Xproject_Production_Allocation"] = 0
        temp_array = pd.pivot_table(production_df.filter(["Closing_stock", "Month", "SKU_Code"]),
                                    values="Closing_stock", index="SKU_Code", columns="Month", aggfunc="sum")
        temp_array.iloc[:, 1:] = temp_array.iloc[:, :-1].to_numpy()
        all_gb_data = self.load_csv_data("gb_df.csv")
        opening_inv = all_gb_data[all_gb_data["Month"] == 202301]
        temp_array.iloc[:, 0] = pd.merge(temp_array.filter("SKU_Code"),
                                         opening_inv.filter(["SKU_Code", "Opening_Stock"]),
                                         how="left",
                                         on="SKU_Code")["Opening_Stock"].to_numpy()
        report_df["Opening_Stock"] = report_df.merge(pd.melt(temp_array.reset_index(),
                                                             id_vars=["SKU_Code"], value_name="Opening",
                                                             var_name="Year"),
                                                     how="left",
                                                     on=["SKU_Code", "Year"])["Opening"]
        report_df["Production_Loss_Total"] = report_df["Production_Loss"]
        report_df["Scenario"] = self.SCENARIO
        report_df = report_df[columns]
        # report_df["Flag"] = "M"
        if self.SYNC:
            print("Uploading report dataframe")
            start = time.perf_counter()
            self.upload_to_server(report_df, table_name="LPExportMonth", index="SKU_Code")
            print(f"Done, completion time = {time.perf_counter() - start}")
        machine_box = machine_box.loc[:, ~machine_box.columns.duplicated()].copy()
        machine_list = lp_data.machine_available_time["Machine_Code"].to_list()
        machine_box_melted = pd.melt(machine_box, id_vars=["SKU_Code", "Month"],
                                     value_vars=machine_list)
        machine_box_melted = machine_box_melted[machine_box_melted["value"] != 0]
        machine_box_melted.rename(columns={"SKU_Code": "index", "Month": "Year", "variable": "Machine", "value": "Box"},
                                  inplace=True)
        machine_box_melted["Scenario"] = self.SCENARIO
        # machine_box_melted["Flag"] = "M"
        Machine_Box = machine_box_melted[["index", "Machine", "Box", "Scenario", "Year"]]
        if self.SYNC:
            print("Uploading machine box dataframe")
            self.upload_to_server(Machine_Box, table_name="LPMachineDetailMonth", index="index")
            print("Done")
        return Machine_Box, report_df

    def upload_Machine_report(self, machine_report, lp_data: LPData, machine_box):
        machine_report.reset_index(inplace=True)
        machine_report_melted = pd.melt(machine_report, id_vars=["Month"], value_vars=list(machine_report.columns)[1:])
        machine_report_melted.rename(columns={"variable": "index", "value": 'usedCapacity_hours'}, inplace=True)
        A = pd.melt(lp_data.machine_available_time, id_vars=['Machine_Code', "Site", "Machine_Description", "Process"],
                    value_vars=list(lp_data.machine_available_time.columns[9:]))
        A.rename(columns={"variable": "Month", "value": "Available_time"}, inplace=True)
        A["id"] = A["Machine_Code"] + A["Month"].astype('str')
        A = A[A["Month"].isin(machine_report["Month"].astype(str).to_list())]
        B = lp_data.allocations.filter(items=["Machine_Code", "Sub_Process"]).drop_duplicates(subset="Machine_Code")
        B = B.replace({"Sub_Process" : {"CAP Filling" : "Capsule Filling", "Compression":"Press"}})
        machine_report_melted["id"] = machine_report_melted["index"] + machine_report_melted["Month"].astype('str')
        mergedDF = A.merge(machine_report_melted.drop("Month", axis=1), how='right', left_on="id", right_on="id")
        mergedDF["Process"] = mergedDF.merge(B, how="left", on="Machine_Code")["Sub_Process"]
        # machine_box["year"] = machine_box["Year"].map(lambda x: int(str(x)[:4]))
        machine_box = machine_box.merge(lp_data.product_data.filter(["SKU_Code", "Num_in_Box"]).rename({"SKU_Code":"index"}, axis=1), on="index", how="left")
        machine_box["UsedCapacity_unit"] = machine_box["Num_in_Box"] *  machine_box["Box"]
        machine_box_aggr = machine_box.groupby(["Machine", "Year"], as_index=False).sum()
        machine_box_aggr["Year"] = machine_box_aggr["Year"].astype(str)
        mergedDF["UsedCapacity_box"] = mergedDF.merge(machine_box_aggr, how="left", left_on=["Machine_Code", "Month"],
                                                      right_on=["Machine", "Year"])["Box"]
        mergedDF["UsedCapacity_hours"] = mergedDF["usedCapacity_hours"] * mergedDF["Available_time"]
        mergedDF["UsedCapacity_unit"] = mergedDF.merge(machine_box_aggr, how="left", left_on=["Machine_Code", "Month"],
                                                      right_on=["Machine", "Year"])["UsedCapacity_unit"]
        mergedDF["UsedCapacity_kilo"] = 0
        mergedDF["AverageOutWeight"] = 0
        mergedDF["Work_days"] = 0
        mergedDF["Box_capacity"] = 0
        mergedDF["Scenario"] = self.SCENARIO
        mergedDF.drop(columns=["Machine_Code", "id"], inplace=True)
        mergedDF.rename(columns={"Month": "Year", "Process": "MachineOp"}, inplace=True)
        mergedDF2 = mergedDF[
            ["index", "UsedCapacity_box", "UsedCapacity_unit", "UsedCapacity_hours", "UsedCapacity_kilo",
             "AverageOutWeight", "Year", "Site", "Available_time", "Work_days", "Box_capacity", "Scenario",
             "MachineOp"]]
        # mergedDF2["Flag"] = 0
        if self.SYNC:
            print("Uploading machine time dataframe")
            self.upload_to_server(mergedDF2, table_name="LPMachineMonth", index="index")
            print("Done")
        return mergedDF2
