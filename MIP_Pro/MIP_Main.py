from docplex.mp.model import Model
import time
import os
import pandas as pd
import numpy as np
from MIP_Pro.LP.LPProblem import LPProblem
from MIP_Pro.LP.LPData import LPData
from MIP_Pro.Data_IO.Data_Import import load_process_input
from MIP_Pro.Data_IO.SolutionHandler import SolutionHandler
from datetime import datetime


def create_data_series(years: list):
    series_dict = {}
    # Determine if input is in years or in months
    if int(years[0] / 10000) == 0:
        # Input is in years !!!
        i = 1
        years.append(years[-1] + 1)
        for year in range(years[0], years[-1] + 1):
            for month in range(1, 13):
                str_month = '0' + str(month) if int(month / 10) == 0 else str(month)
                series_dict[i] = int(str(year) + str_month)
                i += 1
        series_df = pd.DataFrame(series_dict.items(), columns=["series", "month"])
    else:
        # Input is in Months !!!
        i = 1
        first_month = years[0]
        last_month = years[-1] + 2 if (years[-1] + 2) % 100 <= 12 else years[-1] + 100 - 10
        month = first_month
        while True:
            series_dict[i] = month
            month = month + 1 if (month + 1) % 100 <= 12 else month + 100 - 11
            if month >= last_month:
                i += 1
                series_dict[i] = month
                break
            i += 1
        series_df = pd.DataFrame(series_dict.items(), columns=["series", "month"])
    dir_name = "../data/CSV_Data"
    output_path = os.path.join(dir_name, 'date_df.csv')
    series_df.to_csv(output_path, index=False)
    return series_dict


def update_openings(new_wip_df, production_df, first_month, material_closing):
    def load_csv_data(name, index_col=None):
        """Loading model input data from csv files placed in the CSV_DATA folder"""
        data_directory = "..\\data\\CSV_Data"
        return pd.read_csv(os.path.join(data_directory, name), index_col=index_col)

    def output_df(dataframe: pd.DataFrame, name: str, has_index: bool = True):
        dir_name = "../data/CSV_Data"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        output_path = os.path.join(dir_name, name)
        dataframe.to_csv(output_path, index=has_index)
        return

    try:
        material_df = load_csv_data("material_df.csv")
        # TODO: Current fix is flawed and should be addressed in the case of materials that are both RM and PM
        # material_df.drop_duplicates(subset=["Code"], inplace=True)
        material_closing = material_closing[~material_closing.index.duplicated(keep='first')]
        material_closing = material_closing.iloc[:, -1]
        new_material_opening = material_df.filter(items=["Code"]).merge(material_closing,
                                                                        how="left",
                                                                        left_on="Code",
                                                                        right_index=True).fillna(0)
        material_df.loc[:, "Opening"] = new_material_opening.iloc[:, -1].to_numpy()
        output_df(dataframe=material_df, name='material_df.csv', has_index=False)
    except FileNotFoundError:
        pass
    gb_df = load_csv_data("gb_df.csv")
    new_wip_df.insert(loc=0, column="bom_sku", value=new_wip_df.index.map(lambda x: x[:-2]))
    new_wip_df.insert(loc=0, column="Site", value=new_wip_df.index.map(lambda x: x[-1:]))
    new_wip_df.insert(loc=0, column="code", value=new_wip_df["bom_sku"].map(lambda x: x[:7]))
    new_wip_df.insert(loc=0, column="BOM_Version", value=new_wip_df["bom_sku"].map(lambda x: x[8:]))
    new_wip_df.insert(loc=new_wip_df.columns.get_loc("Wet_Granulation"),
                      column="Material", value=np.max(new_wip_df.iloc[:, -8:].to_numpy(), axis=1))
    new_wip_df.drop(labels="bom_sku", inplace=True, axis=1)
    last_month = production_df["Month"].to_numpy().max()
    closing_df = production_df[production_df["Month"] == last_month].loc[:, ["Closing_stock"]]
    closing_df.insert(loc=0, column="SKU_Code", value=pd.unique(production_df["SKU_Code"]))
    # closing_df.drop_duplicates(subset="SKU_Code", inplace=True)
    closing_df.insert(loc=0, column="Month", value=first_month)
    gb_df = pd.merge(gb_df, closing_df, how="left", on=["SKU_Code", "Month"])
    gb_df.loc[gb_df["Month"] == first_month, "Opening_Stock"] = gb_df.loc[
        gb_df["Month"] == first_month, "Closing_stock"]
    gb_df.drop(labels="Closing_stock", inplace=True, axis=1)
    output_df(dataframe=gb_df, name='gb_df.csv', has_index=False)
    output_df(dataframe=new_wip_df, name='wip_df.csv', has_index=False)

    return


def consecutive_run(sites: list, years: list, solution_handler):
    """
    Runs the monthly MPS problem in the form of multiple consecutive models.

    Parameters
    ----------
    sites:
        Currently unused
    years:
        A list that shows the from-to periods of the problem either in months or years.\n
        For instance, [2023, 2025] states that we want to solve the problem from the first month of 2023 to the last
        month of 2025. [202302, 202309] states that we want to solve the problem from the second month of 2023 to the
        ninth month of 2023 (inclusive).
    :return:
    """
    series_dict = create_data_series(years)
    first_month_index = 1
    run_length = 3
    production_report = []
    machine_report = []
    machine_box_report = []
    while True:
        first_month = series_dict[first_month_index]
        lp_model = Model(name="MIP")
        lp_data = LPData(first_month, run_length, sites, series_dict)
        mip_problem = LPProblem(lp_data, lp_model)
        lp_model.parameters.mip.tolerances.mipgap = 0.2
        lp_model.parameters.threads = 6
        # lp_model.parameters.mip.strategy = 1
        lp_model.parameters.parallel = 0
        lp_model.solve(log_output=True)
        production_df, production_box_df = solution_handler.generate_production_report(lp_data, mip_problem)
        # output_dataframe(production_box_df,
        #                  name=f"Production report{lp_data.solve_window_dates[0]}-{lp_data.solve_window_dates[-1]}")
        production_report.append(production_box_df)
        batch_df, wip_df = solution_handler.generate_op_based_report(lp_data, mip_problem)
        monthly_mps_time, monthly_saturation, \
        monthly_box_output = solution_handler.generate_machine_report(lp_data, mip_problem)
        try:
            rm_closing_df, rm_consumption_df = solution_handler.generate_raw_material_report(lp_data, mip_problem)
            pm_closing_df, pm_consumption_df = solution_handler.generate_package_material_report(lp_data, mip_problem)
            material_closing = pd.concat([rm_closing_df, pm_closing_df])
        except:
            material_closing = None
            pass
        machine_report.append(monthly_saturation)
        machine_box_report.append(monthly_box_output)
        first_month_index = first_month_index + run_length
        if first_month_index <= list(series_dict.keys())[-2]:
            update_openings(wip_df, production_box_df, series_dict[first_month_index], material_closing)
        else:
            production_report = pd.concat(production_report)
            machine_report = pd.concat(machine_report)
            machine_box_report = pd.concat(machine_box_report)
            solution_handler.output_dataframe(production_report, name=f"Production report")
            machine_Box, report_df = solution_handler.upload_production_data(production_report, lp_data)
            machine_month = solution_handler.upload_Machine_report(machine_report, lp_data, machine_Box)
            solution_handler.output_dataframe(report_df, name=f"LPExportMonth")
            solution_handler.output_dataframe(machine_Box, name=f"LPMachineDetailMonth")
            solution_handler.output_dataframe(machine_month, name=f"LPMachineMonth")
            break


if __name__ == "__main__":
    # Defining the number of periods to solve for
    datetimeobj = datetime.now()
    timestamp = datetimeobj.strftime("%b-%d-%Y(%H-%M)")
    load_process_input(load_materials=False, load_campaign=False)
    start = time.perf_counter()
    # ******************** MAKE SURE NOT TO SYNC WHEN EXPERIMENTING ************************
    sh = SolutionHandler(scenario="Scenario-integer-WithOutSource", sync=False)
    consecutive_run(sites=[1, 2],
                    years=[202301, 202309],  # TODO: If given dates are fewer than run duration, the program will fail.
                    solution_handler=sh)
    print(f"Completion time = {time.perf_counter() - start}")
