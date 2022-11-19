from docplex.mp.model import Model
import time
import os
import pandas as pd
import numpy as np
from datetime import datetime
from MIP_Pro.LP.LPProblem import LPProblem
from MIP_Pro.LP.LPData import LPData
from MIP_Pro.Data_IO.Data_Import import load_process_input
from MIP_Pro.Data_IO.SolutionHandler import generate_production_report, generate_op_based_report
from MIP_Pro.Data_IO.SolutionHandler import generate_machine_report


def output_dataframe(dataframe: pd.DataFrame, name: str, has_index: bool = True):
    global timestamp
    save_dir = "../data/MIP Output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    folder_name = os.path.join(save_dir, timestamp)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    output_path = os.path.join(folder_name, name + ".csv")
    dataframe.to_csv(output_path, index=has_index)
    return


def create_data_series(years: list):
    series_dict = {}
    i = 1
    years.append(years[-1] + 1)
    for year in years:
        for month in range(1, 13):
            str_month = '0' + str(month) if int(month / 10) == 0 else str(month)
            series_dict[i] = int(str(year) + str_month)
            i += 1
    series_df = pd.DataFrame(series_dict.items(), columns=["series", "month"])
    dir_name = "../data/CSV_Data"
    output_path = os.path.join(dir_name, 'date_df.csv')
    series_df.to_csv(output_path, index=False)
    return series_dict


def update_openings(new_wip_df, production_df, first_month):
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

    wip_df = load_csv_data("wip_df.csv")
    gb_df = load_csv_data("gb_df.csv")
    new_wip_df.insert(loc=0, column="inter_code", value=new_wip_df.index.map(lambda x: x[:-2]))
    new_wip_df.insert(loc=0, column="Site", value=new_wip_df.index.map(lambda x: x[-1:]))
    new_wip_df.insert(loc=0, column="code", value=new_wip_df["inter_code"].map(lambda x: x[:7]))
    new_wip_df.insert(loc=0, column="BOM_Version", value=new_wip_df["inter_code"].map(lambda x: x[8:]))
    new_wip_df.insert(loc=new_wip_df.columns.get_loc("Wet_Granulation"),
                      column="Material", value=np.max(new_wip_df.iloc[:, -8:].to_numpy(), axis=1))
    new_wip_df.drop(labels="inter_code", inplace=True, axis=1)
    gb_df.loc[gb_df["Month"] == first_month]
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


def consecutive_run(sites: list, years: list):
    series_dict = create_data_series(years)
    first_month_index = 1
    run_length = 4
    production_report = []
    machine_report = []
    machine_box_report = []
    while True:
        first_month = series_dict[first_month_index]
        lp_solver = Model(name="MIP")
        lp_data = LPData(first_month, run_length, sites)
        mip_problem = LPProblem(lp_data, lp_solver)
        lp_solver.parameters.mip.tolerances.mipgap = 0.2
        lp_solver.parameters.threads = 32
        # lp_solver.parameters.mip.strategy = 1
        lp_solver.parameters.parallel = 0
        status = lp_solver.solve(log_output=True)
        production_df, production_box_df = generate_production_report(lp_data, mip_problem)
        # output_dataframe(production_box_df,
        #                  name=f"Production report{lp_data.solve_window_dates[0]}-{lp_data.solve_window_dates[-1]}")
        production_report.append(production_box_df)
        batch_df, wip_df = generate_op_based_report(lp_data, mip_problem)
        monthly_mps_time, monthly_saturation, \
        monthly_box_output = generate_machine_report(lp_data, mip_problem)
        machine_report.append(monthly_saturation)
        machine_box_report.append(monthly_box_output)
        first_month_index = first_month_index + run_length
        if first_month_index < list(series_dict.keys())[-12]:
            update_openings(wip_df, production_box_df, series_dict[first_month_index])
        else:
            production_report = pd.concat(production_report)
            machine_report = pd.concat(machine_report)
            machine_box_report = pd.concat(machine_box_report)
            output_dataframe(production_report, name=f"Production report")
            output_dataframe(machine_report, name=f"Machine utilization report")
            output_dataframe(machine_box_report, name=f"Machine box report")
            break


if __name__ == "__main__":
    # Loading all the input data from excel and saving them in csv files to later be accessed
    load_process_input()
    date_time = datetime.now()  # Saving the time that the code was executed to later be used in output generation
    timestamp = date_time.strftime("%b-%d-%Y(%H-%M)")
    start = time.perf_counter()  # Saving the exact time the code was run to later be used in runtime measurements
    consecutive_run(sites=[1, 2],
                    years=[2023])  # The main function that is tasked with running the model consecutively
    print("Completion time = {}".format(time.perf_counter() - start))
