from docplex.mp.model import Model
import time
import os
import pandas as pd
from MIP_Pro.LP.LPProblem import LPProblem
from MIP_Pro.LP.LPData import LPData
from MIP_Pro.Data_IO.Data_Import import load_process_input
from MIP_Pro.Data_IO.SolutionHandler import generate_production_report, generate_op_based_report
from MIP_Pro.Data_IO.SolutionHandler import generate_machine_report, generate_raw_material_report
from MIP_Pro.Data_IO.SolutionHandler import generate_package_material_report
from datetime import datetime

load_process_input()

if __name__ == "__main__":
    # Defining the number of periods to solve for
    period_count = 5

    first_month = 140103
    start = time.perf_counter()
    lp_solver = Model(name="MIP")
    lp_data = LPData(first_month, period_count)
    mip_problem = LPProblem(lp_data, lp_solver)
    lp_solver.parameters.mip.tolerances.mipgap = 0.2
    lp_solver.parameters.threads = 16
    # lp_solver.parameters.mip.strategy = 1
    lp_solver.parameters.parallel = 0

    # lp_solver.parameters
    status = lp_solver.solve(log_output=True)

    print("Completion time = {}".format(time.perf_counter() - start))

    production_df, production_box_df, production_unit_df = generate_production_report(lp_data, mip_problem)
    batch_df = generate_op_based_report(lp_data, mip_problem)
    monthly_mps_time, monthly_saturation, monthly_box_output, monthly_unit_output = generate_machine_report(lp_data,
                                                                                                            mip_problem)
    rm_closing, rm_consumption = generate_raw_material_report(lp_data, mip_problem)
    pm_closing, pm_consumption = generate_package_material_report(lp_data, mip_problem)

    datetimeobj = datetime.now()
    timestamp = datetimeobj.strftime("%b-%d-%Y(%H-%M)")
    save_dir = "../data/MIP Output"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    folder_name = os.path.join(save_dir, timestamp)
    os.mkdir(folder_name)


    def output_dataframe(dataframe: pd.DataFrame, name: str, has_index: bool = True):
        output_path = os.path.join(folder_name, name+".csv")
        dataframe.to_csv(output_path, index=has_index)
        return


    saved = False
    while not saved:
        try:
            output_dataframe(production_df, "Production_report")
            output_dataframe(monthly_mps_time, "MPS_time")
            output_dataframe(batch_df, "Batch_report")
        except:
            print("Saving failed")
            input("Press enter to try again")
            continue
        saved = True
