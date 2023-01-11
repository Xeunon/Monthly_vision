import numpy as np
import openpyxl as xl
import pandas as pd
import os
import time
import glob
import multiprocessing

INPUT_EXCEL_NAME = "Monthly_Vision"
DIR_NAME = "../data/CSV_Data"
USE_CACHE = False

PRODUCT_SHEET = "Product"
MATERIAL_SHEET = "Material"
BOM_SHEET = "BOM"
WIP_SHEET = "WIP"
HOLDING_SHEET = "Holdingtime"
GB_SHEET = "GB-SKU"
AVAILABLE_TIME_SHEET = "AvailableTime"
CAMPAIGN_SHEET = "Campaign"
ALLOCATION_SHEET = "Allocations"

sheets = dict()


def load_process_input(load_materials: bool, load_campaign: bool) -> None:
    """Reads the input Excel file stored in data/Input and transforms and stores the data into separate csv files
       which will later be utilized by the LPData class

       Parameters
       ----------
       load_materials:
           boolean parameter indicating whether material data should be loaded and be considered as a constraint.
       load_campaign:
           boolean parameter indicating whether campaign production data should be loaded and be considered
           as a constraint.
       :return: None
       """
    input_mtime = os.path.getmtime(f'../data/Input/{INPUT_EXCEL_NAME}.xlsx')
    if USE_CACHE:
        try:
            info_df = pd.read_csv(os.path.join(DIR_NAME, "info.csv"), index_col=0).iloc[0].to_dict()
            if input_mtime == info_df["time"] \
                    and load_materials == info_df["material"] \
                    and load_materials == info_df["campaign"]:
                print("Input data has not changed since last run; will use previously imported data.")
                return
        except FileNotFoundError:
            pass
    global sheets
    sheets = load_sheets()
    product_df = dataframe_from_sheet(sheet_name="product_sheet")
    allocation_df = dataframe_from_sheet(sheet_name="timing_sheet")
    wip_df = dataframe_from_sheet(sheet_name="wip_sheet")
    holding_df = dataframe_from_sheet(sheet_name="holding_sheet")
    gb_df = greenband_df_gen()
    timing_df = timing_df_gen()
    available_time_df = available_time_df_gen(timing_df)
    machine_list = available_time_df["Machine_Code"]
    # Sorting timing_df columns (machines) to match available_time machine order
    timing_df = timing_df[machine_list] / 60
    opc_df = generate_opc_dataframe(product_df)
    # Deleting the contents of CSV_Data

    if not os.path.exists(DIR_NAME):
        os.mkdir(DIR_NAME)
    else:
        files = glob.glob(DIR_NAME + "/*.csv")
        for f in files:
            os.remove(f)

    def output_dataframe(dataframe: pd.DataFrame, name: str, has_index: bool = True):
        output_path = os.path.join(DIR_NAME, name)
        dataframe.to_csv(output_path, index=has_index)
        return

    output_dataframe(dataframe=product_df, name='product_df.csv', has_index=False)
    output_dataframe(dataframe=gb_df, name='gb_df.csv', has_index=False)
    output_dataframe(dataframe=timing_df, name='timing_df.csv', has_index=True)
    output_dataframe(dataframe=available_time_df, name='available_time_df.csv', has_index=False)
    output_dataframe(dataframe=opc_df, name='opc_df.csv', has_index=True)
    output_dataframe(dataframe=allocation_df, name='allocation.csv', has_index=False)
    output_dataframe(dataframe=wip_df, name='wip_df.csv', has_index=False)
    output_dataframe(dataframe=holding_df, name='holding_df.csv', has_index=False)
    if load_materials:
        bom_df = dataframe_from_sheet(sheets["bom_sheet"])
        material_df = dataframe_from_sheet(sheets["material_sheet"])
        output_dataframe(dataframe=bom_df, name='bom_df.csv', has_index=False)
        output_dataframe(dataframe=material_df, name='material_df.csv', has_index=False)
    if load_campaign:
        campaign_df = dataframe_from_sheet(sheets["campaign_sheet"])
        output_dataframe(dataframe=campaign_df, name='campaign_df.csv', has_index=False)
    info_dict = {"time": input_mtime, "material": load_materials, "campaign": load_campaign}
    info_df = pd.DataFrame(info_dict, index=[0])
    output_dataframe(info_df, name="info.csv")
    return


def load_sheets() -> dict:
    """Function for reading the input Excel file and loading each sheet into a dict object and returning it."""
    global sheets
    wb = xl.load_workbook(f'../data/Input/{INPUT_EXCEL_NAME}.xlsx', data_only=True, read_only=True)
    # Loading each sheet into separate objects
    sheets = {
        "product_sheet": wb[PRODUCT_SHEET],
        "gb_sku": wb[GB_SHEET],
        "timing_sheet": wb[ALLOCATION_SHEET],
        "available_time_sheet": wb[AVAILABLE_TIME_SHEET],
        "wip_sheet": wb[WIP_SHEET],
        "holding_sheet": wb[HOLDING_SHEET],
    }
    try:
        sheets["bom_sheet"] = wb[BOM_SHEET]
        sheets["material_sheet"] = wb[MATERIAL_SHEET]
    except KeyError:
        try:
            sheets["campaign_sheet"] = wb[CAMPAIGN_SHEET]
        except KeyError:
            pass

    return sheets


def dataframe_from_sheet(sheet_name: str) -> pd.DataFrame:
    """Generating a dataframe from an Excel sheet and standardizing column names in the process; removing signs from
       their names and replacing spaces with underlines.

       Parameters
       -----------
       sheet:
            An Openpyxl sheet object
       :return: Pandas Dataframe object
       """
    data = sheets[sheet_name].values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")
    return df


def generate_opc_dataframe(product_df) -> pd.DataFrame:
    """Generating OPC dataframe based on production method of each product supplied in the product sheet.

    Parameters
    ----------
    product_df:
        Dataframe containing product data, specifically production method.
    :return: Dataframe containing OPC data for each product
    """

    op_order = [
        'Wet_Granulation',
        'Drying',
        'Blending',
        'Roller_Compactor',
        'Compression_CAP_Filling',
        'Coating',
        'Blistering_Counter_Syrup_Filling',
        'Manual_Packaging',
    ]
    bom_sku = product_df["SKU_Code"].astype(str) + "-" + product_df["bom_version"].astype(str)
    opc_df = pd.DataFrame(data=False, index=bom_sku, columns=op_order)
    prod_method = product_df.filter(["Method", "Coated", "Manually_Packaged"])
    prod_method["Method"] = prod_method["Method"].str.upper()
    # \/ \/ \/ \/ \/ \/ \/ \/ \/ \/ This is where the opc matrix is constructed \/ \/ \/ \/ \/ \/ \/ \/ \/ \/
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'WG'].index] = [1, 1, 1, 0, 1, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'DG'].index] = [1, 0, 1, 1, 1, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'DC'].index] = [0, 0, 1, 0, 1, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'DG+'].index] = [0, 0, 1, 1, 1, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'DC+'].index] = [0, 0, 0, 0, 1, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == '[NA]'].index] = [1, 0, 0, 0, 0, 0, 1, 0]
    opc_df.iloc[prod_method.loc[prod_method["Method"] == 'FG'].index] = [0, 0, 0, 0, 0, 0, 0, 1]
    opc_df.iloc[prod_method.loc[(prod_method["Coated"] == 'Coated') |
                                (prod_method["Coated"] == 'Double Layer')].index, 5] = 1
    opc_df.iloc[prod_method.loc[prod_method["Manually_Packaged"] == 1].index, 7] = 1
    # /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\
    opc_df.insert(loc=0, column="Bom_Version", value=product_df["bom_version"].to_numpy())
    opc_df.insert(loc=0, column="SKU_Code", value=product_df["SKU_Code"].to_numpy())
    return opc_df


def offset_limits(df, product_df, lim_name, out_name):
    limit_df = pd.pivot_table(data=df.filter(["Month", "Total_Demand", "SKU_Code"]),
                              values="Total_Demand",
                              index="SKU_Code",
                              columns="Month").fillna(0)
    for index, row in limit_df.iterrows():
        lim = float(product_df[product_df["SKU_Code"] == index].drop_duplicates("SKU_Code")[lim_name])
        temp_vec = np.zeros_like(limit_df.loc[index].to_numpy())
        for i in range(1, 24):  # 24 is just an arbitrarily large number
            if i > int(lim):
                temp_vec[:-i] = temp_vec[:-i] + (limit_df.loc[index][i:].to_numpy() * (lim % 1))
                temp_vec[-int(lim) - 1:] = temp_vec[-int(lim) - 2]
                limit_df.loc[index] = temp_vec
                break
            else:
                temp_vec[:-i] = temp_vec[:-i] + limit_df.loc[index][i:].to_numpy()
    limit_df.reset_index(inplace=True)
    limit_df = pd.melt(limit_df, id_vars="SKU_Code", value_name=out_name)
    return limit_df


def greenband_df_gen() -> pd.DataFrame:
    out_names = ["SL_Limit", "FG_LL_SS", "FG_ML_SS", "FG_UL_SS", "DM_OFFSET"]
    lim_names = ["SL_LIM", "LL_SS", "ML_SS", "UL_SS", "Dmd_LIM"]
    df = dataframe_from_sheet("gb_sku")
    product_df = dataframe_from_sheet("product_sheet")
    args = list(zip([df] * len(lim_names), [product_df] * len(lim_names), lim_names, out_names))
    start = time.perf_counter()
    with multiprocessing.Pool(processes=5) as pool:
        output = pool.starmap(offset_limits, args)
    print(f"Completion time = {time.perf_counter() - start}")
    df = dataframe_from_sheet("gb_sku")
    for dataframe in output:
        df = pd.merge(df, dataframe, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    return df


def available_time_df_gen(timing_df) -> pd.DataFrame:
    """Generating machine available time (hours) dataframe based on work days from the AvailableTime sheet of the input
    and removing machines that are not available in the allocations.
    Also mapping machine subprocesses to OPC process names; e.g. 'Blistering' to 'Blistering_Counter_Syrup_Filling'.

    Parameters
    ----------
    sheet:
        AvalilableTime sheet from the Excel input.
    timing_df:
        Dataframe containing allocations data.
    :return: Dataframe containing OPC data for each product
    """
    df = dataframe_from_sheet("available_time_sheet")
    df["Sub_Process"] = df["Sub_Process"].str.replace(" ", "_", regex=True)
    # Getting common machines between allocations and available_time data
    common_machines = set(timing_df.columns).intersection(set(df["Machine_Code"]))
    # Removing machines that are not common with allocations data
    df = df[df["Machine_Code"].isin(common_machines)]
    op_order = ['Wet_Granulation',
                'Drying',
                'Blending',
                'Roller_Compactor',
                'Compression_CAP_Filling',
                'Coating',
                'Blistering_Counter_Syrup_Filling',
                'Manual_Packaging',
                'Out_Sourcing']
    sub_process_dict = {'Wet_Granulation': 'Wet_Granulation',
                        'Syrup_Mixing': 'Wet_Granulation',
                        'Drying': 'Drying',
                        'Blending': 'Blending',
                        'Roller_Compactor': 'Roller_Compactor',
                        'Compression': 'Compression_CAP_Filling',
                        'CAP_Filling': 'Compression_CAP_Filling',
                        'Coating': 'Coating',
                        'Blistering': 'Blistering_Counter_Syrup_Filling',
                        'Counter': 'Blistering_Counter_Syrup_Filling',
                        'Tube_Filling': 'Blistering_Counter_Syrup_Filling',
                        'Syrup_Filling': 'Blistering_Counter_Syrup_Filling',
                        'Manual_Packaging': 'Manual_Packaging',
                        'Out_Sourcing': 'Out_Sourcing'}
    df.insert(loc=4, column="Process", value=df["Sub_Process"])
    # Assertion to make sure all given sub_process names are available in the above dictionary
    assert not set(df["Process"].unique()) - set(sub_process_dict.keys()), \
        f"There are invalid sub_process names in " \
        f"the available time sheet: " \
        f"{set(df['Process'].unique()) - set(sub_process_dict.keys())}"
    df = df.replace({"Process": sub_process_dict})
    # Reordering operations in the dataframe
    df["Process"] = pd.Categorical(df["Process"],
                                   categories=op_order,
                                   ordered=True)
    # Finding the first occurrence of a month number in the columns
    first_month_col = None
    for _, x in enumerate(df.columns):
        if x.isdigit():
            first_month_col = _
            break
    # Converting workdays to hours
    loading_time = df.iloc[:, first_month_col:] * 24 * df['Saturation'].to_numpy()[:, np.newaxis]
    train_rest = df.iloc[:, first_month_col:] * df['Resting'].to_numpy()[:, np.newaxis] \
                 + df['Training'].to_numpy()[:, np.newaxis]
    pm_spc = loading_time * (df['SPC'].to_numpy()[:, np.newaxis] + df['PM'].to_numpy()[:, np.newaxis])
    df.iloc[:, first_month_col:] = np.maximum(loading_time - train_rest - pm_spc, 0)
    return df


def timing_df_gen() -> pd.DataFrame:
    df = dataframe_from_sheet("timing_sheet")
    df["bom_sku"] = df["SKU_Code"].astype(str) + "-" + df["BOM_Version"].astype(str)
    df["with_site"] = df["bom_sku"] + "-" + df["Type"].astype(str)
    final_df = pd.pivot_table(data=df, values="CT", index="with_site", columns="Machine_Code", fill_value=0)
    return final_df
