import numpy as np
import openpyxl as xl
import pandas as pd
import os


def load_process_input():
    # Load input sheets into objects
    sheets = load_sheets()
    product_df = generate_standard_dataframe(sheets["product_sheet"])
    gb_df = generate_gb_dataframe(sheets["gb_sku"], product_df)
    timing_df = generate_timing_dataframe(sheets["timing_sheet"])
    available_time_df = generate_available_time_dataframe(sheets["available_time_sheet"], timing_df)
    machine_list = available_time_df["Machine_Code"]
    # Sorting timing_df columns (machines) to match available_time machine order
    timing_df = timing_df[machine_list] / 60
    allocation_df = generate_standard_dataframe(sheets["timing_sheet"])
    opc_df = generate_opc_dataframe(product_df)
    wip_df = generate_standard_dataframe(sheets["wip_sheet"])
    holding_df = generate_standard_dataframe(sheets["holding_sheet"])

    # pd.set_option('display.max_columns', None)
    dir_name = "../data/CSV_Data"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    def output_dataframe(dataframe: pd.DataFrame, name: str, has_index: bool = True):
        output_path = os.path.join(dir_name, name)
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

    return


def load_sheets():
    # Loading excel file
    wb = xl.load_workbook('../data/Input/MIP_InputExpr.xlsx', data_only=True, read_only=True)
    # Loading each sheet into separate objects
    sheets = {"product_sheet": wb["Product"],
              "gb_sku": wb["GB-SKU"],
              "timing_sheet": wb["Allocations"],
              "available_time_sheet": wb["AvailableTime"],
              "opc_sheet": wb["OPC"],
              "wip_sheet": wb["WIP"],
              "holding_sheet": wb["Holdingtime"],
              }

    return sheets


def generate_standard_dataframe(sheet):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")
    return df


def generate_opc_dataframe(product_df):
    # Loading product data onto a dataframe
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
    index = product_df["SKU_Code"].astype(str) + "-" + product_df["bom_version"].astype(str)
    opc_df = pd.DataFrame(data=False, index=index, columns=op_order)
    prod_method = product_df.filter(["Method", "Coated", "Manually_Packaged"])
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
    opc_df.insert(loc=0, column="Bom_Version", value=product_df["bom_version"].to_numpy())
    opc_df.insert(loc=0, column="SKU_Code", value=product_df["SKU_Code"].to_numpy())
    return opc_df


def generate_gb_dataframe(sheet, product_df):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")

    def offset_limits(lim_name, out_name):
        limit_df = pd.pivot_table(data=df.filter(["Month", "Total_Demand", "SKU_Code"]),
                                  values="Total_Demand",
                                  index="SKU_Code",
                                  columns="Month").fillna(0)
        for index, row in limit_df.iterrows():
            lim = float(product_df[product_df["SKU_Code"] == index].drop_duplicates("SKU_Code")[lim_name])
            limit_df.loc[index][:-1] = limit_df.loc[index][1:].to_numpy()
            for i in range(1, 5):
                if i >= int(lim):
                    limit_df.loc[index][:-i] = limit_df.loc[index][:-i].to_numpy() + (
                                limit_df.loc[index][i:] * (lim % 1)).to_numpy(dtype=np.int64)
                    break
                else:
                    limit_df.loc[index][:-i] = limit_df.loc[index][:-i].to_numpy() + limit_df.loc[index][i:].to_numpy()
        limit_df.reset_index(inplace=True)
        limit_df = pd.melt(limit_df, id_vars="SKU_Code", value_name=out_name)
        return limit_df

    SL_limit_df = offset_limits("SL_LIM", "SL_Limit")
    LL_limit_df = offset_limits("LL_SS", "FG_LL_SS")
    ML_limit_df = offset_limits("ML_SS", "FG_ML_SS")
    UL_limit_df = offset_limits("UL_SS", "FG_UL_SS")
    DM_offset_df = offset_limits("Dmd_LIM", "DM_OFFSET")
    df = pd.merge(df, SL_limit_df, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    df = pd.merge(df, LL_limit_df, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    df = pd.merge(df, ML_limit_df, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    df = pd.merge(df, UL_limit_df, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    df = pd.merge(df, DM_offset_df, how="left", left_on=["SKU_Code", "Month"], right_on=["SKU_Code", "Month"])
    # df = pd.merge(df, product_df.filter(["SKU_Code", "LL_SS", "ML_SS", "UL_SS"]).drop_duplicates(subset=["SKU_Code"]),
    #               how="left", left_on="SKU_Code", right_on="SKU_Code")
    # df["FG_LL_SS"] = df["SL_Limit"] * df["LL_SS"]
    # df["FG_ML_SS"] = df["SL_Limit"] * df["ML_SS"]
    # df["FG_UL_SS"] = df["SL_Limit"] * df["UL_SS"]
    return df


def generate_available_time_dataframe(sheet, timing_df):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_", regex=True)
    df["Sub_Process"] = df["Sub_Process"].str.replace(" ", "_", regex=True)
    # Getting common machines between allocations and available_time data
    common_machines = set(timing_df.columns).intersection(set(df["Machine_Code"]))
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
    df = df.replace({"Sub_Process": sub_process_dict})
    df["Sub_Process"] = pd.Categorical(df["Sub_Process"],
                                       # reordering operations in the dataframe
                                       categories=op_order,
                                       ordered=True)
    # df.sort_values(["Sub_Process", 'Machine_Code'], inplace=True)
    df.rename(columns={"Sub_Process": "Process"}, inplace=True)
    # Finding the first occurence of a month number in the columns
    for first_month_col, x in enumerate(df.columns):
        if x.isdigit():
            break
    # Converting workdays to hours
    loading_time = df.iloc[:, first_month_col:] * 24 * df['Saturation'].to_numpy()[:, np.newaxis]
    train_rest = df.iloc[:, first_month_col:] * df['Resting'].to_numpy()[:, np.newaxis] \
                 + df['Training'].to_numpy()[:, np.newaxis]
    pm_spc = loading_time * (df['SPC'].to_numpy()[:, np.newaxis] + df['PM'].to_numpy()[:, np.newaxis])
    df.iloc[:, first_month_col:] = np.maximum(loading_time - train_rest - pm_spc, 0)
    return df


def generate_timing_dataframe(sheet):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")
    df["inter_code"] = df["SKU_Code"].astype(str) + "-" + df["BOM_Version"].astype(str)
    df["with_site"] = df["inter_code"] + "-" + df["Site"].astype(str)
    final_df = pd.pivot_table(data=df, values="CT", index="with_site", columns="Machine_Code", fill_value=0)
    # final_df.reset_index(inplace=True)
    # final_df_p = pd.merge(final_df, df.filter(["with_site", "inter_code"]).drop_duplicates(),
    #                       how="right", left_on="with_site", right_on="with_site", sort=False)
    return final_df
