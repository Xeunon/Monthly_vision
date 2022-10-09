import openpyxl as xl
import pandas as pd
import os


def load_process_input():
    # Load input sheets into objects
    sheets = load_sheets()
    product_df = generate_standard_dataframe(sheets["product_sheet"])
    bom_df = generate_standard_dataframe(sheets["bom_sheet"])
    gb_df = generate_standard_dataframe(sheets["gb_sku"])
    timing_df = generate_timing_dataframe(sheets["timing_sheet"])
    available_time_df = generate_available_time_dataframe(sheets["available_time_sheet"])
    machine_list = available_time_df["Machine_Code"]
    timing_df = timing_df[machine_list] / 60
    allocation_df = generate_standard_dataframe(sheets["timing_sheet"])
    opc_df = generate_standard_dataframe(sheets["opc_sheet"])
    date_df = generate_standard_dataframe(sheets["date_sheet"])
    wip_df = generate_standard_dataframe(sheets["wip_sheet"])
    # weight_df = generate_standard_dataframe(sheets["weight_sheet"])
    holding_df = generate_standard_dataframe(sheets["holding_sheet"])
    opening_df = generate_standard_dataframe(sheets["opening_sheet"])
    order_df = generate_standard_dataframe(sheets["order_sheet"])
    # cheat_df = generate_standard_dataframe(sheets["cheat_sheet"])

    # pd.set_option('display.max_columns', None)
    dir_name = "../data/CSV_Data"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

    def output_dataframe(dataframe: pd.DataFrame, name: str, has_index: bool = True):
        output_path = os.path.join(dir_name, name)
        dataframe.to_csv(output_path, index=has_index)
        return

    output_dataframe(dataframe=date_df, name='date_df.csv', has_index=False)
    output_dataframe(dataframe=product_df, name='product_df.csv', has_index=False)
    output_dataframe(dataframe=bom_df, name='bom_df.csv', has_index=False)
    output_dataframe(dataframe=gb_df, name='gb_df.csv', has_index=False)
    output_dataframe(dataframe=timing_df, name='timing_df.csv', has_index=True)
    output_dataframe(dataframe=available_time_df, name='available_time_df.csv', has_index=False)
    output_dataframe(dataframe=opc_df, name='opc_df.csv', has_index=False)
    output_dataframe(dataframe=allocation_df, name='allocation.csv', has_index=False)
    output_dataframe(dataframe=wip_df, name='wip_df.csv', has_index=False)
    # output_dataframe(dataframe=weight_df, name='weight_df.csv', has_index=False)
    output_dataframe(dataframe=holding_df, name='holding_df.csv', has_index=False)
    output_dataframe(dataframe=opening_df, name='opening_df.csv', has_index=False)
    output_dataframe(dataframe=order_df, name='order_df.csv', has_index=False)
    # output_dataframe(dataframe=cheat_df, name='cheat_df.csv', has_index=False)

    return


def load_sheets():
    # Loading excel file
    wb = xl.load_workbook('../data/Input/MIP_InputExpr.xlsx', data_only=True, read_only=True)
    # Loading each sheet into separate objects
    sheets = {"product_sheet": wb["Product"],
              "date_sheet": wb["DateSeries"],
              "bom_sheet": wb['BOM'],
              "gb_sku": wb["GB-SKU"],
              "timing_sheet": wb["Timing"],
              "available_time_sheet": wb["AvailableTime"],
              "opc_sheet": wb["OPC"],
              "wip_sheet": wb["WIP"],
              # "weight_sheet": wb["Weights"],
              "holding_sheet": wb["Holdingtime"],
              "opening_sheet": wb["Opening"],
              "order_sheet": wb["NewPurchase"],
              # "cheat_sheet": wb["DirectInput"]
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


def generate_available_time_dataframe(sheet):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_", regex=True)
    df["Sub_Process"] = df["Sub_Process"].str.replace(" ", "_", regex=True)
    op_order = ['Wet_Granulation',
                'Drying',
                'Blending',
                'Roller_Compactor',
                'Compression_CAP_Filling',
                'Coating',
                'Blistering__Counter_Syrup_Filling',
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
                        'Blistering': 'Blistering__Counter_Syrup_Filling',
                        'Counter': 'Blistering__Counter_Syrup_Filling',
                        'Syrup_Filling': 'Blistering__Counter_Syrup_Filling',
                        'Out_Sourcing': 'Out_Sourcing'}
    df = df.replace({"Sub_Process": sub_process_dict})
    df["Sub_Process"] = pd.Categorical(df["Sub_Process"],
                                       # reordering operations in the dataframe
                                       categories=op_order,
                                       ordered=True)
    # df.sort_values(["Sub_Process", 'Machine_Code'], inplace=True)
    df.rename(columns={"Sub_Process": "Process"}, inplace=True)
    return df


def generate_timing_dataframe(sheet):
    # Loading product data onto a dataframe
    data = sheet.values
    cols = next(data)  # Getting the first row of product_sheet
    cols = list(map(str, list(cols)))
    df = pd.DataFrame(data, columns=cols)
    df.columns = df.columns.str.replace("[&,/,.,+,']", "", regex=True)
    df.columns = df.columns.str.replace(" ", "_")
    df = df[df["Active"] != 2]
    df["inter_code"] = df["Product_Code"].astype(str) + "-" + df["BOM_Version"].astype(str)
    final_df = pd.pivot_table(data=df, values="CT", index="inter_code", columns="Machine_Code", fill_value=0)
    return final_df


load_process_input()
