import numpy as np
import pandas as pd
import os


class LPData:
    """Input data class designed to handle all forms of needed data acquirement and transformation
       in order for them to be fed into the optimization model"""

    def __init__(self, first_month, run_duration, sites, all_dates):
        self.problem_periods = all_dates
        self.future_length = 1  # Set by the user
        self.first_month = first_month
        self.run_duration = run_duration
        # All data related to the solve window
        self.all_period_dates, self.solve_window_dates, self.future_date = self._calculate_run_period_details()
        # All data related to the present products during the solve window
        self.product_data, self.prod_active_window, self.future_active_window = self._generate_active_prod_data()
        # Green-band data ordered to match present products
        self.demands, self.ftr_demand_mat, self.sales_loss_lim, \
        self.ll_ss, self.ul_ss, self.future_ul_ss, self.future_ll_ss, self.dm_offset = \
            self._generate_gb_data()
        # All data related to the orders of operation for each product
        self.opc_data = self._generate_operations_data(name="opc_df.csv")
        self.holding_data = self._generate_operations_data(name="holding_df.csv")
        # Cycle-time data and available time of each machine
        self.machine_available_time, self.timing_data, \
        self.prod_allocations, self.mask, self.allocations = self._generate_allocation_data()
        # TODO: Updating product_data to incorporate sites into it (WARNING)
        self._sync_product_allocation()
        # All data related to material opening stock and arrival schedule
        try:
            self.material_opening, self.material_arrival, self.bom_data = self._generate_material_data()
        except FileNotFoundError:
            pass
        try:
            self.campaign_data = self._generate_campaign_data()
        except FileNotFoundError:
            pass
        # All extra weighting data
        # Opening Work In Progress data
        self.opening_wip, self.wip_tensor = self._generate_wip_data()
        self.farvardin, self.farvardin_f, self.strategic_tier, self.monthly, self.monthly_f = \
            self._generate_weights()

    @staticmethod
    def load_csv_data(name, index_col=None):
        """Loading model input data from csv files placed in the CSV_DATA folder"""
        data_directory = "..\\data\\CSV_Data"
        return pd.read_csv(os.path.join(data_directory, name), index_col=index_col)

    def _calculate_run_period_details(self):
        """Calculating run period dates and month indices based on first month and run duration"""
        time_data = self.load_csv_data("date_df.csv")
        first_month_ndx = time_data[time_data["month"] == self.first_month].iloc[0, 0]
        last_month_ndx = first_month_ndx + (self.run_duration - 1) + self.future_length
        all_period_dates = time_data[time_data["series"].isin(range(first_month_ndx, last_month_ndx + 1))][
            "month"].to_numpy()
        solve_window_dates = all_period_dates[:-self.future_length]
        future_date = all_period_dates[-self.future_length:]
        return all_period_dates, solve_window_dates, future_date

    def _sync_product_allocation(self):
        product_data, active_window, ftr_window = self.product_data, self.prod_active_window, self.future_active_window
        active_window = pd.DataFrame(index=product_data["bom_sku"], data=active_window)
        ftr_window = pd.DataFrame(index=product_data["bom_sku"], data=ftr_window)
        with_site_code = self.timing_data.copy()
        with_site_code.insert(loc=0, column="bom_sku", value=with_site_code.index.map(lambda x: x[:-2]))
        with_site_code.insert(loc=0, column="with_site", value=with_site_code.index)
        self.product_data = pd.merge(product_data, with_site_code.filter(["bom_sku", "with_site"]),
                                     how="right", on="bom_sku")
        allocations = self.load_csv_data("allocation.csv")
        allocations.insert(loc=0, column="with_site",
                           value=allocations["SKU_Code"].astype(str) + "-"
                                 + allocations["BOM_Version"].astype(str) + "-"
                                 + allocations["Type"].astype(str))
        self.product_data = pd.merge(self.product_data,
                                     allocations.drop_duplicates(subset="with_site").filter(["with_site", "Batch_Box", "Num_in_Box"]),
                                     how="left",
                                     on="with_site"
                                     )
        all_gb_data = self.load_csv_data("gb_df.csv")
        opening_inv = all_gb_data[all_gb_data["Month"] == self.first_month]
        self.product_data = pd.merge(self.product_data,
                                     opening_inv.filter(["SKU_Code", "Opening_Stock"]),
                                     how="left",
                                     on="SKU_Code"
                                     )
        self.prod_active_window = pd.merge(active_window.reset_index(),
                                           with_site_code.filter(["bom_sku", "with_site"]),
                                           how="right",
                                           on="bom_sku").drop(labels=["bom_sku", "with_site"], axis=1).to_numpy()
        self.future_active_window = pd.merge(ftr_window.reset_index(),
                                             with_site_code.filter(["bom_sku", "with_site"]),
                                             how="right",
                                             on="bom_sku").drop(labels=["bom_sku", "with_site"],
                                                                   axis=1).to_numpy()
        # Syncing OPC
        opc_df = self.opc_data
        self.opc_data = pd.merge(opc_df,
                                 with_site_code.filter(["bom_sku", "with_site"]),
                                 how="right",
                                 on="bom_sku").drop(labels=["with_site"], axis=1).set_index(keys="bom_sku")
        holding_data = self.holding_data
        self.holding_data = pd.merge(holding_data,
                                     with_site_code.filter(["bom_sku", "with_site"]),
                                     how="right",
                                     on="bom_sku").drop(labels=["with_site"], axis=1).set_index(keys="bom_sku")

    def _generate_active_prod_data(self):
        """Return the product data of the products that will be produced during run window.
            Also returns the boolean matrix showing which periods each product is active on
            and will have a variable assigned to it (for both run periods and future period)"""
        all_product_data = self.load_csv_data("product_df.csv")
        # all_product_data = all_product_data[all_product_data["Active_Products"] != 0]
        all_product_data = all_product_data[all_product_data["Active_Products"] != 0]
        all_product_data["bom_sku"] = \
            all_product_data["SKU_Code"].astype(str) + "-" + all_product_data["bom_version"].astype(int).astype(str)
        all_product_count = len(all_product_data.index)
        all_dates_matrix = np.repeat(self.all_period_dates[np.newaxis, :], all_product_count, axis=0)
        allocation_df = self.load_csv_data("allocation.csv")
        active_product_mat = self.is_active_calculator(all_product_data, all_dates_matrix, allocation_df)
        product_period_counts = np.sum(active_product_mat, axis=1)
        active_product_mask = product_period_counts != 0
        active_product_data = all_product_data[active_product_mask].reset_index(drop=True)
        # active_product_periods = product_period_counts[active_product_mask]
        product_active_window = active_product_mat[active_product_mask][:, :-self.future_length]
        future_active_window = active_product_mat[active_product_mask][:, -self.future_length]
        return active_product_data, product_active_window, future_active_window

    @staticmethod
    def is_active_calculator(products_df, date_matrix, allocation_df):
        """Function for determining if a product is active
            for the period that we are optimizing for."""
        rayvarz_vec = products_df["bom_sku"].to_numpy()
        # Vector of rayvarzIDs repeated "period_count" times
        num_cols = date_matrix.shape[1]
        # Matrix determining if month is greater or equal to entry month
        entry_exit_vec = products_df["Active_Products"].to_numpy()
        # Vector of entry/exit dates repeated into matrix
        entry_exit_mat = entry_exit_vec[:, np.newaxis].repeat(repeats=num_cols, axis=1)
        has_entered = np.logical_or(np.equal(entry_exit_mat, 1), np.greater_equal(date_matrix, entry_exit_mat))
        # Matrix determining if month is less than exit month
        has_not_left = np.logical_or(np.greater_equal(entry_exit_mat, 1),
                                     np.less_equal(date_matrix, np.absolute(entry_exit_mat)))
        allocation_df["bom_sku"] = allocation_df["SKU_Code"].astype(str) + "-" + \
                                      allocation_df["BOM_Version"].astype(int).astype(str)
        allocated_prods = allocation_df["bom_sku"].drop_duplicates()
        has_allocation = pd.Series(rayvarz_vec).isin(allocated_prods).to_numpy()
        print(f"The following RayvarzIDs do not have any allocation data and will not be considered in the "
              f"model: {rayvarz_vec[has_allocation == False]}")
        # In BOM and Active in Products AND is in Active period
        active_product = has_allocation[:, np.newaxis] * has_entered * has_not_left
        return active_product

    def _generate_campaign_data(self):
        campaign_df = self.load_csv_data("campaign_df.csv")
        campaign_df["Process"] = campaign_df["Process"].str.replace(" ", "_")
        campaign_df = campaign_df[campaign_df["Active"] == 1]
        campaign_sites = campaign_df["Site"].unique()
        campaign_dict = {}
        for site in campaign_sites:
            campaign_machines = self.machine_available_time[self.machine_available_time["Site"] == site]
            campaign_ops = campaign_df[campaign_df["Site"] == site]["Process"].to_list()
            campaign_machines = campaign_machines[campaign_machines["Sub_Process"].isin(campaign_ops)]
            products = self.allocations[
                self.allocations["Machine_Code"].isin(campaign_machines["Machine_Code"].to_list())]["with_site"].unique()
            products = self.product_data[self.product_data["with_site"].isin(products.tolist())]
            excluded_franchises = campaign_df[campaign_df["Site"] == site]["Franchise_exclusion"].unique().tolist()
            campaign_products = products[~products["Franchise"].isin(excluded_franchises)]
            campaign_dict[site] = [campaign_products, campaign_machines]
        return campaign_dict

    def _generate_gb_data(self):
        active_prods = self.product_data["SKU_Code"].to_list()
        active_periods = self.all_period_dates
        all_gb_data = self.load_csv_data("gb_df.csv")
        gb_data = all_gb_data[(all_gb_data["SKU_Code"].isin(active_prods)) &
                              (all_gb_data["Month"].isin(active_periods))]
        demand_mat = self._aggregate_gb_data(gb_data, "Total_Demand").iloc[:, :-self.future_length]
        ftr_demand_mat = self._aggregate_gb_data(gb_data, "Total_Demand").iloc[:, -self.future_length]
        salesloss_mat = self._aggregate_gb_data(gb_data, "SL_Limit").iloc[:, :-self.future_length]
        ll_ss_mat = self._aggregate_gb_data(gb_data, "FG_LL_SS").iloc[:, :-self.future_length]
        ul_ss_mat = self._aggregate_gb_data(gb_data, "FG_UL_SS").iloc[:, :-self.future_length]
        ftr_ll_ss_mat = self._aggregate_gb_data(gb_data, "Total_Demand").iloc[:, -self.future_length]
        ftr_ul_ss_mat = self._aggregate_gb_data(gb_data, "Total_Demand").iloc[:, -self.future_length]
        dmd_offset = self._aggregate_gb_data(gb_data, "DM_OFFSET").iloc[:, :-self.future_length]
        return demand_mat, ftr_demand_mat, salesloss_mat, ll_ss_mat, ul_ss_mat, ftr_ll_ss_mat, ftr_ul_ss_mat, dmd_offset

    def _aggregate_gb_data(self, gb_dataframe: pd.DataFrame, name: str):
        allocation_df = self.load_csv_data("allocation.csv")
        sku_numbox = allocation_df.filter(["SKU_Code", "Num_in_Box"]).drop_duplicates(subset="SKU_Code")
        gb_data = gb_dataframe.merge(sku_numbox, how="left", on="SKU_Code")
        inter_codes = self.product_data.filter(["SKU_Code",
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

    def _generate_operations_data(self, name: str) -> pd.DataFrame:
        operation_data = self.load_csv_data(name)
        operation_data["bom_sku"] = operation_data["SKU_Code"].astype(str) + "-" + operation_data[
            "Bom_Version"].astype(str)
        inter_sku = self.product_data.filter(items=["bom_sku"])
        operation_data = operation_data.merge(inter_sku, how="right", on="bom_sku", copy=True)
        operation_data = operation_data.filter(items=["Wet_Granulation",
                                                      "Drying",
                                                      "Blending",
                                                      "Roller_Compactor",
                                                      "Compression_CAP_Filling",
                                                      "Coating",
                                                      "Blistering_Counter_Syrup_Filling",
                                                      "Manual_Packaging",
                                                      "bom_sku"]).fillna(0)
        operation_data.set_index("bom_sku", drop=True, inplace=True)
        return operation_data.astype(int)

    def _generate_wip_data(self):
        wip_df = self.load_csv_data("wip_df.csv")
        available_wip_df = wip_df[wip_df["Material"] > 0]
        with_site_code = self.timing_data.copy()
        with_site_code.insert(loc=0, column="bom_sku", value=with_site_code.index.map(lambda x: x[:-2]))
        with_site_code.insert(loc=0, column="with_site", value=with_site_code.index)
        with_site_code.insert(loc=0, column="SKU_Code", value=with_site_code.index.map(lambda x: int(x[:7])))
        inter_sku = with_site_code.filter(items=["bom_sku", "SKU_Code"])
        inter_sku["period_active"] = self.prod_active_window[:, 0]
        available_wip_df.insert(loc=0, value=available_wip_df["code"].astype(str) + "-" + \
                                             available_wip_df["BOM_Version"].astype(str) + "-" + \
                                             available_wip_df["Site"].astype(str), column="with_site")
        available_wip_df = available_wip_df.merge(inter_sku, on="with_site", how="right")
        available_wip_df.set_index(keys="bom_sku", drop=True, inplace=True)
        available_wip_df[(available_wip_df["period_active"] == False)] = 0
        available_wip_df = available_wip_df.loc[:, "Material":"Manual_Packaging"].fillna(0)
        available_wip_mat = available_wip_df.iloc[:, 1:].to_numpy()
        available_wip_tens = np.repeat(available_wip_mat[:, :, np.newaxis],
                                       axis=2,
                                       repeats=len(self.solve_window_dates))
        available_wip_tens[:, :, 1:] = 0
        return available_wip_df, available_wip_tens

    def _generate_allocation_data(self):
        available_time_data = self.load_csv_data("available_time_df.csv")
        machine_codes = available_time_data["Machine_Code"].to_list()
        timing_data_df = self.load_csv_data("timing_df.csv", index_col=0)
        timing_data_df = timing_data_df[machine_codes]
        assert (available_time_data["Machine_Code"] == timing_data_df.columns).all(), \
            f'Available time and cycle time machine order does not match'
        inter_sku = self.product_data.filter(items=["bom_sku"])
        timing_data_df.insert(loc=0, column="bom_sku", value=timing_data_df.index.map(lambda x: x[:-2]))
        timing_data_df.insert(loc=0, column="with_site", value=timing_data_df.index)
        avail_ct_data = inter_sku.merge(timing_data_df, how="left", on="bom_sku")
        avail_ct_data.set_index(keys="with_site", drop=True, inplace=True)
        avail_ct_data.drop(labels=["bom_sku"], inplace=True, axis=1)
        assert not avail_ct_data.iloc[:, 1].isnull().values.any(), \
            f'Cycle time missing for {avail_ct_data[avail_ct_data.iloc[:, 1].isnull()]}'
        allocations = self.load_csv_data("allocation.csv")
        allocations = allocations[(allocations["Machine_Code"].isin(list(avail_ct_data.columns)))]
        allocations = allocations[(allocations["Active"] != 1) | (allocations["Deactivate"] != 1)]
        allocations.insert(loc=0, column="with_site",
                           value=allocations["SKU_Code"].astype(str) + "-"
                                 + allocations["BOM_Version"].astype(str) + "-"
                                 + allocations["Type"].astype(str))
        periodwise_avail_ct = np.repeat(avail_ct_data.to_numpy()[:, :, np.newaxis],
                                        repeats=len(self.solve_window_dates),
                                        axis=2)
        for _, entry in allocations.iterrows():
            bom_sku = entry["with_site"]
            if bom_sku not in avail_ct_data.index:
                continue
            entry, removal, machine = entry["Active"], entry["Deactivate"], entry["Machine_Code"]
            assert removal > entry if removal > 200000 else True, f"Removal date should be bigger than entry date " \
                                                                  f"for {bom_sku} on {machine}"
            if ((entry == 1 or entry <= self.solve_window_dates.min()) and
                (removal == 1 or removal > self.solve_window_dates.max())):
                continue
            elif (entry == 0 or removal == 0) \
                    or (200000 < removal <= self.solve_window_dates.min()) \
                    or (entry > self.solve_window_dates.max()):
                deactive_indices = [i for i in range(len(self.solve_window_dates))]
            else:
                try:
                    entry_index = list(self.solve_window_dates).index(entry)
                except ValueError:
                    entry_index = 0
                try:
                    removal_index = list(self.solve_window_dates).index(removal)
                except ValueError:
                    removal_index = self.run_duration
                deactive_indices = [i for i in range(entry_index)]
                deactive_indices.extend([i for i in range(removal_index, len(self.solve_window_dates))])
            product_index = avail_ct_data.index.to_list().index(bom_sku)
            machine_index = avail_ct_data.columns.to_list().index(machine)
            periodwise_avail_ct[product_index, machine_index, deactive_indices] = 0
        # available_time_data.set_index(keys="Machine_Code", drop=True, inplace=True)
        # available_time_data = available_time_data[list(map(str, self.solve_window_dates))]
        inter_codes = pd.DataFrame(data=avail_ct_data.index.to_numpy(), columns=["with_site"])
        inter_codes.insert(loc=0, value=inter_codes["with_site"].map(lambda x: x[:-2]), column="bom_sku")
        inter_codes.insert(loc=0, value=inter_codes["bom_sku"].map(lambda x: x[:7]), column="SKU_Code")
        unique_codes, count = np.unique(inter_codes["bom_sku"], return_counts=True)
        duplicate_codes = unique_codes[count > 1]
        mask = inter_codes[inter_codes["bom_sku"].isin(duplicate_codes)].groupby("bom_sku").cumcount()
        # Index of duplicate inter_codes that should be deleted
        mask = list(mask[mask != 0].index)
        allocations = self.load_csv_data("allocation.csv")
        allocations = allocations[(allocations["Active"] != 0) & (allocations["Deactivate"] != 0)]
        allocations["with_site"] = allocations["SKU_Code"].astype(str) + "-" +\
            allocations["BOM_Version"].astype(str) + "-" + allocations["Type"].astype(str)
        return available_time_data, avail_ct_data, periodwise_avail_ct, mask, allocations

    def _generate_material_data(self):
        material_df = self.load_csv_data("material_df.csv")
        arrival_dates = material_df.columns[material_df.columns.map(lambda x: x.isdigit())].to_list()
        bom_data = self.load_csv_data("bom_df.csv")
        bom_data.insert(loc=1, column="bom_sku",
                        value=bom_data["Product_Code"].astype(str) + "-" + bom_data["BOM_VERSION"].astype(str))
        bom_data = bom_data[bom_data["bom_sku"].isin(self.product_data["bom_sku"])]
        missing_codes = self.product_data[self.product_data["bom_sku"]
                                              .isin(bom_data["bom_sku"]) == False]["bom_sku"].to_list()
        assert not missing_codes, f'BOM data does not include \n {missing_codes}'
        avail_bom_data = bom_data[(bom_data["MIP_Activation"] == 1) &
                                  (bom_data["BOM_Status"] == 1) &
                                  (bom_data["Usage"] == 1)]
        material_codes = avail_bom_data["Material_Code"].unique()
        batch_box_df = self.allocations.filter(["SKU_Code", "BOM_Version", "Batch_Box"])
        batch_box_df["bom_sku"] = batch_box_df["SKU_Code"].astype(str) + "-" + \
                                     batch_box_df["BOM_Version"].astype(str)
        avail_bom_data = avail_bom_data.merge(batch_box_df.filter(["bom_sku", "Batch_Box"]),
                                              on="bom_sku", how="left")
        avail_bom_data["ratio"] = avail_bom_data["Batch_Box"]/avail_bom_data["Standard_Batch_(box)"]
        avail_bom_data["Consumption_Factor_(Wastage)"] = avail_bom_data["Consumption_Factor_(Wastage)"] * \
                                                         avail_bom_data["ratio"]
        material_df = material_df[(material_df["Code"].isin(material_codes)) & (material_df["Level_Num"] == 1)]
        material_df.set_index(keys="Code", drop=True, inplace=True)
        opening_df = material_df.filter(items=["Production_Support", "LT", "Opening"])
        order_df = material_df.filter(items=arrival_dates)
        order_df = order_df[list(map(str, self.solve_window_dates))]
        return opening_df, order_df, avail_bom_data

    def _generate_weights(self):
        def duplicate_sku_calc():
            """Since numpy.unique automatically sorts its outputs, we have to reorder the outputs to match
               our input data."""
            all_skus = self.product_data["SKU_Code"].to_numpy()
            unique_skus, sku_counts = np.unique(all_skus, return_counts=True)
            sorted_df = pd.DataFrame(data=np.vstack([unique_skus, sku_counts]).T, columns=["SKU", "Counts"])
            ordered_df = pd.DataFrame(data=pd.unique(all_skus), columns=["SKU"])
            ordered_df = pd.merge(ordered_df, sorted_df, how="left", on="SKU")
            unique_skus, sku_counts = ordered_df["SKU"].to_numpy(), ordered_df["Counts"].to_numpy()
            duplicated_skus = unique_skus[sku_counts >= 2]
            dupl_sku_index = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]
            return dupl_sku_index, sku_counts

        dupl_sku_index, sku_counts = duplicate_sku_calc()
        farvardin_weight = []
        for date in self.all_period_dates:
            if date % 100 == 4:
                farvardin_weight.append(10000)
            elif date % 100 == 3:
                farvardin_weight.append(5000)
            else:
                farvardin_weight.append(1)
        product_count = len(sku_counts)
        farvardin = np.repeat(np.array(farvardin_weight[:-1])[np.newaxis, :],
                              repeats=product_count, axis=0)
        farvardin_f = np.repeat(np.array(farvardin_weight[-1]),
                                repeats=product_count, axis=0)
        strategic_tier = np.tile(250, (product_count, self.run_duration))
        monthly_weight = np.repeat(np.arange(150000, 9000, -10000)[np.newaxis, :],
                                   product_count, axis=0)[:, :self.run_duration]
        future_monthly = np.repeat(np.arange(150000, 9000, -10000)[np.newaxis, :],
                                   product_count, axis=0)[:, self.run_duration]
        return farvardin, farvardin_f, strategic_tier, monthly_weight, future_monthly

    def get_prod_weight(self, weight_name: str, future=False) -> np.ndarray:
        unique_key = self.product_data.drop_duplicates(subset="SKU_Code", keep="first")["with_site"].to_list()
        unique_mask = self.product_data["with_site"].isin(unique_key)
        if future:
            return self.product_data.loc[unique_mask, weight_name].to_numpy()
        period_count = len(self.solve_window_dates)
        return np.repeat(self.product_data.loc[unique_mask, weight_name].to_numpy()[:, np.newaxis],
                         period_count, axis=1)[:, -1]

    def generate_holding_bigm(self):
        holding_mat = self.holding_data.iloc[:, -7:].to_numpy()
        batch_size = self.product_data["Batch_Box"].to_numpy()
        ll_ss = pd.merge(self.ll_ss,
                         self.product_data.filter(["SKU_Code", "with_site"]),
                         how="left",
                         on="SKU_Code").drop(labels=["SKU_Code", "with_site"], axis=1).to_numpy()
        bigm_mat = np.ceil(ll_ss / batch_size[:, np.newaxis])
        bigm_tensor = np.repeat(bigm_mat[:, np.newaxis, :], axis=1, repeats=8)
        product_count = len(self.product_data.index)
        for p in range(product_count):
            mask = np.zeros_like(bigm_tensor[0, :, :], dtype=bool)
            for o in range(7):
                if holding_mat[p, o] > 0:
                    mask[o, -holding_mat[p, o]:] = True
            bigm_tensor[p, :, :] = np.multiply(bigm_tensor[p, :, :], mask)
        return bigm_tensor

    def get_duplicate_sku_index(self):
        all_skus = self.product_data["SKU_Code"].to_numpy()
        unique_skus, sku_count = np.unique(all_skus, return_counts=True)
        duplicated_skus = unique_skus[sku_count >= 2]
        dupl_sku_indices = [np.where(all_skus == sku)[0][0] for sku in duplicated_skus]
        return dupl_sku_indices

