import numpy as np
import pandas as pd
from docplex.mp.model import Model
from MIP_Pro.LP.LPData import LPData


# TODO: Needs to be reimplemented using hierarchical classes
class TimeVariable:
    """ """

    def __init__(self,
                 lp_data: LPData,
                 model: Model):
        self.campaign_data = lp_data.campaign_data
        self.skus = lp_data.product_data["SKU_Code"]
        self.periods, self.period_count = self._calculate_periods(lp_data.prod_active_window)
        self.model = model
        self.var = self.create_time_variables()

    def create_time_variables(self):
        """Creates the slack variables based on period count"""

        slack_dict = {}
        name_dict = {}
        sites = list(self.campaign_data.keys())
        for site in sites:
            franchise_list = self.campaign_data[site][0]["Franchise"].unique().tolist()
            machine_list = self.campaign_data[site][1]["Machine_Code"].unique().tolist()
            slack = np.empty(shape=[len(franchise_list), len(machine_list), self.period_count], dtype=np.object)
            slack[:, :, :] = 0
            fran_idx, machine_idx, period_idx = np.where(slack == 0)
            for index in zip(fran_idx, machine_idx, period_idx):
                slack[index] = self.model.continuous_var(0, 1000, f'{franchise_list[index[0]]}{list(index)}')
            slack_dict[site] = slack
        return slack_dict

    @staticmethod
    def _calculate_periods(active_window):
        if len(active_window) > 1:
            period_count = active_window.shape[1]
        else:
            period_count = 1
        alt_periods = active_window
        return alt_periods, period_count
