import numpy as np
import pandas as pd
from docplex.mp.model import Model
from MIP_Pro.LP.LPData import LPData


# TODO: Needs to be reimplemented using hierarchical classes
class SlackVariable:
    """Optimization variables class designed to work with docplex that defines
       slack variables that represent various losses during production planning."""

    def __init__(self,
                 slack_name: str,
                 lp_data: LPData,
                 solver: Model):
        self.name = slack_name
        self.mask = lp_data.mask
        self.skus = lp_data.product_data["SKU_Code"]
        self.periods, self.period_count = self._calculate_periods(lp_data.prod_active_window)
        self.product_count = self.periods.shape[0]
        self.solver = solver
        self.var = self.create_num_slack_variables()

    def create_num_slack_variables(self):
        """Creates the slack variables based on period count"""

        slack = np.empty(shape=[self.product_count, self.period_count], dtype=np.object)
        slack[:, :] = 0
        if len(self.periods.shape) > 1:
            prod_idx, period_idx = np.where(self.periods == True)
        else:
            prod_idx = np.where(self.periods == True)[0]
            period_idx = np.zeros_like(prod_idx)
        for index in zip(prod_idx, period_idx):
            slack[index] = self.solver.continuous_var(0, self.solver.infinity, f'{self.name}{list(index)}')
        return slack

    def _calculate_periods(self, active_window):
        if len(active_window) > 1:
            period_count = active_window.shape[1]
        else:
            period_count = 1
        if self.name != "expiry":
            unique_skus = self.skus.unique()
            skus = np.delete(self.skus.to_numpy(), self.mask, axis=0)
            periods = np.delete(active_window, self.mask, axis=0)
            alt_periods = np.empty(shape=(unique_skus.shape[0], period_count), dtype=np.bool)
            for i in range(unique_skus.shape[0]):
                sku = unique_skus[i]
                alt_periods[i] = np.sum(periods[np.where(skus == sku)[0]], axis=0)
        else:
            alt_periods = active_window
        return alt_periods, period_count


