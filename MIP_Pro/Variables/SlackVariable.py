import numpy as np
import pandas as pd
from docplex.mp.model import Model


class SlackVariable:
    """Optimization variables class designed to work with docplex that defines
       slack variables that represent various losses during production planning."""
    def __init__(self,
                 slack_name: str,
                 product_periods: np.ndarray,
                 product_data: pd.DataFrame,
                 solver: Model):
        self.name = slack_name
        self.periods = product_periods
        self.skus = product_data["SKU_Code"]
        if len(self.periods.shape) > 1:
            self.period_count = self.periods.shape[1]
        else:
            self.period_count = 1
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
        if self.name == "expiry":
            return slack
        unique_skus = self.skus.unique()
        skus = self.skus.to_numpy()
        aggr_slack = np.empty(shape=(unique_skus.shape[0], self.period_count), dtype=np.object)
        for i in range(unique_skus.shape[0]):
            sku = unique_skus[i]
            aggr_slack[i] = np.sum(slack[np.where(skus == sku)[0]], axis=0)
        return aggr_slack
