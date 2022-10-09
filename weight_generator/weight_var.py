import numpy as np
import pandas as pd
from docplex.mp.model import Model


class WeightVariable:
    """Optimization variables class designed to work with docplex that defines variables that will later be used
       in the main MIP problem as various weights."""

    def __init__(self, weight_data, solver):
        self.data = weight_data
        self.solver = solver
        self.sl_var = self._create_weight_vars("sl")
        # self.gb_var = self._create_weight_vars("gb")
        # self.pl_var = self._create_weight_vars("pl")
        # self.os_var = self._create_weight_vars("os")

    def _create_weight_vars(self, name):
        """Creates the weight variables based on product and period count."""
        product_space = self.data.prod_active_window
        var = np.empty_like(product_space, dtype=np.object)
        var[:, :] = 0
        prod_idx, period_idx = np.where(var == 0)
        for index in zip(prod_idx, period_idx):
            var[index] = self.solver.continuous_var(lb=0.0001, ub=self.solver.infinity, name=f'{name}{list(index)}')
        return var
