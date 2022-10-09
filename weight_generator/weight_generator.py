import numpy as np
import pandas as pd
from docplex.mp.model import Model
from weight_input import WeightingData
from weight_policy import PolicyDefinition


if __name__ == "__main__":
    period_count = 14
    first_month = 140103
    lp_solver = Model(name="Weights")
    weight_data = WeightingData(first_month, period_count)
    policy = PolicyDefinition(weight_data, lp_solver)
    status = lp_solver.solve(log_output=True)

    solution_matrix = np.zeros_like(policy.weights.sl_var, dtype=float)
    prod, month = np.where(policy.weights.sl_var)
    for var in zip(prod, month):
        try:
            solution_matrix[var] = policy.weights.sl_var[var].solution_value
        except AttributeError:
            solution_matrix[var] = policy.weights.sl_var[var]
    solution_matrix = np.round(solution_matrix, decimals=2)
    weight_df = pd.DataFrame(data=solution_matrix, index=weight_data.product_data["SKU_name"])
