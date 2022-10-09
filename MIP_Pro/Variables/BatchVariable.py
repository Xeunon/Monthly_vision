import numpy as np
import pandas as pd
from docplex.mp.model import Model


class BatchVariable:
    """Optimization variables class designed to work with docplex that defines variables that represent
     the number of batches produced(of each product on each machine in each period). Also contains
     helper functions for getting access to needed aggregations of the variables."""

    def __init__(self,
                 product_data: pd.DataFrame,
                 product_periods: np.ndarray,
                 timing_data: pd.DataFrame,
                 op_machines: pd.DataFrame,
                 product_opc: pd.DataFrame,
                 product_allocations: np.ndarray,
                 solver: Model):
        self.inter_codes = timing_data.index.to_numpy()
        self.product_level = product_data.filter(["Product_Code", "Product_Counter"])
        self.periods = product_periods
        self.allocations = product_allocations
        self.machines = timing_data.columns
        self.op_machines = op_machines
        self.opc = product_opc
        self.solver = solver
        self.x = self._create_int_batch_variables()
        self.op_machine_indices = self._get_op_machine_indices(return_dict=False)

    # Ready to use
    def _create_int_batch_variables(self):
        """Creates the production batch variables based on available allocations and period count"""
        allocation_space = self.allocations
        variable_space = np.multiply(allocation_space, self.periods[:, np.newaxis])
        x = np.empty_like(variable_space, dtype=np.object)
        x[:, :, :] = 0
        prod_idx, machine_idx, period_idx = np.where(variable_space)
        for index in zip(prod_idx, machine_idx, period_idx):
            x[index] = self.solver.integer_var(lb=0, ub=self.solver.infinity, name=f'x{list(index)}')
        return x

    # Ready to use
    def generate_op_batch_variables(self, wip_tensor: np.ndarray):
        """Returns the batch variables that belong to the same operation"""
        opc_matrix = self.opc.to_numpy()
        op_count = opc_matrix.shape[1]
        product_count = self.inter_codes.shape[0]
        period_count = self.periods.shape[1]
        op_batch = np.empty(shape=[product_count, op_count, period_count], dtype=np.object)
        op_batch[:, :, :] = 0
        for prod in range(product_count):
            for op in range(op_count):
                if opc_matrix[prod, op]:
                    op_batch[prod, op, :] = self.get_op_batch_variable(prod, op)
        return op_batch + wip_tensor

    # Ready to use
    def get_op_batch_variable(self, product_index, op_index):
        """Aggregates variables that belong to the same operation for all periods"""
        machine_indices = self.op_machine_indices[op_index]
        periods = self.periods.shape[1]
        batch_var = np.empty(shape=periods, dtype=np.object)
        if type(product_index) == int:
            variables = self.x[product_index, machine_indices, :]
            for p in range(periods):
                batch_var[p] = self.solver.sum(variables[:, p])
        else:
            variables = self.x[product_index, :, :][:, machine_indices, :]
            for p in range(periods):
                batch_var[p] = self.solver.sum(variables[:, :, p])
        return batch_var

    # Ready to use
    def generate_cumulative_batch_variables(self, wip_tensor: np.ndarray):
        """Cumulatively sums the batches produced on each operation through different periods"""
        return np.cumsum(self.generate_op_batch_variables(wip_tensor), axis=2)

    # Ready to use
    def get_input_batch(self):
        """Returns the batches produced on the final operation of each
        product in each period"""
        first_op_indices = self.opc.to_numpy().argmax(axis=1)
        product_count = self.inter_codes.shape[0]
        period_count = self.periods.shape[1]
        input_batch = np.empty(shape=[product_count, period_count], dtype=np.object)
        input_batch[:, :] = 0
        for prod in range(product_count):
            op_index = first_op_indices[prod]
            input_batch[prod, :] = self.get_op_batch_variable(prod, op_index)
        return input_batch

    # Ready to use
    def get_output_batch(self, wip_tensor: np.ndarray):
        """Returns the batches produced on the final operation of each
        product in each period and also sums the output of products with
        identical product codes."""
        last_op_indices = np.maximum(self.opc.shape[1] - np.flip(self.opc.to_numpy(), axis=1).argmax(axis=1) - 1, 6)
        product_codes = self.product_level["Product_Code"].to_numpy()
        product_counters = self.product_level["Product_Counter"].to_numpy()
        unique_codes, count = np.unique(product_codes, return_counts=True)
        duplicate_codes = unique_codes[count > 1]
        product_count = self.inter_codes.shape[0]
        period_count = self.periods.shape[1]
        output_batch = np.empty(shape=[product_count, period_count], dtype=np.object)
        output_batch[:, :] = 0
        for prod in range(product_count):
            prod_code = product_codes[prod]
            # machine_indices = op_index_list[last_op_indices[prod]]
            op_index = last_op_indices[prod]
            if prod_code in duplicate_codes:
                if product_counters[prod] == 1:
                    product_indices = list(np.where(product_codes == prod_code)[0])
                    output_batch[prod, :][self.periods[prod]] = \
                        self.get_op_batch_variable(product_indices, op_index)[self.periods[prod]] + \
                        wip_tensor[prod, op_index, :][self.periods[prod]]
            else:
                output_batch[prod, :] = self.get_op_batch_variable(prod, op_index) + \
                                        wip_tensor[prod, op_index, :]

        return output_batch

    # TODO: Cumulative output batch

    # Ready to use
    def _get_op_machine_indices(self, return_dict=True):
        """Defining a dictionary/list of the indexes of the machines in each op (OP:[Machines-Indices])"""
        op_index_dict = {}
        for op in self.opc:
            op_index_dict[op] = (self.op_machines.index[self.op_machines["Process"] == op])
        if return_dict:
            return op_index_dict
        else:
            return list(map(pd.Series.to_list, list(op_index_dict.values())))
