import numpy as np
import pandas as pd
from docplex.mp.model import Model


class BatchVariable:
    """Optimization variables class designed to work with docplex that defines variables that represent
     the number of batches produced(of each product on each machine in each period). Also contains
     helper functions for getting access to needed aggregations of the variables."""

    def __init__(self,
                 lp_data,
                 model: Model):
        self.with_site = lp_data.timing_data.index.to_numpy()
        self.mask = lp_data.mask
        self.product_level = lp_data.product_data.filter(["Product_Code", "Product_Counter", "Batch_Box"])
        self.periods = lp_data.prod_active_window
        self.allocations = lp_data.prod_allocations
        self.machines = lp_data.timing_data.columns
        self.op_machines = lp_data.machine_available_time.filter(['Machine_Code', 'Process'])
        self.opc = lp_data.opc_data.iloc[:, -8:]
        self.model = model
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
            x[index] = self.model.integer_var(lb=0, ub=self.model.infinity, name=f'x{list(index)}')
        return x

    # Ready to use
    def generate_op_batch_variables(self, wip_tensor: np.ndarray):
        """Returns the batch variables that belong to the same operation"""
        opc_matrix = self.opc.to_numpy()
        op_count = opc_matrix.shape[1]
        product_count = self.with_site.shape[0]
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
                batch_var[p] = self.model.sum(variables[:, p])
        else:
            variables = self.x[product_index, :, :][:, machine_indices, :]
            for p in range(periods):
                batch_var[p] = self.model.sum(variables[:, :, p])
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
        product_count = self.with_site.shape[0]
        period_count = self.periods.shape[1]
        input_batch = np.empty(shape=[product_count, period_count], dtype=np.object)
        input_batch[:, :] = 0
        for prod in range(product_count):
            op_index = first_op_indices[prod]
            input_batch[prod, :] = self.get_op_batch_variable(prod, op_index)
        return input_batch

    # Ready to use
    def get_output_box(self, wip_tensor: np.ndarray):
        """Returns the batches produced on the final operation of each
        product in each period and also sums the output of products with
        identical product codes."""
        last_op_indices = np.maximum(self.opc.shape[1] - np.flip(self.opc.to_numpy(), axis=1).argmax(axis=1) - 1, 6)
        product_codes = self.product_level["Product_Code"].to_numpy()
        product_counters = self.product_level["Product_Counter"].to_numpy()
        unique_codes, count = np.unique(product_codes, return_counts=True)
        duplicate_codes = unique_codes[count > 1]
        product_count = self.with_site.shape[0]
        period_count = self.periods.shape[1]
        output_batch = np.empty(shape=[product_count, period_count], dtype=np.object)
        output_batch[:, :] = 0
        for prod in range(product_count):
            prod_code = product_codes[prod]
            op_index = last_op_indices[prod]
            output_batch[prod, :] = self.get_op_batch_variable(prod, op_index) + \
                                    wip_tensor[prod, op_index, :]
        output_box = output_batch * self.product_level["Batch_Box"].to_numpy()[:, np.newaxis]
        pure_output_box = np.zeros_like(output_box)
        pure_output_box += output_box
        for prod in range(product_count):
            prod_code = product_codes[prod]
            if prod_code in duplicate_codes:
                product_indices = list(np.where(product_codes == prod_code)[0])
                output_box[product_indices] = np.sum(output_box[product_indices, :], axis=0)
                # Warning: This method permanently changes duplicate_codes
                duplicate_codes = np.delete(duplicate_codes, np.where(duplicate_codes == prod_code)[0])

        # output_batch = np.delete(output_batch, self.mask, axis=0)
        output_box *= self.product_level["Product_Counter"].to_numpy()[:, np.newaxis]
        output_box = np.delete(output_box, self.mask, axis=0)
        return output_box, pure_output_box

    def get_site_box(self):
        pass

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
