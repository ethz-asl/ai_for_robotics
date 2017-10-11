import numpy as np
from random import shuffle


class Batchmaker:
    def __init__(self, input_data, output_data, batch_size=64):
        self.input_data = input_data
        self.input_shape = input_data.shape[1] * input_data.shape[2]
        self.output_data = output_data
        self.output_shape = output_data.shape[1]

        assert len(input_data) == len(output_data)
        assert type(batch_size) is int
        if batch_size > len(input_data):
            print(
                "WARNING: more examples per batch than possible examples in all input_data"
            )
            self.batch_size = len(input_data)
        else:
            self.batch_size = batch_size

        # initialize example indices list
        self.remaining_example_indices = list(range(len(input_data)))
        shuffle(self.remaining_example_indices)

    def next_batch(self):
        # Create a single batch
        batch_input_values = np.zeros([self.batch_size] + [self.input_shape])
        batch_output_values = np.zeros([self.batch_size] + [self.output_shape])
        for i_example in range(self.batch_size):
            if not self.remaining_example_indices:
                self.remaining_example_indices = list(
                    range(len(self.input_data)))
                shuffle(self.remaining_example_indices)

        # Create training example at index 'pos' in input_data.
            pos = self.remaining_example_indices.pop(0)
            batch_input_values[i_example] = np.reshape(self.input_data[pos],
                                                       [self.input_shape])
            batch_output_values[i_example] = np.reshape(
                self.output_data[pos], [self.output_shape])
        return batch_input_values, batch_output_values
