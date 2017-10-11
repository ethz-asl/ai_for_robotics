import numpy as np


class Variables():
    """
    Variable structure for networks variables / parameters and their
    gradients
    """
    weights = []
    biases = []

    def __init__(self):
        self.weights = []
        self.biases = []

    def __len__(self):
        if len(self.weights) == len(self.biases):
            return len(self.weights)
        else:
            print("Dimension mismatch.")
            raise

    def __mul__(self, factor):
        new_p = Variables()
        for w, b in zip(self.weights, self.biases):
            new_p.weights.append(w * factor)
            new_p.biases.append(b * factor)
        return new_p

    def __add__(self, other_variables):
        assert len(self.weights) == len(
            other_variables.weights), 'Number of weight entries have to match.'
        assert len(self.biases) == len(
            other_variables.biases), 'Number of bias entries have to match.'

        new_p = Variables()
        for w, b, o_w, o_b in zip(self.weights, self.biases,
                                  other_variables.weights,
                                  other_variables.biases):
            new_p.weights.append(w + o_w)
            new_p.biases.append(b + o_b)
        return new_p

    def __sub__(self, other_variables):
        return self.__add__(other_variables * (-1))

    def __eq__(self, other_variables):
        assert len(self.weights) == len(
            other_variables.weights), 'Number of weight entries have to match.'
        assert len(self.biases) == len(
            other_variables.biases), 'Number of bias entries have to match.'

        variables_equal = True

        for i in range(len(self.weights)):
            if not np.all(self.weights[i] == other_variables.weights[i]):
                variables_equal = False
                break
            if not np.all(self.biases[i] == other_variables.biases[i]):
                variables_equal = False
                break

        return variables_equal

    def __ne__(self, other_variables):
        return not self.__eq__(other_variables)


# Methods
def computeScore(network, x, labels):
    n_samples = x.shape[0]
    correct_classifications = 0.0
    for i in range(n_samples):
        if np.argmax(network.output(x[i, :])) == np.argmax(labels[i, :]):
            correct_classifications += 1.0

    return correct_classifications / float(n_samples)
