import numpy as np


class StudentAI:
    weights_matrix_list = None
    number_of_entry_values = None

    def __init__(self, number_of_entry_values):
        self.number_of_entry_values = number_of_entry_values

    def neuron(self, inputs, weights, bias):
        return np.dot(inputs, weights) + bias

    def neural_network(self, input_vector, weights_matrix):
        return np.matmul(weights_matrix, input_vector)

    def deep_neural_network(self, input_vector, weights_matrix_list):
        inputs = input_vector
        for weights_matrix in weights_matrix_list:
            inputs = self.neural_network(inputs, weights_matrix)
        return inputs

    def add_layer(self, n, weight_range_values):

    def generate_weight_matrix(self, n, weight_range_values):


# ZAD 2
weights_matrix = np.matrix('0.1 0.1 -0.3; 0.1 0.2 0.0; 0.0 0.7 0.1; 0.2 0.4 0.0; -0.3 0.5 0.1')
input_vector = np.matrix('0.5; 0.75; 0.1')
# print(neural_network(input_vector, weights_matrix))

# ZAD 3
weights_matrix_1 = weights_matrix
weights_matrix_2 = np.matrix('0.7 0.9 -0.4 0.8 0.1; 0.8 0.5 0.3 0.1 0.0; -0.3 0.9 0.3 0.1 -0.2')
weights_matrix_list = np.array([weights_matrix_1, weights_matrix_2], 'object')
print(deep_neural_network(input_vector, weights_matrix_list))

# ZAD 4