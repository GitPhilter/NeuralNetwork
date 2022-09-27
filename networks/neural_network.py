#!/usr/bin/env python
"""
Provide a feed forward neural network with full functionality.

The network can compute an output, given an input of the correct length, i.e. number of input nodes. The input vector
must contain numbers, i.e. ints or floats.
The error function is 1/2(y - t)², so that the derivative is just (y - t) and can be easily computed, without having
to actually calculate derivatives along the way.
"""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

from networks.activation_functions import get_activation_function_deriv


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


def get_new_weight(old_weight, eta, deriv_e_w):
    """Get new weight according to delta-function."""
    new_weight = old_weight - eta * deriv_e_w
    return new_weight


class NeuralNetwork:
    """
    A classic feed forward neural network.

    The layers consist of nodes, which in turn have an activation function, weights and a bias.
    The layer with index 0 is the input layer, the layer with index len(layers) - 1 is the output layer.
    """
    def __init__(self, name, layers):
        self.name = name
        self.layers = layers
        for layer in self.layers:
            for node in layer:
                node.bias = float(node.bias)
                for index, weight in enumerate(node.weights):
                    node.weights[index] = float(weight)
        self.output = [None]

    def compute_output(self, input_vector):
        """Compute the output of the network, given the input_vector."""
        if len(input_vector) != len(self.layers[0]):
            logger.error("Cannot compute output: input_vector and weights are of different lengths.")
        for layer in self.layers:
            new_input_vector = []
            for node in layer:
                node.compute_output(input_vector)
                new_input_vector.append(node.output)
            input_vector = new_input_vector.copy()
        # set network output
        self.output = []
        for node in self.layers[-1]:
            self.output.append(node.output)
        return self.output.copy()

    def backpropagation(self, actual_output, expected_output, eta):
        """
        Apply backpropagation with gradient descent and change weights and biases accordingly.

        As the error function 1/2(sum(y_i - t_i))^2 is used, so that the partial derivatives are easily
        computed as just y_i - t_i.
        """
        """Compute the total error / loss function."""
        error_derivs = []
        for index, ao in actual_output:
            """Append the derivatives of the loss function to the error_derivs."""
            error_derivs.append(ao - expected_output[index])
        """Go through all layers."""
        for layer_index in range(len(self.layers) - 1, 0, -1):
            next_error_derivs = []
            for node_index, node in self.layers[layer_index]:
                """Compute the result of the inner derivation of the activation function"""
                node_inner_deriv = get_activation_function_deriv(node.total_input)
                for weight_index, weight in node.weights:
                    """The outer derivative is always the output of the respective node."""
                    node_outer_deriv = self.layers[layer_index - 1][node_index].output
                    partial_derivation = node_outer_deriv * node_inner_deriv * error_derivs[node_index]
                    next_error_derivs.append(partial_derivation)
                    """Adapt the weight."""
                    node.weights[weight] = get_new_weight(node.weights[weight_index], eta, partial_derivation)
                """adapt bias"""
                partial_derivation = 1 * node_inner_deriv * error_derivs[node_index]
                node.bias = get_new_weight(node.bias, eta, partial_derivation)
            error_derivs = next_error_derivs

    def online_training(self, data_object, eta):
        output = self.compute_output(data_object.data)
        self.backpropagation(output, data_object.expected_output, eta)

    def set_weights_and_biases(self, network):
        """Copy all weights and biases from the given network to self."""
        for self_layer_index, layer in enumerate(network.layers):
            self.layers[self_layer_index] = layer.copy()

    def __str__(self):
        result_string = "Printing a network:\n"
        result_string += f"### Neural Network '{self.name}' ###\n"
        for layer_index, layer in enumerate(self.layers):
            result_string += f"# Layer no. {layer_index}: #\n"
            if layer_index == 0:
                for node_index, node in enumerate(layer):
                    result_string += f"Node no. {node_index}: '{node.name}': Connected to input no. {node_index}\n"
            else:
                for node_index, node in enumerate(layer):
                    result_string += f"Node no. {node_index}: '{node.name}'\n"
                    result_string += f"Bias: {node.bias}\n"
                    for weight_index, weight in enumerate(node.weights):
                        result_string += f"Weight no. {weight_index}: {weight}\n"
        return result_string

    def __repr__(self):
        return self.__str__()

