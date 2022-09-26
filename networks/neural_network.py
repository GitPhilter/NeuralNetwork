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


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


def get_new_weight(old_weight, eta, deriv_E_w):
    """Get new weight according to delta-function."""
    new_weight = old_weight - eta * deriv_E_w
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

    def train_by_avg_error(self, avg_error, eta):
        """output layer"""
        for output_node_index, output_node in enumerate(self.layers[-1]):
            deriv_E_node = -avg_error[output_node_index]
            output_node.deriv_e_node = deriv_E_node
            """adapt each weight"""
            for previous_node_index, previous_node in enumerate(self.layers[-2]):
                deriv_node_w = previous_node.output
                deriv_E_w = deriv_E_node * deriv_node_w
                new_weight = get_new_weight(output_node.weights[previous_node_index], eta, deriv_E_w)
                output_node.weights[previous_node_index] = new_weight
            """adapt bias"""
            new_bias = get_new_weight(output_node.bias, eta, deriv_E_node)
            output_node.bias = new_bias
        """hidden layers"""
        for layer_index in range(len(self.layers) - 2, 0, -1):
            for hidden_node_index, hidden_node in enumerate(self.layers[layer_index]):
                deriv_E_hidden_node = 0
                for following_node_index, following_node in enumerate(self.layers[layer_index + 1]):
                    respective_weight = following_node.weights[hidden_node_index]
                    deriv_E_hidden_node += following_node.deriv_e_node * respective_weight
                hidden_node.deriv_e_node = deriv_E_hidden_node
                """adapt weights"""
                for previous_node_index, previous_node in enumerate(self.layers[layer_index - 1]):
                    deriv_E_w = deriv_E_hidden_node * previous_node.output
                    new_weight = get_new_weight(hidden_node.weights[previous_node_index], eta, deriv_E_w)
                    hidden_node.weights[previous_node_index] = new_weight
                """adapt bias"""
                new_bias = get_new_weight(hidden_node.bias, eta, deriv_E_hidden_node)
                hidden_node.bias = new_bias

    def train(self, actual_output, expected_output, eta):
        """Train the neural network."""
        avg_error = []
        for i in range(0, len(actual_output)):
            avg_error.append(actual_output[i] - expected_output[i])
        self.train_by_avg_error(avg_error, eta)

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

