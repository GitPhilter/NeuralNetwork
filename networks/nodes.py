#!/usr/bin/env python
"""
Provide nodes for usage in a neural network.

Nodes basically have a name, an activation_function, weights and a bias.
Note that the bias is kept as a single variable and not a member of the list of weights. This makes it easier to
extract the bias from the outside without having to wonder at what index in the weights-list it might be.
The init-functions might differ according to what information is really unique to a given instance of a given type of
node.
Alongside the nodes are functions to generate random nodes, i.e. nodes with random weights and bias.
"""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

import random
from networks import activation_functions
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


class Node:
    """The base class for every node."""
    def __init__(self, name, activation_function, weights, bias):
        self.name = name
        self.activation_function = activation_function
        self.weights = weights
        self.fake_weights = weights.copy()
        self.bias = bias
        self.output = None
        self.total_input = None
        self.partial_derivs = []
        self.deriv_E_node = None  # the influence this node has on the overall error

    def compute_output(self, input_vector):
        """Compute the output of the node, given an input_vector."""
        if len(input_vector) != len(self.weights):
            logger.error("Cannot compute output: input_vector and weights are of different lengths.")
        input_sum = 0
        for index, iv in enumerate(input_vector):
            input_sum += iv * self.weights[index]
        self.total_input = input_sum + self.bias
        self.output = self.activation_function(self.total_input)


class Id_Node(Node):
    """Node with the Id-function as activation function."""
    def __init__(self, *, weights, bias, name):
        super().__init__(name, activation_functions.id_function, weights, bias)


def get_input_node_weights(index, number_of_inputs):
    """Create the list of weights for an Input_Node"""
    weights = []
    for i in range(0, number_of_inputs):
        if i == index:
            weights.append(1)
        else:
            weights.append(0)
    return weights


class Input_Node(Id_Node):
    """A simple Input_Node that only passes along one incoming input-value."""
    def __init__(self, *, index, number_of_inputs, name=None):
        weights = get_input_node_weights(index, number_of_inputs)
        bias = 0
        if name is None:
            name = f"Input Node {index}"
        super().__init__(weights=weights, bias=bias, name=name)
        self.index = index


class Relu_Node(Node):
    """Node with the ReLU-function as activation function."""
    def __init__(self, *, weights, bias, name=None):
        if name is None:
            name = "ReLU Node"
        super().__init__(name, activation_functions.relu_function, weights, bias)


def get_random_relu_node(number_of_inputs, name="ReLU_node"):
    """Return a ReLU_Node with random weights and bias."""
    weights = []
    for i in range(0, number_of_inputs):
        weight = random.random() * 2 - 1
        weights.append(weight)
    bias = random.random() * 2 - 1
    return Relu_Node(weights=weights,  bias=bias, name=name)


def get_zero_relu_node(number_of_inputs, name="ReLU_node"):
    """Return a ReLU_Node with all weights and bias set to zero."""
    weights = []
    for i in range(0, number_of_inputs):
        weights.append(0)
    return Relu_Node(weights=weights, bias=0, name=name)


class Sigmoid_Node(Node):
    """Node with the Sigmoid-function as activation function."""
    def __init__(self, *, weights, bias=0, name=None):
        if name is None:
            name = "Sigmoid Node"
        super().__init__(name, activation_functions.sigmoid_function, weights, bias)


def get_random_sigmoid_node(number_of_inputs, name="Sigmoid_Node", set_bias=False):
    """Return a Sigmoid_Node with random weights and bias."""
    weights = []
    for i in range(0, number_of_inputs):
        weight = random.random() * 2 - 1
        weights.append(weight)
    bias = 0
    if set_bias:
        bias = random.random() * 2 - 1
    return Sigmoid_Node(weights=weights, bias=bias, name=name)


def get_zero_sigmoid_node(number_of_inputs, name="Sigmoid_Node"):
    """Return a Sigmoid_Node with all weights and bias set to zero."""
    weights = []
    for i in range(0, number_of_inputs):
        weights.append(0)
    return Sigmoid_Node(weights=weights, bias=0, name=name)


def get_start_sigmoid_node(number_of_inputs, name="Sigmoid_Node"):
    """Return a Sigmoid_Node with all weights and bias set to 1/2."""
    weights = []
    for i in range(0, number_of_inputs):
        weights.append(0.5)
    return Sigmoid_Node(weights=weights, bias=0, name=name)
