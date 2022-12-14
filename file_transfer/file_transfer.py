#!/usr/bin/env python
"""
Provide functions for saving and loading neural networks from/to a file.
"""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

from networks.neural_network import NeuralNetwork
from networks.discriminator_network import GreyscaleDiscriminatorNetwork
from networks.image_classification_network import ImageGreyscaleBinaryClassifyNetwork
from networks.generator_network import GreyscaleGeneratorNetwork
from networks.nodes import Input_Node, Relu_Node, Sigmoid_Node
import logging

from training.data import DataObject

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


def save_data_objects(data_objects, file_path):
    """Save a given data objects to a file defined by file_path."""
    logger.debug(f"Opening file '{file_path}'")
    file = open(file_path, 'w')
    #
    for do in data_objects:
        file.write(f"{do.name}\n{do.data}\n{do.expected_result}\n")
    #
    file.close()


def load_data_objects(file_path):
    """Load and return a list of data objects from file."""
    lines = open(file_path).readlines()
    data_objects = []
    for i in range(0, len(lines) - 1, 3):
        name = lines[i]
        tokens = lines[i + 1][1:-2].split(",")
        data = []
        for t in tokens:
            data.append(float(t))
        tokens = lines[i + 2][1:-2].split(",")
        expected_result = []
        for t in tokens:
            expected_result.append(float(t))
        data_objects.append(DataObject(name, data, expected_result))
    return data_objects


def save_network_to_file(network, file_path):
    """Save a given network to a file defined by file_path."""
    logger.debug(f"Opening file '{file_path}'")
    file = open(file_path, 'w')
    """Write network name and class."""
    file.write(f"network_name\n{network.name}\nnetwork_class\n{type(network).__name__}\n")
    """Write input layer"""
    file.write(f"number_of_input_nodes\n{len(network.layers[0])}\n")
    """Go through hidden and output layers"""
    for layer in network.layers[1:]:
        file.write("layer\n")
        for node in layer:
            file.write("node\n")
            """node_class"""
            node_class = type(node).__name__
            file.write(f"node_class\n{node_class}\n")
            """node name"""
            file.write(f"node_name\n{node.name}\n")
            """weights"""
            file.write("weights\n")
            weights_line = ""
            for weight in node.weights[:-1]:
                weights_line += str(weight) + ","
            weights_line += str(node.weights[-1]) + "\n"
            file.write(weights_line)
            """bias"""
            file.write(f"bias\n{node.bias}\n")
    logger.debug(f"Closing file '{file_path}'...")
    file.close()


def load_network_from_file(file_path):
    """Load a neural network from the file located at the given file_path."""
    lines = open(file_path, 'r').read().split("\n")
    """Initialize variables"""
    network_name = "? Unnamed Network ?"
    network_class = "NeuralNetwork"
    layers = []
    index = 0
    while index < len(lines):
        line = lines[index]
        print(line)
        if line == "network_class":
            network_class = lines[index + 1]
            index += 1
        if line == "network_name":
            network_name = lines[index + 1]
            index += 1
        elif line == "number_of_input_nodes":
            number_of_input_nodes = int(lines[index + 1])
            input_layer = _get_input_layer(number_of_input_nodes)
            layers.append(input_layer)
            index += 1
        elif line == "layer":
            """Parse layer"""
            index, new_layer = _parse_layer(index + 1, lines)
            layers.append(new_layer)
        else:
            index += 1
    if network_class == "GreyscaleDiscriminatorNetwork":
        return GreyscaleDiscriminatorNetwork(network_name, layers)
    if network_class == "ImageGreyscaleBinaryClassifyNetwork":
        return ImageGreyscaleBinaryClassifyNetwork(network_name, layers)
    if network_class == "GreyscaleGeneratorNetwork":
        return GreyscaleGeneratorNetwork(network_name, layers)
    return NeuralNetwork(network_name, layers)


def _get_input_layer(number_input_nodes):
    """Return the input layer for a given number of input nodes."""
    input_layer = []
    for i in range(0, number_input_nodes):
        input_layer.append(Input_Node(index=i, number_of_inputs=number_input_nodes, name=f"Input Node {i}"))
    return input_layer


def _parse_layer(index, lines):
    """Parse a layer from the given lines, starting at the given index.."""
    line = lines[index]
    layer = []
    while line != "layer" and index < len(lines):
        if line == "node":
            index, new_node = _parse_node(index + 1, lines)
            layer.append(new_node)
        else:
            index += 1
        if index < len(lines):
            line = lines[index]
    return index, layer


def _parse_node(index, lines):
    """Parse a node from the given lines, starting at the given index."""
    node_class = None
    node_name = None
    weights = []
    bias = None
    line = lines[index]
    while line != "node" and index < len(lines):
        if line == "node_class":
            node_class = lines[index + 1]
            index += 2
        elif line == "node_name":
            node_name = lines[index + 1]
            index += 2
        elif line == "weights":
            weights = _parse_weights(lines[index + 1])
            index += 2
        elif line == "bias":
            bias = float(lines[index + 1])
            index += 2
        else:
            index += 1
        if index < len(lines):
            line = lines[index]
    new_node = globals()[node_class](weights=weights, bias=bias, name=node_name)
    return index - 1, new_node


def _parse_weights(weights_string):
    """Parse weights from the given string."""
    weights = []
    tokens = weights_string.split(",")
    for token in tokens:
        weights.append(float(token))
    return weights


def _compare_two_networks(network_1, network_2):
    """Compare two networks. Used for debugging and testing."""
    strings_1 = str(network_1).split("\n")
    strings_2 = str(network_2).split("\n")
    if len(strings_1) != len(strings_2):
        logger.error(f"len(strings_1) = {len(strings_1)} != len(strings_2) = {len(strings_2)}")
        return False
    for i in range(0, len(strings_1)):
        if strings_1[i] != strings_2[i]:
            logger.error(f"line '{strings_1[i]}' != '{strings_2[i]}'")
            return False
    return True
