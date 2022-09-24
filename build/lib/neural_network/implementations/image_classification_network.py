#!/usr/bin/env python
"""
First attempt at creating a neural network that is able to decide whether a 16x16 pixel shows a circle.
The scenario is that any given original images have been prepared to fit into this NN.
For example, a large circle has been detected and fit inside a bounding box and then convoluted to a 16x16 picture.
"""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

from networks import NeuralNetwork
from networks.nodes import Input_Node, get_random_relu_node, get_random_sigmoid_node, \
    get_zero_relu_node, get_zero_sigmoid_node


class ImageGreyscaleBinaryClassifyNetwork(NeuralNetwork):

    def __init__(self, name, image_width, image_height, random_weights=True):
        number_of_nodes = image_width * image_height
        input_layer = []
        hidden_layer = []
        for i in range(0, number_of_nodes):
            input_layer.append(Input_Node(i, number_of_nodes))
            relu_name = "Hidden_Node no. " + str(i)
            if random_weights:
                hidden_layer.append(get_random_relu_node(number_of_nodes, name=relu_name))
            else:
                hidden_layer.append(get_zero_relu_node(number_of_nodes, name=relu_name))
        if random_weights:
            output_layer = [get_random_sigmoid_node(number_of_nodes, "Output_Relu_Node", set_bias=False)]
        else:
            output_layer = [get_zero_sigmoid_node(number_of_nodes, "Output_Relu_Node")]
        super().__init__(name, [input_layer, hidden_layer, output_layer])


