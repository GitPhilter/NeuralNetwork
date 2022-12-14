import math

import PIL.Image
from networks.neural_network import NeuralNetwork
from networks.nodes import Input_Node, get_random_relu_node, get_random_sigmoid_node


def get_GreyscaleGeneratorNetwork(name, result_image_width, result_image_height, number_of_input_nodes=10):
    #self.result_image_width = result_image_width
    #self.result_mage_height = result_image_height
    """Determine number of layers."""
    number_of_output_nodes = result_image_width * result_image_height
    number_of_layers = math.ceil(math.log(number_of_output_nodes - number_of_input_nodes))
    nodes_per_layer = []
    for i in range(1, number_of_layers):
        number_of_nodes = number_of_input_nodes + math.ceil(math.exp(i))
        nodes_per_layer.append(number_of_nodes)
    layers = []
    """Create input layer."""
    input_layer = []
    for i in range(0, number_of_input_nodes):
        input_layer.append(Input_Node(index=i, number_of_inputs=number_of_input_nodes, name=f"Input Node {i}"))
    layers.append(input_layer)
    """create hidden and layers."""
    number_of_previous_nodes = number_of_input_nodes
    for i in range(0, number_of_layers - 1):
        layer = []
        for n in range(0, nodes_per_layer[i]):
            layer.append(get_random_relu_node(number_of_previous_nodes))
        number_of_previous_nodes = nodes_per_layer[i]
        layers.append(layer)
    """create output layer"""
    output_layer = []
    for i in range(0, number_of_output_nodes):
        output_layer.append(get_random_sigmoid_node(number_of_previous_nodes))
    layers.append(output_layer)
    greyscale_generator_network = GreyscaleGeneratorNetwork(name, layers)
    greyscale_generator_network.result_image_width = result_image_width
    greyscale_generator_network.result_mage_height = result_image_height
    greyscale_generator_network.number_of_inputs = len(input_layer)
    return greyscale_generator_network


def get_small_GreyscaleGeneratorNetwork(name, result_image_width, result_image_height, number_of_input_nodes=10):
    #self.result_image_width = result_image_width
    #self.result_mage_height = result_image_height
    """Determine number of layers."""
    number_of_output_nodes = result_image_width * result_image_height
    # number_of_layers = math.ceil(math.log(number_of_output_nodes - number_of_input_nodes))
    number_of_layers = 3
    nodes_per_layer = []
    for i in range(1, number_of_layers):
        number_of_nodes = number_of_input_nodes + math.ceil(math.exp(i))
        nodes_per_layer.append(number_of_nodes)
    layers = []
    """Create input layer."""
    input_layer = []
    for i in range(0, number_of_input_nodes):
        input_layer.append(Input_Node(index=i, number_of_inputs=number_of_input_nodes, name=f"Input Node {i}"))
    layers.append(input_layer)
    """create hidden layers."""
    number_of_previous_nodes = number_of_input_nodes
    """
    for i in range(0, number_of_layers - 1):
        layer = []
        for n in range(0, nodes_per_layer[i]):
            layer.append(get_random_relu_node(number_of_previous_nodes))
        number_of_previous_nodes = nodes_per_layer[i]
        layers.append(layer)
    """
    """create output layer"""
    output_layer = []
    for i in range(0, number_of_output_nodes):
        output_layer.append(get_random_sigmoid_node(number_of_previous_nodes))
    layers.append(output_layer)
    greyscale_generator_network = GreyscaleGeneratorNetwork(name, layers)
    greyscale_generator_network.result_image_width = result_image_width
    greyscale_generator_network.result_mage_height = result_image_height
    greyscale_generator_network.number_of_inputs = len(input_layer)
    return greyscale_generator_network



class GreyscaleGeneratorNetwork(NeuralNetwork):

    """
    A generator network for usage in GANs.

    It uses the fixed number of 100 as a size for the random input vector.
    100 seems to be a magical number that many networks use. This is totally arbitrary and might be
    changed in later networks. But to get things started, just use 10, as the images that will be
    created in the beginning are going to be small to get faster results instead of "large" results.
    The number of hidden layers is determined by the natural logarithm of the number of pixels generated minus
    the number of input variables.
    """
    def __init__(self, name, layers):
        self.result_image_width = -1
        self.result_mage_height = -1
        self.number_of_inputs = -1
        super().__init__(name, layers)

    def get_image_from_output(self):
        image = PIL.Image.new('RGB', (self.result_image_width, self.result_mage_height))
        pixel_counter = 0
        for x in range(0, self.result_image_width):
            for y in range(0, self.result_mage_height):
                greyscale_value = round(self.output[pixel_counter] * 255)
                #print(greyscale_value)
                image.putpixel((x, y), (greyscale_value, greyscale_value, greyscale_value))
                pixel_counter += 1
        return image


if __name__=="__main__":
    network = get_GreyscaleGeneratorNetwork("hallo", 32, 32, 10)
    #print(network)
    input = [0.5, 0.1, 0.6, 0, 0, 0.3, 0.7, 0.8, 0.9, 0.24]
    network.compute_output(input)
    #print(network.output)
    image = network.get_image_from_output()
    image.save("/home/kwetschmann/PycharmProjects/NeuralNetwork/image_1.png", 'PNG')