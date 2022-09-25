import random

from networks.neural_network import NeuralNetwork, get_new_weight
from networks.nodes import Input_Node, get_random_sigmoid_node
from training.data import DataObject
from training.training import train_neural_network_single

"""
NOTE:
1 ~ real image
0 ~ fake image
"""
THRESHOLD = 0.1
ETA = 0.01


def get_GreyscaleDiscriminatorNetwork(name, image_width, image_height):
    number_of_nodes = image_width * image_height
    input_layer = []
    # hidden_layer = []
    for i in range(0, number_of_nodes):
        input_layer.append(Input_Node(i, number_of_nodes))
        relu_name = "Hidden_Node no. " + str(i)
        # hidden_layer.append(get_random_relu_node(number_of_nodes, name=relu_name))
    output_layer = [get_random_sigmoid_node(number_of_nodes, "Output_Relu_Node")]
    # super().__init__(name, [input_layer, hidden_layer, output_layer])
    return GreyscaleDiscriminatorNetwork(name, [input_layer, output_layer])


class GreyscaleDiscriminatorNetwork(NeuralNetwork):
    """
    A Discriminator-network for usage in a GAN.

    First Draft: Use 1 hidden layer. If this is not working, the number can be increased.
    """
    def __init__(self, name, layers):
        super().__init__(name, layers)

    def get_error_vector(self, fake_image_vector):
        """
        Pass a fake_image to the network and return the error vector, that is used to train the generator.

        The vector used here is supposed to be created by a generator that uses for example the Sigmoid-function,
        so that all input values are between 0 and 1.
        """
        for layer in self.layers:
            for node in layer:
                node.fake_weights = node.weights.copy()
        self.compute_output(fake_image_vector)
        if self.fake_image_not_classified_as_real():
            """output layer"""
            for output_node_index, output_node in enumerate(self.layers[-1]):
                output_node.fake_weights = output_node.weights.copy()
                #actual_node_output = actual_output[output_node_index]
                #expected_node_output = expected_output[output_node_index]
                #deriv_E_node = actual_node_output - expected_node_output
                deriv_E_node = self.output[0] - 1
                output_node.fake_deriv_e_node = deriv_E_node
                """adapt each weight"""
                for previous_node_index, previous_node in enumerate(self.layers[-2]):
                    deriv_node_w = previous_node.output
                    deriv_E_w = deriv_E_node * deriv_node_w
                    new_weight = get_new_weight(output_node.weights[previous_node_index], ETA, deriv_E_w)
                    output_node.fake_weights[previous_node_index] = new_weight
                """adapt bias"""
                #new_bias = get_new_weight(output_node.bias, ETA, deriv_E_node)
                #output_node.bias = new_bias
            """hidden layers"""
            for layer_index in range(len(self.layers) - 2, 0, -1):
                for hidden_node_index, hidden_node in enumerate(self.layers[layer_index]):
                    deriv_E_hidden_node = 0
                    for following_node_index, following_node in enumerate(self.layers[layer_index + 1]):
                        respective_weight = following_node.fake_weights[hidden_node_index]
                        deriv_E_hidden_node += following_node.fake_deriv_e_node * respective_weight
                    hidden_node.fake_deriv_E_node = deriv_E_hidden_node
                    """adapt weights"""
                    for previous_node_index, previous_node in enumerate(self.layers[layer_index - 1]):
                        deriv_E_w = deriv_E_hidden_node * hidden_node.fake_weights[previous_node_index]
                        new_weight = get_new_weight(hidden_node.weights[previous_node_index], ETA, deriv_E_w)
                        """Dirty way of clamping weights to range [-100, 100]."""
                        if new_weight > 100:
                            new_weight = float(100)
                        if new_weight < -100:
                            new_weight = float(-100)
                        hidden_node.fake_weights[previous_node_index] = new_weight
                    """adapt bias"""
                    #new_bias = get_new_weight(hidden_node.bias, eta, deriv_E_hidden_node)
                    #hidden_node.bias = new_bias
            error_vector = []
            for index, input_node in enumerate(self.layers[0]):
                gradient_sum = 0
                for second_node in self.layers[1]:
                    gradient_sum += second_node.fake_weights[index] - second_node.weights[index]
                error_vector.append(gradient_sum)
            return error_vector
        return None


    def fake_image_not_classified_as_real(self):
        if self.output[0] < 0.5:
            return True
        return False

    def fake_image_classified_as_fake(self):
        if self.output - THRESHOLD <= 0:
            return True
        return False


if __name__=="__main__":
    D_network = GreyscaleDiscriminatorNetwork("test", 2, 2)
    print(D_network)
    real_input_vectors = [[255, 255, 0, 0],
                          [254, 255, 1, 0],
                          [255, 254, 1, 0],
                          [255, 255, 1, 1],
                          [255, 254, 0, 1]]
    fake_input_vectors = [[120, 120, 120, 120],
                          [0, 0 , 255, 255],
                          [0, 100, 50, 20],
                          [100, 100, 100, 0],
                          [0, 0, 1, 0],
                          [12, 11, 10, 8]]
    data_objects = []
    for rv in real_input_vectors:
        data_objects.append(DataObject("real data example", rv, 1))
    for fv in fake_input_vectors:
        data_objects.append(DataObject("fake data example", fv, 0))
    for i in range(0, 100000):
        randex = random.randint(0, len(data_objects) - 1)
        D_network.compute_output(data_objects[randex].data)
        train_neural_network_single(D_network, data_objects[randex])

    generator_generated_data = [120, 120, 120, 120]
    error_vector = D_network.get_error_vector(generator_generated_data)
    print(error_vector)

