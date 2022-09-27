import random

from networks.neural_network import NeuralNetwork, get_new_weight
from networks.nodes import Input_Node, get_random_sigmoid_node
from training.data import DataObject
from training.training import train_neural_network_single
from networks.activation_functions import get_activation_function_deriv


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
        input_layer.append(Input_Node(index=i, number_of_inputs=number_of_nodes, name=f"Input Node {i}"))
        relu_name = "Hidden_Node no. " + str(i)
        # hidden_layer.append(get_random_relu_node(number_of_nodes, name=relu_name))
    output_layer = [get_random_sigmoid_node(number_of_nodes, "Output_Sigmoid_Node")]
    # super().__init__(name, [input_layer, hidden_layer, output_layer])
    return GreyscaleDiscriminatorNetwork(name, [input_layer, output_layer])


class GreyscaleDiscriminatorNetwork(NeuralNetwork):
    """
    A Discriminator-network for usage in a GAN.

    First Draft: Use 1 hidden layer. If this is not working, the number can be increased.
    """

    def __init__(self, name, layers):
        super().__init__(name, layers)

    def get_error_vector(self, fake_image_vector, eta=ETA):
        """
        Pass a fake_image to the network and return the error vector, that is used to train the generator.

        The vector used here is supposed to be created by a generator that uses for example the Sigmoid-function,
        so that all input values are between 0 and 1.
        """
        actual_output = self.compute_output(fake_image_vector)
        expected_output = [1]
        if actual_output[0] < 0.5:
            error_derivs = []
            for index, ao in enumerate(actual_output):
                """Append the derivatives of the loss function to the error_derivs."""
                # print(f"Appending error function for output layer: {ao - expected_output[index]}")
                """Trying out the cross entropy derivation"""
                # error_derivs.append(ao - expected_output[index])
                error_derivs.append(-((ao - expected_output[index]) / (0.001 + (ao - 1) * ao)))
            """Go through all layers."""
            for layer_index in range(len(self.layers) - 1, 0, -1):
                next_error_derivs = []
                for node_index, node in enumerate(self.layers[layer_index]):
                    # print(f"Adapting node: {node.name}")
                    """Compute the result of the inner derivation of the activation function"""
                    node_inner_deriv = get_activation_function_deriv(node.activation_function, node.total_input)
                    # print(f"Inner deriv: {node_inner_deriv}")
                    for weight_index, weight in enumerate(node.weights):
                        """The outer derivative is always the output of the respective node."""
                        node_outer_deriv = self.layers[layer_index - 1][weight_index].output
                        # print(f"Weight index: {weight_index}, outer_deriv: {node_outer_deriv}")
                        partial_derivation = node_outer_deriv * node_inner_deriv * error_derivs[node_index]
                        # print(f"Partial derivation: {partial_derivation}")
                        next_error_derivs.append(partial_derivation)
                        """Adapt the weight."""
                        # print(f"Adapting the weight from {node.weights[weight_index]} to...")
                        node.fake_weights[weight_index] = get_new_weight(node.weights[weight_index], eta, partial_derivation)
                        # print(f"New weight: {node.weights[weight_index]}")
                    """adapt bias"""
                    partial_derivation = 1 * node_inner_deriv * error_derivs[node_index]
                    # print(f"Partial derivation for bias: {partial_derivation}")
                    # print(f"Adapting bias from: {node.bias}...")
                    node.fake_bias = get_new_weight(node.bias, eta, partial_derivation)
                    # print(f"to new bias: {node.bias}")
                error_derivs = next_error_derivs
            error_vector = []
            for index in range(len(self.layers[0])):
                sum = 0
                for node in self.layers[1]:
                    sum += node.weights[index] - node.fake_weights[index]
                error_vector.append(sum)
            print(f"Returning error vector: {error_vector}")
            return error_vector

    def backpropagation(self, actual_output, expected_output, eta):
        """
        Apply backpropagation with gradient descent and change weights and biases accordingly.

        As the error function the cross entropy is used. l = -(y log(p) + (1 - y)log(1 - p))
        """
        """Compute the total error / loss function."""
        error_derivs = []
        for index, ao in enumerate(actual_output):
            """Append the derivatives of the loss function to the error_derivs."""
            #print(f"Appending error function for output layer: {ao - expected_output[index]}")
            """Trying out the cross entropy derivation"""
            #error_derivs.append(ao - expected_output[index])
            error_derivs.append(-((ao - expected_output[index]) / (0.001 + (ao - 1)*ao)))
        """Go through all layers."""
        for layer_index in range(len(self.layers) - 1, 0, -1):
            next_error_derivs = []
            for node_index, node in enumerate(self.layers[layer_index]):
                #print(f"Adapting node: {node.name}")
                """Compute the result of the inner derivation of the activation function"""
                node_inner_deriv = get_activation_function_deriv(node.activation_function, node.total_input)
                #print(f"Inner deriv: {node_inner_deriv}")
                for weight_index, weight in enumerate(node.weights):
                    """The outer derivative is always the output of the respective node."""
                    node_outer_deriv = self.layers[layer_index - 1][weight_index].output
                    #print(f"Weight index: {weight_index}, outer_deriv: {node_outer_deriv}")
                    partial_derivation = node_outer_deriv * node_inner_deriv * error_derivs[node_index]
                    #print(f"Partial derivation: {partial_derivation}")
                    next_error_derivs.append(partial_derivation)
                    """Adapt the weight."""
                    #print(f"Adapting the weight from {node.weights[weight_index]} to...")
                    node.weights[weight_index] = get_new_weight(node.weights[weight_index], eta, partial_derivation)
                    #print(f"New weight: {node.weights[weight_index]}")
                """adapt bias"""
                partial_derivation = 1 * node_inner_deriv * error_derivs[node_index]
                #print(f"Partial derivation for bias: {partial_derivation}")
                #print(f"Adapting bias from: {node.bias}...")
                node.bias = get_new_weight(node.bias, eta, partial_derivation)
                #print(f"to new bias: {node.bias}")
            error_derivs = next_error_derivs

    def online_training(self, data_object, eta):
        #print("Online Training in Discriminator")
        output = self.compute_output(data_object.data)
        if output[0] < 0.5 and data_object.expected_result[0] == 1 or \
                output[0] >= 0.5 and data_object.expected_result[0] == 0:
            self.backpropagation(output, data_object.expected_result, eta)

    def fake_image_not_classified_as_real(self):
        if self.output[0] < 0.5:
            return True
        return False

    def fake_image_classified_as_fake(self):
        if self.output - THRESHOLD <= 0:
            return True
        return False


if __name__ == "__main__":
    D_network = GreyscaleDiscriminatorNetwork("test", 2, 2)
    print(D_network)
    real_input_vectors = [[255, 255, 0, 0],
                          [254, 255, 1, 0],
                          [255, 254, 1, 0],
                          [255, 255, 1, 1],
                          [255, 254, 0, 1]]
    fake_input_vectors = [[120, 120, 120, 120],
                          [0, 0, 255, 255],
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
