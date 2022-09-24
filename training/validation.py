"""Threshold for clamping outputs to 0 and 1."""
import random

THRESHOLD = 0.1


def get_number_of_correct_predictions(network, data_objects, iterations):
    correct_predictions = 0
    for i in range(0, iterations):
        index = random.randint(0, len(data_objects) - 1)
        was_correct = validate_binary_prediction(network, data_objects[index])
        if was_correct:
            correct_predictions += 1
    return correct_predictions


def validate_binary_prediction(network, data_object):
    output = network.compute_output(data_object.data)
    for index, o in enumerate(output):
        if o > 0.5 and data_object.expected_result[index] == 0:
            return False
        if o <= 0.5 and data_object.expected_result[index] == 1:
            return False
    return True


def get_error_from_single_training(network, data_object):
    """Feed a DataObject to a network and train the network if the classification was incorrect."""
    output = network.compute_output(data_object.data)
    expected_output = data_object.expected_result
    error = []
    for index, op in enumerate(output):
        error.append(output[index] - expected_output[index])
    return error


def is_within_threshold(actual_result, expected_result):
    """Check whether a result is close enough to an expected result to be considered equal."""
    if (actual_result - THRESHOLD < expected_result < actual_result) or \
            (actual_result + THRESHOLD > expected_result > actual_result):
        return True
    return False