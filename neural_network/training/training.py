#!/usr/bin/env python
"""Provide functions for training a neural network."""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

from random import randrange
from neural_network.training.data import DataObject
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


"""Threshold for clamping outputs to 0 and 1."""
THRESHOLD = 0.1
"""The learning rate eta."""
ETA = 0.001


def train_neural_network_single(network, data_object: DataObject):
    """Feed a DataObject to a network and train the network if the classification was incorrect."""
    network.compute_output(data_object.data)
    if is_within_threshold(network.output[0], data_object.expected_result):
        print(f"classification correct!")
    else:
        print(f"classification incorrect! Training network...")
        network.train(network.output, [data_object.expected_result], ETA)


def run_training_iterations(network, data_objects: [DataObject], iterations):
    """Run a number of training iterations given a list of DataObjects."""
    for i in range(0, iterations):
        print(f"Training iteration no. {i}")
        randex = randrange(len(data_objects))
        train_neural_network_single(network, data_objects[randex])


def is_within_threshold(actual_result, expected_result):
    """Check whether a result is close enough to an expected result to be considered equal."""
    if (actual_result - THRESHOLD < expected_result < actual_result) or \
            (actual_result + THRESHOLD > expected_result > actual_result):
        return True
    return False





