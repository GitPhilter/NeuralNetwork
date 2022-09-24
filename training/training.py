#!/usr/bin/env python
"""Provide functions for training a neural network."""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

import random
from random import randrange
from training.data import DataObject
import logging

from training.validation import get_error_from_single_training

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)


def set_logger(_logger, log_level=logging.INFO):
    """Set the logger for all networks."""
    global logger
    logger = _logger
    logger.setLevel(log_level)


"""The learning rate eta."""
ETA = 0.001


def train_neural_network_single(network, data_object: DataObject, eta=ETA):
    """Feed a DataObject to a network and train the network if the classification was incorrect."""
    network.compute_output(data_object.data)
    if network.output[0] >= 0.5 and data_object.expected_result == 1:
        return True
    elif network.output[0] < 0.5 and data_object.expected_result == 0:
        return True
    network.train(network.output, [data_object.expected_result], eta)
    return False


def run_training_iterations(network, data_objects: [DataObject], iterations):
    """Run a number of single training iterations given a list of DataObjects."""
    for i in range(0, iterations):
        # print(f"Training iteration no. {i}")
        randex = randrange(len(data_objects))
        train_neural_network_single(network, data_objects[randex])


def run_batch(network, data_objects: [DataObject], iterations, eta=ETA):
    """Run a batch of training iterations given a list of DataObjects."""
    error_sums = []
    for i in range(0, len(network.layers[-1])):
        error_sums.append(0)
    for i in range(0, iterations):
        # print(f"Training iteration no. {i}")
        randex = randrange(len(data_objects))
        error = get_error_from_single_training(network, data_objects[randex])
        for index, e in enumerate(error):
            error_sums[index] += e
    network.train_by_avg_error(error_sums, eta)
















