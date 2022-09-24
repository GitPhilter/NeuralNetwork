#!/usr/bin/env python
"""Provide class DataObject for training of a neural network."""

__author__ = "Mike Grätz"
__date__ = "09-22-22"


class DataObject:
    """A piece of data used for training or evaluating a neural network."""
    def __init__(self, name, data, expected_result):
        """.
        :param name: name of the underlying object (for example the name of the respective image)
        :param data: a vector that is suitable as an input for a neural network
        :param expected_result: the expected result, for example '0' or '1' for binary classification problems.
        """
        self.name = name
        self.data = data
        self.expected_result = expected_result

    def __str__(self):
        return "data: " + self.name + " -> expected result: " + str(self.expected_result)

    def __repr__(self):
        return self.__str__()

