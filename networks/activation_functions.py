#!/usr/bin/env python
"""Provide activation functions to be used in nodes in a neural network."""

__author__ = "Mike Grätz"
__date__ = "09-22-22"

from math import exp


def id_function(x):
    """The Id-function."""
    return x


def id_function_deriv(x):
    return 1


def relu_function(x):
    """The ReLU-function."""
    return max(0, x)


def relu_function_deriv(x):
    if x > 0:
        return 1
    return 0


def sigmoid_function(x):
    """The Sigmoid-function."""
    try:
        exp_term = exp(-x)
    except OverflowError:
        if x < 0:
            return 0
        if x >= 0:
            return 1
    return 1 / (1 + exp_term)


def sigmoid_function_deriv(x):
    try:
        exp_term = exp(-x)
    except OverflowError:
        if x < 0:
            return 1
        if x >= 0:
            return 0
    return exp_term / (exp_term + 1)
