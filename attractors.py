#!/usr/bin/env python3.11
"""Define different kinds of attractors."""
import numpy as np


def clifford_attractor(a, b, c, d):
    def inner(point):
        new_x = np.sin(a * point[:, 1]) + c * np.cos(a * point[:, 0])
        new_y = np.sin(b * point[:, 0]) + d * np.cos(b * point[:, 1])
        return np.array([new_x, new_y]).T

    return inner


def modified_clifford_attractor(a, b, c, d):
    def approxsin(v):
        return v - v**3 / (3 * 2 * 1) + v**5 / (5 * 4 * 3 * 2 * 1) - v**7 / (7 * 6 * 5 * 4 * 3 * 2 * 1)

    def inner(point):
        new_x = np.sin(a * point[:, 1]) + c * np.cos(a * point[:, 0])
        new_y = approxsin(b * point[:, 0]) + d * approxsin(b * point[:, 1])

        # x - x³/6 + x⁵/120

        # sum_{n=0}^{infty}(-1)^n{x^{2n+1}}/{(2n+1)!}
        return np.array([new_x, new_y]).T

    return inner


def jason_rampe_3_attractor(a, b, c, d):
    def inner(point):
        new_x = np.sin(b * point[:, 1]) + c * np.cos(b * point[:, 0])
        new_y = np.cos(a * point[:, 0]) + d * np.sin(a * point[:, 1])
        return np.array([new_x, new_y]).T

    return inner
