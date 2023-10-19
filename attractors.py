#!/usr/bin/env python3.11
""""""
import numpy as np


def clifford_attractor(a, b, c, d):
    # x and y both start at 0.1
    #
    # x_new=sin(a*y)+c*cos(a*x)
    # y_new=sin(b*x)+d*cos(b*y)
    #
    # Variables a,b,c and d are floating point values bewteen -3 and +3

    def inner(point):
        new_x = np.sin(a * point[:, 1]) + c * np.cos(a * point[:, 0])
        new_y = np.sin(b * point[:, 0]) + d * np.cos(b * point[:, 1])
        return np.array([new_x, new_y]).T

    return inner
