# -*- coding: utf-8 -*-


# Python packages
import matplotlib.pyplot
import numpy
import os


# MRG packages
import _env
import preprocessing
import processing
import postprocessing
#import solutions

import matplotlib.pyplot as plt


if __name__ == '__main__':
    N = 50  # number of points along x-axis
    M = 2 * N  # number of points along y-axis
    level = 0  # level of the fractal

    # -- set geometry of domain
    domain_omega, x, y, _, _ = preprocessing._set_geometry_of_domain(
        M, N, level)
    plt.imshow(domain_omega)
    plt.show()

    print('End.')
