import math

import numpy as np

def simple_similiarity(first, second, lamb=0.5):
    """
    Simple semantic similarity for Euclidian Spaces.
    See [DESC15] Section 4.0
    #TODO any other source/maths for this?
    :param first:
    :param second:
    :param lamb: lambda for similarity-measure
    :return:
    """
    assert lamb > 0
    return math.exp((-1*lamb)*np.linalg.norm(first-second))