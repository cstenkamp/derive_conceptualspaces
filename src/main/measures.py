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


def between_a(first, second, candidate):
    """
    First betweenness-measure from [DESC15]. Since it is unlikely that a point b will be perfectly between two other points
    a and c in S, we measure degree of betweenness. For that we take into account the Collinearity $Col(a,b,c) = ∥\overrightarrow{bp}∥$,
    where p is the orthogonal projection of b on the line connecting a and c. Further we need to check if p is between a and c by
    cos(\overrightarrow{ac},\overrightarrow{ab}) >= 0 and cos(\overrightarrow{ca},\overrightarrow{cb}) >= 0.
    If between_a == 0, then c is perfectly between a and b, higher scores mean weaker similiarity.
    See [DESC15] Section 4.1
    See https://python-advanced.quantecon.org/orth_proj.html#The-Orthogonal-Projection-Theorem: y^:=\argmin_{z∈S}∥y−z∥
    :param first:
    :param second:
    :param candidate:
    :return:
    """
    projection = orthogonal_project(first, second, candidate)



def orthogonal_project(first, second, candidate):
    """see https://www.geeksforgeeks.org/vector-projection-using-python/"""
    vector_first_second = second-first #with startingpoint first
    # Task: Project vector candidate onto vector_first_second
    v_norm = np.sqrt(sum(vector_first_second**2))
    projected  = (np.dot(candidate, vector_first_second)/v_norm**2)*vector_first_second
    return projected