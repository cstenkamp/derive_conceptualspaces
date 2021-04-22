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
    rejection = ortho_rejection(first, second, candidate)
    if cosine_similiarity(second-first, candidate-first) > 0 and cosine_similiarity(first-second, candidate-second) > 0:
        return np.linalg.norm(rejection)
    return math.inf



def cosine_similiarity(first,second):
    """https://stackoverflow.com/a/43043160/5122790"""
    return np.dot(first, second)/(np.linalg.norm(first)*np.linalg.norm(second))


def ortho_rejection(first, second, candidate):
    """https://en.wikipedia.org/wiki/Vector_projection"""
    b = second-first
    a = candidate
    a1_sc = np.dot(a,(b/np.linalg.norm(b)))
    b_hat = b/np.linalg.norm(b)
    a1 = a1_sc*b_hat
    a2 = a-a1
    return a2*-1

def ortho_projection(first, second, candidate):
    """https://en.wikipedia.org/wiki/Vector_projection"""
    b = second-first
    a = candidate
    a1_sc = np.dot(a,(b/np.linalg.norm(b)))
    b_hat = b/np.linalg.norm(b)
    a1 = a1_sc*b_hat
    return a1