import math

import numpy as np

######################### simple similiarity #########################

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

######################### between_a #########################

def between_a(first, second, candidate):
    """
    First betweenness-measure from [DESC15]. Since it is unlikely that a point b will be perfectly between two other points
    a and c in S, we measure degree of betweenness. For that we take into account the Collinearity $Col(a,b,c) = ∥\overrightarrow{bp}∥$,
    where p is the orthogonal projection of b on the line connecting a and c. Further we need to check if p is between a and c by
    cos(\overrightarrow{ac},\overrightarrow{ab}) >= 0 and cos(\overrightarrow{ca},\overrightarrow{cb}) >= 0.
    If between_a == 0, then c is perfectly between a and b, higher scores mean weaker similiarity.
    See [DESC15] Section 4.1
    See https://python-advanced.quantecon.org/orth_proj.html#The-Orthogonal-Projection-Theorem: y^:=\argmin_{z∈S}∥y−z∥
    #...heavily visualized in visualize_projection.ipynb
    :param first:
    :param second:
    :param candidate:
    :return:
    """
    rejection = ortho_rejection(first-second, candidate-second)
    if cosine_similiarity(second-first, candidate-first) > 0 and cosine_similiarity(first-second, candidate-second) > 0:
        return np.linalg.norm(rejection)
    return math.inf

def cosine_similiarity(first,second):
    """https://stackoverflow.com/a/43043160/5122790"""
    return np.dot(first, second)/(np.linalg.norm(first)*np.linalg.norm(second))


def ortho_rejection(first, candidate):
    """https://en.wikipedia.org/wiki/Vector_projection"""
    b = first
    a = candidate
    a1_sc = np.dot(a,(b/np.linalg.norm(b)))
    b_hat = b/np.linalg.norm(b)
    a1 = a1_sc*b_hat
    a2 = a-a1
    return a2*-1

def ortho_projection(first, candidate):
    """https://en.wikipedia.org/wiki/Vector_projection"""
    b = first
    a = candidate
    a1_sc = np.dot(a,(b/np.linalg.norm(b)))
    b_hat = b/np.linalg.norm(b)
    a1 = a1_sc*b_hat
    return a1

######################### between_b #########################

def between_b(first, second, candidate):
    """
    Second betweenness-measure from [DESC15]. Based on the observation that $∥\overrightarrow{ac}∥$ <= $∥\overrightarrow{ab}∥$ + $∥\overrightarrow{bc}∥$
    (by the triangle inequality), $∥\overrightarrow{ac}∥$ = $∥\overrightarrow{ab}∥$ + $∥\overrightarrow{bc}∥$ iff b is exactly between a and c
    So, between_b(a,b,c) = $∥\overrightarrow{ac}∥$ / ($∥\overrightarrow{ab}∥$ + $∥\overrightarrow{bc}∥$).
    NOTE: there is an ERROR in the paper, it says between_b(a,b,c) = $∥\overrightarrow{aB}∥$ / ($∥\overrightarrow{ab}∥$ + $∥\overrightarrow{bc}∥$) !!!
    In contrast to btween_a,higher values for between_b represent a stronger betweenness relation, with a score of 1
    denoting perfect betweenness. With this alternative definition, points near a or b will get some degree of betweenness,
    even if their projection p is not between a and b.
    See [DESC15] Section 4.1
    :param first:
    :param second:
    :param candidate:
    :return:
    """
    len_ab = np.linalg.norm(candidate-first)
    len_ac = np.linalg.norm(second-first)
    len_bc = np.linalg.norm(second-candidate)
    return len_ac/(len_ab+len_bc)

def between_b_inv(*args):
    """inverse of between_b, such that low values mean high betweenness as it does in between_a"""
    return 1-between_b(*args)