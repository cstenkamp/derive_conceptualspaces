import numpy as np

from main.load_semanticspaces import load_mds_representation, get_names
from static.settings import DATA_BASE, DATA_SET, MDS_DIMENSIONS

from main.measures import simple_similiarity, ortho_rejection, ortho_projection, between_a

def test_simplesimiliarity_for_movies():
    for n_dims in [20,50,100,200]:
        tmp = load_mds_representation(DATA_BASE, "movies", n_dims)
        tmp2 = get_names(DATA_BASE, "movies")
        name_mds = dict(zip(tmp2, tmp))
        starwars = [i for i in name_mds.keys() if "Star Wars" in i]
        starwarssims = [simple_similiarity(name_mds[starwars[0]], name_mds[starwars[i]]) for i in range(1, len(starwars))]
        # print("Star Wars to Star Wars avg Similarity:", sum(starwarssims) / len(starwarssims))
        othersims = [simple_similiarity(name_mds[starwars[0]], i) for i in name_mds.values()]
        # print("Star Wars to All Others avg Similarity:", sum(othersims) / len(othersims))
        assert sum(starwarssims) / len(starwarssims) > sum(othersims) / len(othersims)


def test_ortho_rejection_projection_numbers():
    test_sets = [
        (np.array([3,5]), np.array([8,2]), np.array([3,3]), np.array([-2.11764, -3.52941]), np.array([ 0.88235, -0.52941])),
        (np.array([-1,5]), np.array([1,2]), np.array([-1,3]), np.array([-0.69231, -0.46154]), np.array([-1.69231, 2.53846])),
        (np.array([3,5]), np.array([8,2]), np.array([-1,3]), np.array([-1.05882, -1.76471]), np.array([-2.05882, 1.23529])),
    ]
    for a, b, c, rej, proj in test_sets:
        assert np.allclose(ortho_rejection(a,b,c), rej, atol=0.0001), f"Orthogonal Rejection seems wrong! Should be {rej} but is {ortho_rejection(a,b,c)}"
        assert np.allclose(ortho_projection(a, b, c), proj, atol=0.0001), f"Orthogonal Rejection seems wrong! Should be {proj} but is {ortho_projection(a,b,c)}"


def test_between_a_for_numbers():
    test_sets = [
        (np.array([1,1]), np.array([3,3]), np.array([2.5,2.5]), 6.280369834735101e-16),
        (np.array([1,1]), np.array([3,3]), np.array([2.5,2.6]), 0.07071067811865482),
        (np.array([1,1]), np.array([3,3]), np.array([2,3]), 0.7071067811865476),
        (np.array([1,1]), np.array([3,3]), np.array([4,4]), np.inf),
        (np.array([1,1]), np.array([3,3]), np.array([-1,1]), np.inf),
        (np.array([0.01,0.01]), np.array([3,3]), np.array([-1,1.1]), 1.4849242404917498),
        (np.array([0.01,0.01]), np.array([3,3]), np.array([-1,1]), np.inf),
        (np.array([3,5]), np.array([8,2]), np.array([-1,1.1]), np.inf),
        (np.array([3,5]), np.array([8,2]), np.array([3,3]), 4.1159660434202126),
        (np.array([3,5])*3, np.array([8,2])*3, np.array([3,3])*3, 12.347898130260637),
    ]
    for a, b, c, between in test_sets:
        assert between_a(a,b,c) == between, f"Between_A seems wrong! Should be {between} but is {between_a(a,b,c)}"


#TODO test between_b_for_numbers!!