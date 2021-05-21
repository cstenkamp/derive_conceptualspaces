import numpy as np
import logging

from src.main.load_semanticspaces import load_mds_representation, get_names
from src.static.settings import DATA_BASE, DATA_SET, MDS_DIMENSIONS
from src.main.measures import simple_similiarity, ortho_rejection, ortho_projection, between_a
from src.main.util.pretty_print import pretty_print as print, fmt

###################################### own realistic-looking tests ######################################

def test_simplesimiliarity_for_movies():
    """tests if a star wars movie is more similar to another star wars movie than a random one."""
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

###################################### number-tests ######################################

def test_ortho_rejection_projection_numbers():
    test_sets = [
        (np.array([3,5]), np.array([8,2]), np.array([3,3]), np.array([0.88235294, 1.47058824]), np.array([-4.11764706, 2.47058824])),
        (np.array([-1,5]), np.array([1,2]), np.array([-1,3]), np.array([0.92307692, 0.61538462]), np.array([-1.07692308, 1.61538462])),
        (np.array([3,5]), np.array([8,2]), np.array([-1,3]), np.array([1.94117647, 3.23529412]), np.array([-7.05882353, 4.23529412])),
    ]
    for a, b, c, rej, proj in test_sets:
        assert np.allclose(ortho_rejection(a-b,c-b), rej, atol=0.0001), f"Orthogonal Rejection seems wrong! Should be {rej} but is {ortho_rejection(a-b,c-b)}"
        assert np.allclose(ortho_projection(a-b,c-b), proj, atol=0.0001), f"Orthogonal Rejection seems wrong! Should be {proj} but is {ortho_projection(a-b,c-b)}"


def test_between_a_for_numbers():
    test_sets = [
        (np.array([1,1]), np.array([3,3]), np.array([2.5,2.5]), 1.5700924586837752e-16),
        (np.array([1,1]), np.array([3,3]), np.array([2.5,2.6]), 0.07071067811865482),
        (np.array([1,1]), np.array([3,3]), np.array([2,3]), 0.7071067811865476),
        (np.array([1,1]), np.array([3,3]), np.array([4,4]), np.inf),
        (np.array([1,1]), np.array([3,3]), np.array([-1,1]), np.inf),
        (np.array([0.01,0.01]), np.array([3,3]), np.array([-1,1.1]), 1.4849242404917498),
        (np.array([0.01,0.01]), np.array([3,3]), np.array([-1,1]), np.inf),
        (np.array([3,5]), np.array([8,2]), np.array([-1,1.1]), np.inf),
        (np.array([3,5]), np.array([8,2]), np.array([3,3]), 1.7149858514250889),
        (np.array([3,5])*3, np.array([8,2])*3, np.array([3,3])*3, 5.144957554275266),
    ]
    for a, b, c, between in test_sets:
        assert between_a(a,b,c) == between, f"between_a seems wrong! Should be {between} but is {between_a(a,b,c)}"

#TODO test between_b_for_numbers!!

###################################### paper claim tests ######################################

#TODO also do test for movies, that's table 2 in the paper
def test_paper_table1_claims():
    tmp = load_mds_representation(DATA_BASE, "places", 50)
    tmp2 = get_names(DATA_BASE, "places")
    name_mds = dict(zip(tmp2, tmp))
    paper_candidates = [
        ("fast food restaurant", "french restaurant", "american restaurant"),
        ("restaurant space", "tea room", "bistro"),
        ("marketplace", "slaughterhouse", "butcher shop"),
        ("coffee shop", "restaurant", "cafe"),
        ("bakery", "fast food restaurant", "deli"),
        ("gourmet shop", "liquor store", "wine shop"),
        ("home store", "vintage store", "furniture store"),
        ("convenience store", "farmers market", "grocery store"),
        ("history museum", "planetarium", "science museum"),
        ("japanese restaurant", "tapas restaurant", "sushi restaurant"),
        ("castle", "chapel", "abbey"),
        ("heath", "wetland", "bog"),
        ("mall", "newsstand", "bookstore"),
        ("greenhouse", "playhouse", "conservatory"),
        ("detached house", "triplex", "duplex"),
        ("flowerbed", "park", "garden"),
        ("flower shop", "toy store", "gift shop"),
        ("castle", "mansion house", "manor"),
        ("bamboo forest", "cropland", "rice paddy"),
        ("garden center", "gift shop", "flower shop")
    ]
    betweeness_positions = [find_betweenness_position(name_mds, first, second, third, between_a) for first, second, third in paper_candidates]
    print(f"{sum(1 for i in betweeness_positions if i == 1)} out of {len(betweeness_positions)} are actually #1")
    print(f"{sum(1 for i in betweeness_positions if i <= 5)} out of {len(betweeness_positions)} are in the top 5.")
    print(f"{sum(betweeness_positions)/len(betweeness_positions)} out of {len(name_mds)} is average betweenness position.")
    assert sum(1 for i in betweeness_positions if i == 1) > 10
    assert sum(1 for i in betweeness_positions if i <= 5) > 15
    assert sum(betweeness_positions)/len(betweeness_positions) < 25


def find_betweenness_position(name_mds, first, second, third, betweenness_measure=between_a, descriptions=None):
    logging.info(fmt(f"Question: Is *b*{third}*b* between *b*{first}*b* and *b*{second}*b*?"))
    a, b = name_mds[first], name_mds[second]
    scores = {}
    for name, candidate in name_mds.items():
        if any(candidate != a) and any(candidate != b):
            scores[name] = betweenness_measure(a, b, candidate)
    valid_scores = [val for val in scores.values() if val < np.inf]
    logging.info(fmt("Mean Betweenness:", sum(valid_scores) / len(valid_scores), "| Candidate betweeness:", scores[third]))
    sorted_items = [i[0] for i in sorted(scores.items(), key=lambda x: x[1])]
    logging.info(fmt("Top Between Candidates: *d*", "*d*, *d*".join(sorted_items[:5])+"*d*"))
    if descriptions:
        for item in [first, second, third] + sorted_items[:2]:
            logging.debug(fmt(f"Description of *b*{item}*b*: *d*{descriptions[item]}*d*"))
    logging.info(fmt(f"Their Candidate is betweenness place:", f"{sorted_items.index(third) + 1}/{len(scores)}"))
    return sorted_items.index(third) + 1

if __name__ == '__main__':
    test_between_a_for_numbers()
    test_ortho_rejection_projection_numbers()
    test_simplesimiliarity_for_movies()