from main.load_semanticspaces import load_mds_representation, get_names
from static.settings import DATA_BASE, DATA_SET, MDS_DIMENSIONS

from main.measures import simple_similiarity

def test_simplesimiliarity_for_movies():
    tmp = load_mds_representation(DATA_BASE, DATA_SET, MDS_DIMENSIONS)
    tmp2 = get_names(DATA_BASE, DATA_SET)
    name_mds = dict(zip(tmp2, tmp))
    starwars = [i for i in name_mds.keys() if "Star Wars" in i]
    starwarssims = [simple_similiarity(name_mds[starwars[0]], name_mds[starwars[i]]) for i in range(1, len(starwars))]
    # print("Star Wars to Star Wars avg Similarity:", sum(starwarssims) / len(starwarssims))
    othersims = [simple_similiarity(name_mds[starwars[0]], i) for i in name_mds.values()]
    # print("Star Wars to All Others avg Similarity:", sum(othersims) / len(othersims))
    assert sum(starwarssims) / len(starwarssims) > sum(othersims) / len(othersims)




