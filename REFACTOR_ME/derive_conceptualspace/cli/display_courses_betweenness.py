from os.path import join

from src.main.load_data.load_semanticspaces import load_mds_representation, get_names
from src.static.settings import SPACES_DATA_BASE, MDS_DIMENSIONS
from src.main.measures import between_a
from src.main.util.logutils import setup_logging
from src.test.test_semanticspaces_measures import find_betweenness_position

SOME_IDS = {"Computer Vision": 4155, "Computergrafik": 547, "Computergrafikpraktikum": 453, "Machine Learning": 1685, "Rechnernetzepraktikum": 1921}


def main():
    setup_logging("DEBUG")
    show_betwennesses()


def get_descriptions():
    from src.static.settings import SID_DATA_BASE
    from derive_conceptualspace.cli.run_pipeline import load_mds
    names, descriptions, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{MDS_DIMENSIONS}.json"))
    return dict(zip(names, descriptions))


def show_betwennesses():
    mds = load_mds_representation(SPACES_DATA_BASE, "courses", MDS_DIMENSIONS)[0]
    names = get_names(SPACES_DATA_BASE, "courses")[0]
    name_mds = dict(zip(names, mds))
    candidates = [("Computergrafik", "Computer Vision", "Machine Learning"), ("Rechnernetzepraktikum", "Computergrafik", "Computergrafikpraktikum")]
    descriptions = get_descriptions()
    candidates = [tuple(names[SOME_IDS[j]] for j in i) for i in candidates]
    for first, second, third in candidates:
        find_betweenness_position(name_mds, first, second, third, between_a, descriptions=descriptions)



if __name__ == '__main__':
    main()
