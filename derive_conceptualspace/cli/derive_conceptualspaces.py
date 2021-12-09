"""Section 4.2 in DESC15"""
from src.static.settings import SPACES_DATA_BASE, DATA_SET, MDS_DIMENSIONS
from src.main.load_data.load_semanticspaces import load_mds_representation, get_names, get_grouped_candidates

def main():
    mds, mds_path = load_mds_representation(SPACES_DATA_BASE, DATA_SET, MDS_DIMENSIONS)
    names, names_path = get_names(SPACES_DATA_BASE, DATA_SET)
    candidates, group_vectors = get_grouped_candidates(SPACES_DATA_BASE, DATA_SET, MDS_DIMENSIONS)
    print()

if __name__ == '__main__':
    main()