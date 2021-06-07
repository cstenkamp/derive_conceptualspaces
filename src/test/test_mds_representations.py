from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split

import numpy as np

from main.load_data.load_semanticspaces import load_mds_representation
from src.static.settings import DATA_BASE, MDS_DIMENSIONS, SID_DATA_BASE
from scripts.create_siddata_dataset import load_mds, display_mds

n_dims = 20

def test_representations_equal():
    names, descriptions, mds = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_{n_dims}.json"))
    mds2 = load_mds_representation(DATA_BASE, "courses", n_dims)
    assert mds.embedding_.shape == mds2.shape
    assert np.array_equal(mds.embedding_, mds2)