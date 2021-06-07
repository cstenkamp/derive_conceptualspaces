from functools import partial
import math
import logging
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split

import numpy as np
from tqdm import tqdm
from sklearn.manifold import MDS

from static.settings import RANDOM_SEED
from src.main.util.np_tools import np_divide, np_log
from src.main.load_data.siddata_data_prep.tokenizers import tokenize_sentences_countvectorizer, tokenize_sentences_nltk
from src.main.load_data.siddata_data_prep.jsonloadstore import json_dump, json_dumps, Struct

logger = logging.getLogger(basename(__file__))

def preprocess_data(df, n_dims, use_nltk_tokenizer=False, max_elems=None):
    """3.4 in [DESC15]"""
    # TODO there's https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    name_sent_dict = df.set_index("Name")["Beschreibung"].to_dict()
    # if DEBUG:
    #     name_sent_dict = {key: name_sent_dict[key] for key in random.sample(name_sent_dict.keys(), k=20)}
    names, descriptions = list(name_sent_dict.keys()), list(name_sent_dict.values())
    vocab, counts = tokenize_sentences_nltk(descriptions) if use_nltk_tokenizer else tokenize_sentences_countvectorizer(descriptions)
    #TODO einschränken welche terms wir considern? ("let us use v_e to denote the resulting rep of e, i.e. if the considered terms are t1,...,tk) -> how do I know which ones?? -> TODO: parameter k
    ppmis = ppmi(counts) #das ist jetzt \textbf{v}_e with all e's as rows
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    if max_elems:
        ppmis = ppmis[:max_elems,:]; names = names[:max_elems]; descriptions = descriptions[:max_elems]
    dissim_mat = create_dissimilarity_matrix(ppmis)
    #TODO - isn't isomap better suited than MDS? https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling
    embedding = MDS(n_components=n_dims, random_state=RANDOM_SEED if RANDOM_SEED else None)
    X_transformed = embedding.fit(dissim_mat)
    mds = Struct(**X_transformed.__dict__) #let's return the dict of the MDS such that we can load it from json and its equal
    return names, descriptions,  mds

def pmi(df, positive=False):
    """calculation of ppmi/pmi ([DESC15] 3.4 first lines)
    see https://stackoverflow.com/a/58725695/5122790
    see https://www.overleaf.com/project/609bbdd6a07c203c38a07ab4
    """
    logger.info("Calculating PMIs...")
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals)
    expected = np_divide(expected, total)
    df = np_divide(df, expected)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np_log(df)
    if positive:
        df[df < 0] = 0.0
    return df

ppmi = partial(pmi, positive=True)


def create_dissimilarity_matrix(arr):
    """returns the dissimilarity matrix, needed as input for the MDS. Input is the dataframe
    that contains all ppmi's of all entities (entities are rows, columns are terms, cells are then the
    ppmi(e,t) for all entity-term-combinations. Output is the normalized angular difference between
    all entities ei,ej --> an len(e)*len(e) matrix. This is needed as input for the MDS.
    See [DESC15] section 3.4."""
    logger.info("Creating the dissimilarity matrix...")
    res = np.zeros((arr.shape[0],arr.shape[0]))
    for n1, e1 in enumerate(tqdm(arr)):
        for n2, e2 in enumerate(arr):
            if n1 != n2:
                assert e1.max() and e2.max()
                p1 = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                if 0 < p1-1 < 1e-12:
                    p1 = 1 #aufgrund von rundungsfehlern kann es >1 sein
                res[n1,n2] = 2 / math.pi * math.acos(p1)
    assert np.allclose(res, res.T, atol=1e-10) #TODO if this takes too long I can also not create the upperleft half
    return res
