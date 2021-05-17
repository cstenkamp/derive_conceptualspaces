"""The purpose of this file is to create a dataset from the Siddata-data that looks like the three datasets used in [DESC15],
available at http://www.cs.cf.ac.uk/semanticspaces/. Meaning: MDS, ..."""
from os.path import join, isdir, isfile, abspath, dirname, splitext, basename, split
import re
from collections import Counter
import random
from functools import partial
import math
import logging

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
import nltk
import pandas as pd
from HanTa import HanoverTagger as ht
from sklearn.manifold import MDS

from src.static.settings import SID_DATA_BASE
from static.settings import MDS_DIMENSIONS, DEBUG, RANDOM_SEED
from src.main.util.np_tools import np_divide, np_log
from src.main.util.logging import setup_logging

logger = logging.getLogger(basename(__file__))

def main():
    setup_logging("INFO")
    if RANDOM_SEED:
        logger.warning("Using a random seed!")
    random.seed(RANDOM_SEED)
    df = get_data()
    kwargs = {"max_elems": 100} if DEBUG else {}
    names, descriptions, mds = preprocess_data(df, **kwargs)
    mins = np.argmin(np.ma.masked_equal(mds.dissimilarity_matrix_, 0.0, copy=False), axis=0)
    for cmp1, cmp2 in enumerate(mins):
        print(f"{names[cmp1]} is most similar to {names[cmp2]}")
        if cmp1 > 20: break
    print()


def get_data(min_desc_len=10):
    #TODO in exploration I also played around with Levenhsthein-distance etc!
    df = pd.read_csv(join(SID_DATA_BASE, "kurse-beschreibungen.csv"))
    #remove those for which the Name (exluding stuff in parantheses) is equal...
    df['NameNoParanth'] = df['Name'].str.replace(re.compile(r'\([^)]*\)'), '')
    df = df.drop_duplicates(subset='NameNoParanth')
    #remove those with too short a description...
    df = df[~df['Beschreibung'].isna()]
    df.loc[:, 'desc_len'] = [len(i) for i in df['Beschreibung']]
    df = df[df["desc_len"] > min_desc_len]
    df = df.drop(columns=['desc_len','NameNoParanth'])
    #remove those with equal Veranstaltungsnummer...
    df = df.drop_duplicates(subset='VeranstaltungsNummer')
    return df


def preprocess_data(df, use_nltk_tokenizer=False, n_dims=MDS_DIMENSIONS, max_elems=None):
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
    return names, descriptions, X_transformed

def pmi(df, positive=False):
    """calculation of ppmi/pmi ([DESC15] 3.4 first lines)
    see https://stackoverflow.com/a/58725695/5122790
    see https://www.overleaf.com/project/609bbdd6a07c203c38a07ab4
    """
    logger.info("Calculating PMIs...")
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
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


def tokenize_sentences_countvectorizer(descriptions):
    #TODO CountVectorizer can be customized a lot, see https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    vectorizer = CountVectorizer()
    counted = vectorizer.fit_transform(descriptions)
    vocab = vectorizer.get_feature_names()
    return vocab, counted.toarray()

def tokenize_sentences_nltk(descriptions):
    #https://textmining.wp.hs-hannover.de/Preprocessing.html#Satzerkennung-und-Tokenization
    sent_list = [nltk.sent_tokenize(x, language="german") for x in descriptions]
    #so as we're really only concerning bags-of-words here, we run a lemmatizer
    # (see https://textmining.wp.hs-hannover.de/Preprocessing.html#Lemmatisierung)
    tagger = ht.HanoverTagger('morphmodel_ger.pgz')
    res = []
    words = set()
    for n, sample in tqdm(enumerate(sent_list)):
        #TODO: recognize language ^^
        all_tags = []
        for sent in sample:
            tokenized_sent = nltk.tokenize.word_tokenize(sent, language='german')
            tags = tagger.tag_sent(tokenized_sent)
            all_tags.extend([i[1].casefold() for i in tags if i[1] != "--"]) #TODO: not sure if I should remove the non-word-tokens completely..?
        res.append(all_tags) # we could res.append(Counter(all_tags))
        words.update(all_tags)
    words = list(words)
    alls = []
    for wordlist in res:
        cnt = Counter(wordlist)
        alls.append(np.array([cnt[i] for i in words]))
    return words, np.array(alls)




if __name__ == '__main__':
    main()



