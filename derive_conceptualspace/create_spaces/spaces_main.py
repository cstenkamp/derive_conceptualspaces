from os.path import join, isfile, dirname, basename
import re
import logging
import os
import json
import random

import numpy as np
import pandas as pd
import html

from sklearn.manifold._mds import MDS

from misc_util.pretty_print import pretty_print as print
from .create_mds import ppmi, create_dissimilarity_matrix

from .translate_descriptions import create_load_languages_file
from derive_conceptualspace.util.mds_object import Description
from derive_conceptualspace.util.jsonloadstore import json_dump, json_load
from derive_conceptualspace.settings import TRANSL, ORIGLAN, ONLYENG, DEBUG_N_ITEMS
from derive_conceptualspace.settings import SID_DATA_BASE, SPACES_DATA_BASE, get_setting
from derive_conceptualspace.util.text_tools import run_preprocessing_funcs, make_bow, tf_idf
from derive_conceptualspace.util.dtm_object import DocTermMatrix

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]


########################################################################################################################
########################################################################################################################
########################################################################################################################
# pipeline to the MDS representation from course descriptions

def preprocess_descriptions_full(from_path, translate_policy, pp_components, from_name="kurse-beschreibungen.csv"):
    #TODO options to consider language, fachbereich, and to add [translated] title to description
    df = load_preprocess_raw_course_file(join(from_path, from_name))
    if get_setting("DEBUG"):
        df = pd.DataFrame([df.iloc[key] for key in random.sample(range(len(df)), k=DEBUG_N_ITEMS)])
    descriptions = handle_translations(from_path, list(df["Name"]), list(df["Beschreibung"]), translate_policy)
    vocab, descriptions = preprocess_descriptions(descriptions, pp_components)
    return vocab, descriptions


def load_preprocessed_descriptions(filepath):
    vocab, descriptions, pp_components = (tmp := json_load(filepath))["vocab"], tmp["descriptions"], tmp["pp_components"]
    descriptions = [Description.fromstruct(i[1]) for i in descriptions]
    if get_setting("DEBUG"):
        assert DEBUG_N_ITEMS <= len(descriptions), f"The Descriptions-Dataset contains {len(descriptions)} samples, but you want to draw {DEBUG_N_ITEMS}!"
        descriptions = [descriptions[key] for key in random.sample(range(len(descriptions)), k=DEBUG_N_ITEMS)]
        vocab = sorted(set(flatten([set(i.bow.keys()) for i in descriptions])))
    return vocab, descriptions, pp_components


def create_dissim_mat(base_dir, pp_descriptions_filename, quantification_measure):
    vocab, descriptions, pp_components = load_preprocessed_descriptions(join(base_dir, pp_descriptions_filename))
    assert quantification_measure in ["tf-idf", "ppmi"]
    # TODO: # vocab, counts = tokenize_sentences_nltk(descriptions) if use_nltk_tokenizer else tokenize_sentences_countvectorizer(descriptions) allow countvectorizer!
    dtm = DocTermMatrix(all_terms=vocab, descriptions=descriptions)
    if quantification_measure == "tf-idf":
        quantification = tf_idf(dtm, verbose=bool(get_setting("VERBOSE")), descriptions=descriptions)
    elif quantification_measure == "ppmi":
        quantification = ppmi(dtm, verbose=bool(get_setting("VERBOSE")), descriptions=descriptions)  # das ist jetzt \textbf{v}_e with all e's as rows
    quantification = DocTermMatrix({"doc_term_matrix": quantification, "all_terms": dtm.all_terms})
    #cannot use ppmis directly, because a) too sparse, and b) we need a geometric representation with euclidiean props (betweeness, parallism,..)
    assert all(len(set((lst := [i[0] for i in dtm]))) == len(lst) for dtm in quantification.dtm)
    dissim_mat = create_dissimilarity_matrix(quantification.as_csr())
    return quantification, dissim_mat, pp_components


def create_mds_json(base_dir, dissim_mat_filename, n_dims):
    #TODO - isn't isomap better suited than MDS? https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling
    # !! [DESC15] say they compared it and it's worse ([15] of [DESC15])!!!
    loaded = json_load(join(base_dir, dissim_mat_filename))
    embedding = MDS(n_components=n_dims, random_state=get_setting("RANDOM_SEED", default_none=True), dissimilarity="precomputed")
    mds = embedding.fit(loaded["dissim_mat"])
    return {"mds": mds, "pp_components": loaded["pp_components"], "quant_measure": loaded["quant_measure"]}
    #TODO translate_policy wird nicht mitgespeichert und die referenz auf die Descriptions wäre noch nice


    # # names, descriptions, mds = preprocess_descriptions(df, n_dims=n_dims, **kwargs)
    # # return names, descriptions, mds


def load_preprocess_raw_course_file(fpath, min_desc_len=10):
    """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
        dropping duplicates"""
    #TODO in exploration I also played around with Levenhsthein-distance etc!
    df = pd.read_csv(fpath)
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
    df["Name"] = df["Name"].str.strip()
    return df


def handle_translations(from_path, names, descriptions, translate_policy, translations_filename="translated_descriptions.json", assert_all_translated=True):
    lang_dict = create_load_languages_file(from_path, names, descriptions)
    orig_lans = [lang_dict[i] for i in names]
    if translate_policy == ORIGLAN:
        result = [Description(text=descriptions[i], lang=orig_lans[i], for_name=names[i], orig_lang=orig_lans[i]) for i in range(len(descriptions))]
    elif translate_policy == ONLYENG:
        indices = [ind for ind, elem in enumerate(orig_lans) if elem == "en"]
        print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones")
        names, descriptions = [names[i] for i in indices], [descriptions[i] for i in indices]
        result = [Description(text=descriptions[i], lang="en", for_name=names[i], orig_lang="en") for i in range(len(descriptions))]
    elif translate_policy == TRANSL:
        result = []
        with open(join(from_path, translations_filename), "r") as rfile:
            translations = json.load(rfile)
        missing_names = set()
        for desc, name in zip(descriptions, names):
            if lang_dict[name] == "en":
                result.append(Description(text=desc, lang="en", for_name=name, orig_lang="en", orig_text=desc))
            elif name in translations:
                result.append(Description(text=translations[name], lang="en", for_name=name, orig_lang=lang_dict[name], orig_text=desc))
            elif name+" " in translations:
                result.append(Description(text=translations[name+" "], lang="en", for_name=name, orig_lang=lang_dict[name], orig_text=desc))
            elif " "+name in translations:
                result.append(Description(text=translations[" "+name], lang="en", for_name=name, orig_lang=lang_dict[name], orig_text=desc))
            else:
                missing_names.add(name)
        if len(result) < len(names):
            print(f"Dropped {len(names) - len(result)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
        assert not (len(result) < len(names) and assert_all_translated)
    else:
        assert False
    return result


def preprocess_descriptions(descriptions, components):
    """3.4 in [DESC15]"""
    # TODO there's https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    # TODO it must save which kind of preprocessing it did (removed stop words, convered lowercase, stemmed, ...)
    descriptions = run_preprocessing_funcs(descriptions, components) #TODO allow the CountVectorizer!
    print("Ran the following preprocessing funcs:", ", ".join([i[1] for i in descriptions[0].processing_steps]))
    vocab, descriptions = make_bow(descriptions)
    return vocab, descriptions


#
# def display_mds(mds, names, max_elems=30):
#     """
#     Args:
#          mds: np.array or data_prep.jsonloadstore.Struct created from sklearn.manifold.MDS or sklearn.manifold.MSD
#          name: list of names
#          max_elems (int): how many to display
#     """
#     if hasattr(mds, "embedding_"):
#         mds = mds.embedding_
#     mins = np.argmin(np.ma.masked_equal(mds, 0.0, copy=False), axis=0)
#     for cmp1, cmp2 in enumerate(mins):
#         print(f"*b*{names[cmp1]}*b* is most similar to *b*{names[cmp2]}*b*")
#         if max_elems and cmp1 >= max_elems-1:
#             break
#
#
# def create_descstyle_dataset(n_dims, dsetname, from_path=SID_DATA_BASE, from_name_base="siddata_names_descriptions_mds_{n_dims}.json", to_path=SPACES_DATA_BASE, translate_policy=ORIGLAN):
#     names, descriptions, mds, languages = load_translate_mds(from_path, from_name_base.format(n_dims=n_dims), translate_policy)
#     display_mds(mds, names)
#     fname = join(to_path, dsetname, f"d{n_dims}", f"{dsetname}{n_dims}.mds")
#     os.makedirs(dirname(fname), exist_ok=True)
#     embedding = list(mds.embedding_)
#     indices = np.argsort(np.array(names))
#     names, descriptions, embedding = [names[i] for i in indices], [descriptions[i] for i in indices], np.array([embedding[i] for i in indices])
#     if isfile(namesfile := join(dirname(fname), "../main", "courseNames.txt")):
#         with open(namesfile, "r") as rfile:
#             assert [i.strip() for i in rfile.readlines()] == [i.strip() for i in names]
#     else:
#         with open(namesfile, "w") as wfile:
#             wfile.writelines("\n".join(names))
#     if isfile(fname):
#         raise FileExistsError(f"{fname} already exists!")
#     np.savetxt(fname, embedding, delimiter="\t")
#
# ########################################################################################################################
# ########################################################################################################################
# ########################################################################################################################
#
# def load_translate_mds(file_path, file_name, translate_policy, assert_meta=(), translations_filename="translated_descriptions.json", assert_allexistent=True):
#     #TODO what now with this?! is this superflous? What about filgering the MDS?!
#     print("DEPRECATED!!!")
#     print(f"Working with file *b*{file_name}*b* in *b*{file_path}*b*!")
#     loaded = json_load(join(file_path, file_name), assert_meta=assert_meta)
#     names, descriptions, mds = loaded["names"], loaded["descriptions"], loaded["mds"]
#     if assert_allexistent:
#         assert len(names) == len(descriptions) == mds.embedding_.shape[0]
#     languages = create_load_languages_file(file_path, names, descriptions)
#     orig_n_samples = len(names)
#     additional_kwargs = {}
#     if translate_policy == ORIGLAN:
#         pass
#     elif translate_policy == ONLYENG:
#         indices = [ind for ind, elem in enumerate(languages) if elem == "en"]
#         print(f"Dropped {len(names)-len(indices)} out of {len(names)} descriptions because I will take only the english ones")
#         names, descriptions, languages = [names[i] for i in indices], [descriptions[i] for i in indices], [languages[i] for i in indices]
#         mds.embedding_ = np.array([mds.embedding_[i] for i in indices])
#         mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in indices])
#     elif translate_policy == TRANSL:
#         additional_kwargs["original_descriptions"] = descriptions
#         with open(join(file_path, translations_filename), "r") as rfile:
#             translations = json.load(rfile)
#         new_descriptions, new_indices = [], []
#         for ind, name in enumerate(names):
#             if languages[name] == "en":
#                 new_descriptions.append(descriptions[ind])
#                 new_indices.append(ind)
#             elif name in translations:
#                 new_descriptions.append(translations[name])
#                 new_indices.append(ind)
#         dropped_indices = set(range(len(new_indices))) - set(new_indices)
#         if dropped_indices:
#             print(f"Dropped {len(names) - len(new_indices)} out of {len(names)} descriptions because I will take english ones and ones with a translation")
#         descriptions = new_descriptions
#         names, languages = [names[i] for i in new_indices], [list(languages.values())[i] for i in new_indices]
#         mds.embedding_ = np.array([mds.embedding_[i] for i in new_indices])
#         mds.dissimilarity_matrix_ = np.array([mds.dissimilarity_matrix_[i] for i in new_indices])
#     descriptions = [html.unescape(i).replace("  ", " ") for i in descriptions]
#     if assert_allexistent:
#         assert len(names) == len(descriptions) == mds.embedding_.shape[0] == orig_n_samples
#     return MDSObject(names, descriptions, mds, languages, translate_policy, orig_n_samples, **additional_kwargs)

