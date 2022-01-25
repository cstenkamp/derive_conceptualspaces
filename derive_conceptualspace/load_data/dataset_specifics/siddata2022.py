import pandas as pd
from derive_conceptualspace.load_data.dataset_specifics import BaseDataset
import nltk
from Levenshtein import distance
from tqdm import tqdm

flatten = lambda l: [item for sublist in l for item in sublist]


class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.csv"
    )
    additionals = ["ddc_code", "type", "veranstaltungsnummer"]

    @staticmethod
    def preprocess_raw_file(df, pp_components, min_ges_nwords=20):
        """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
            dropping duplicates"""
        #TODO in exploration I also played around with Levenhsthein-distance etc!
        #remove those for which the Name (exluding stuff in parantheses) is equal...
        assert isinstance(df, pd.DataFrame)
        df = df.reset_index().drop(columns=["Unnamed: 0", "index", "n_descs"])
        # df = df[~df['description'].isnull()]
        df = df[df["description"]!="[]"]
        df = Dataset.merge_multidescs(df, pp_components)
        df.loc[:, 'ges_nwords'] = df["description"].str.count(" ").fillna(0)
        if pp_components.add_title:
            df.loc[:, 'ges_nwords'] += df["title"].str.count(" ").fillna(0)
        if pp_components.add_subtitle:
            df.loc[:, 'ges_nwords'] += df["subtitle"].str.count(" ").fillna(0)
        df = df[df["ges_nwords"] >= min_ges_nwords]
        with pd.option_context('mode.chained_assignment', None):
            for column in ["title", "description", "subtitle"]:
                df.loc[:, column] = df[column].copy().str.strip()
        return df

    @staticmethod
    def merge_multidescs(df, pp_components):
        if pp_components.sentwise_merge:
            df["subtitle"] = merge_col(df["subtitle"])
            df["description"] = merge_col(df["description"], pgbar="pre-preprocessing descriptions")
        else:
            assert False, "no way how to merge"
        return df


def merge_col(col, pgbar=False):
    new_col = []
    col = [eval(i) if isinstance(i, str) else "" for i in col]
    if pgbar: col = tqdm(col, desc=pgbar)
    for descs in col:
        if len(descs) == 1:
            new_col.append(descs[0].strip())
        else:
            descs = [i.strip() for i in descs]
            descs = check_subsumed(check_doubledot(descs))
            descs = check_levenshtein(descs)
            if len(descs) == 1:
                new_col.append(descs[0].strip())
            else:
                descs = [nltk.sent_tokenize(part) for part in descs]
                descs = " ".join(list({k: None for k in flatten(descs)}.keys()))
                new_col.append(descs)
    return new_col


def check_subsumed(course):
    rm_inds = []
    descs = [nltk.sent_tokenize(part) for part in course]
    for ndesc, desc in enumerate(descs):
        for ndesc2, desc2 in enumerate(descs):
            if ndesc == ndesc2:
                continue
            if all(sent in desc for sent in desc2):
                if ndesc not in rm_inds:
                    rm_inds.append(ndesc2)
    return [i for n, i in enumerate(course) if n not in rm_inds]


def check_levenshtein(course, rel_tolerance=0.003):
    rm_inds = []
    for ndesc, desc in enumerate(course):
        for ndesc2, desc2 in enumerate(course):
            if ndesc == ndesc2:
                continue
            if distance(desc, desc2) / len(desc) < rel_tolerance:
                if ndesc not in rm_inds:
                    rm_inds.append(ndesc2)
    return [i for n, i in enumerate(course) if n not in rm_inds]


def check_doubledot(course):  # there are SO MANY courses where the only difference in the descriptions are stupid ".." instead of "." at sentence end
    return [desc.strip().replace("...", "<TRIPLEDOT>").replace("..", ".").replace("<TRIPLEDOT>", "...") for desc in course]