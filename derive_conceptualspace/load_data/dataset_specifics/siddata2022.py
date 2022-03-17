import pandas as pd
import nltk
from Levenshtein import distance
from tqdm import tqdm

from derive_conceptualspace.load_data.dataset_specifics import BaseDataset
from derive_conceptualspace.settings import get_setting
from fb_classifier.preprocess_data import make_classifier_dict

flatten = lambda l: [item for sublist in l for item in sublist]


class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.csv",
        candidate_min_term_count = 25, #TODO see the notes in config/derrac2015.yml!
    )
    additionals = ["ddc_code", "type", "veranstaltungsnummer", "is_uos", "is_bremen", "is_hannover", "is_other", "format", "source"]
    #subject is added to subtitle (#TODO something explicit)!

    FB_MAPPER = {1: "Sozial,Kultur,Kunst", 3: "Theologie,Lehramt,Musik", 4: "Physik", 5: "Bio,Chemie", 6: "Mathe,Info",
                 7: "Sprache,Literatur", 8: "Humanwiss", 9: "Wiwi", 10: "Rechtswiss"}

    @staticmethod
    def get_custom_class(name, descriptions):
        if name == "fachbereich":
            osna_descriptions = [i for num, i in enumerate(descriptions._descriptions) if i._additionals["publisher"] and "de.uni-osnabrueck.studip" in eval(i._additionals["publisher"])]
            forbidden_coursenums = {onum: [num for num, j in enumerate(eval(i._additionals["publisher"])) if j != "de.uni-osnabrueck.studip"] for onum, i in enumerate(osna_descriptions) if [num for num, j in enumerate(eval(i._additionals["publisher"])) if j != "de.uni-osnabrueck.studip"]}
            veranst_nums = [eval(i._additionals["veranstaltungsnummer"]) if i._additionals["veranstaltungsnummer"] else None for i in descriptions._descriptions]
            clas_di = make_classifier_dict(dict(enumerate(veranst_nums)))
            usables = {k: [int(v) for v in vs if v != "other" and int(v) <= 10] for k, vs in clas_di.items() if vs != "other"}
            usables = {k: [Dataset.FB_MAPPER.get(v) for v in vs] for k, vs in usables.items()}
            usables = {k: v for k, v in usables.items() if v and any(i is not None for i in v)}

    @staticmethod
    def preprocess_raw_file(df, pp_components, min_ges_nwords=20):
        """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
            dropping duplicates"""
        #TODO in exploration I also played around with Levenhsthein-distance etc!
        #remove those for which the Name (exluding stuff in parantheses) is equal...
        assert isinstance(df, pd.DataFrame)
        df = df.reset_index().drop(columns=["Unnamed: 0", "index"])
        # df = df[~df['description'].isnull()]
        df = df[df["description"]!="[]"]
        if get_setting("DEBUG"):
            df = df[:get_setting("DEBUG_N_ITEMS")*2]
        df = Dataset.merge_multidescs(df, pp_components)
        df.loc[:, 'ges_nwords'] = df["description"].str.count(" ").fillna(0)
        df["subtitle"] = df["subtitle"] + df["subject"] #TODO: maybe have an extra pp_comp for this?
        if pp_components.add_title:
            df.loc[:, 'ges_nwords'] += df["title"].str.count(" ").fillna(0)
        if pp_components.add_subtitle:
            df.loc[:, 'ges_nwords'] += df["subtitle"].str.count(" ").fillna(0)
        df = df[df["ges_nwords"] >= min_ges_nwords]
        with pd.option_context('mode.chained_assignment', None): #TODO publisher to get uni
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


def merge_col(col, pgbar=None):
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


if __name__ == "__main__":
    from os.path import join
    import os
    from derive_conceptualspace.pipeline import CustomContext, SnakeContext, load_envfiles
    from derive_conceptualspace.settings import DEFAULT_BASE_DIR
    from derive_conceptualspace.create_spaces.preprocess_descriptions import PPComponents

    load_envfiles("placetypes")
    ctx = SnakeContext.loader_context(silent=True)
    obj = pd.read_csv(join(ctx.get_config("BASE_DIR"), "siddata2022", "raw_descriptions.csv"))
    df = Dataset.preprocess_raw_file(obj, pp_components=PPComponents.from_str("mfacsd2"), min_ges_nwords=10)
    print()