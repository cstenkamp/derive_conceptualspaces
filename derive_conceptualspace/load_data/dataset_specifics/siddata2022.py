import warnings

import pandas as pd
import nltk
from Levenshtein import distance
from tqdm import tqdm

from derive_conceptualspace.load_data.dataset_specifics import BaseDataset
from derive_conceptualspace.settings import get_setting
from fb_classifier.preprocess_data import make_classifier_dict

flatten = lambda l: [item for sublist in l for item in sublist]
unique = lambda iterable: list({i:None for i in iterable}.keys())


class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.csv",
        candidate_min_term_count = 25, #TODO see the notes in config/derrac2015.yml!
    )
    additionals = ["ddc_code", "type", "veranstaltungsnummer", "is_uos", "is_bremen", "is_hannover", "is_other", "format", "source"]
    #subject is added to subtitle (#TODO something explicit)!

    #mapper from https://www.uni-osnabrueck.de/fileadmin/documents/public/1_universitaet/1.2_zahlen_daten_fakten/Zahlenspiegel_2009-2010.pdf
    # FB_MAPPER = {1: "Sozial,(Kultur,Kunst)", 2:"Kultur,Geo", 3: "Erziehung,Kultur(Theo,Lehramt,Musik)", 4: "Physik", 5: "Bio,Chemie", 6: "Mathe,Info",
    #              7: "Sprache,Literatur", 8: "Humanwiss", 9: "Wiwi", 10: "Rechtswiss"}
    FB_MAPPER = {1: "Sozialwissenschaften", 2:"Kultur-/Geowissenschaften", 3: "Erziehungs-/Kulturwissenschaften", 4: "Physik", 5: "Biologie/Chemie", 6: "Mathematik/Informatik",
                 7: "Sprach-/Literaturwissenschaften", 8: "Humanwissenschaften", 9: "Wirtschaftswissenschaften", 10: "Rechtswissenschaften"}

    #mapper from https://en.wikipedia.org/wiki/Dewey_Decimal_Classification#Classes
    DDC_MAPPER = {0: "Computer science, information, general", 1:"Philosophy and psychology", 2: "Religion", 3: "Social sciences", 4: "Language", 5: "Pure Science",
                  6: "Technology", 7: "Arts and recreation", 8: "Literature", 9: "History and geography"}

    @staticmethod
    def get_custom_class(name, descriptions, verbose=True):
        name = name.lower()
        if name == "fachbereich":
            veranst_nums = [eval(i._additionals.get("veranstaltungsnummer")) or None for i in descriptions._descriptions]
            new_dset = make_classifier_dict(dict(enumerate(veranst_nums)))
            usables = {k: [int(v) for v in vs if v != "other" and int(v) <= 10] for k, vs in new_dset.items() if vs != "other"}
            usables = {k: v for k, v in usables.items() if v and any(i is not None for i in v)}
            if verbose:
                print(f"Dropping {len(new_dset)-len(usables)}/{len(new_dset)} ({(len(new_dset)-len(usables))/len(new_dset):.2%}) courses - there is no Fachbereich for them")
                print(f"{sum([1 for i in usables.values() if len(i) > 1])} courses are assigned more than 1 Fachbereich!")
                warnings.warn("Will return the first Fachbereich for those ambiguous courses!")
            return lambda x: usables.get(x)[0], list(usables.keys()), Dataset.FB_MAPPER
        elif name.startswith("ddc_"):
            level = int(name.split("_")[1].replace("level","").replace("l",""))
            ddcs = [eval(i._additionals.get("ddc_code")) if i._additionals.get("ddc_code") else None for i in descriptions._descriptions]
            if verbose:
                print(f"Dropping {len(ddcs)-len([i for i in ddcs if i])}/{len(ddcs)} ({(len(ddcs)-len([i for i in ddcs if i]))/len(ddcs):.2%}) courses - there is no DDC for them")
            assert all(all(j.isnumeric() for j in i) for i in ddcs if i), "The Format of the DDCs is unexpected!"
            ddcs = [unique([j[:level] for j in i]) if i else None for i in ddcs]
            if verbose:
                print(f"{len([i for i in ddcs if i and len(i) > 1])} courses have multiple differing DDCs at this level!")
                warnings.warn("Will return the first DDC for those ambiguous courses!")
            ddcs = {k: v[0] for k, v in enumerate(ddcs) if v is not None}
            return lambda x: ddcs.get(x), list(ddcs.keys()), (Dataset.DDC_MAPPER if level == 1 else None)


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