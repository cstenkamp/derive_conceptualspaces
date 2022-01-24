import pandas as pd
from derive_conceptualspace.load_data.dataset_specifics import BaseDataset
from pandas.core.common import SettingWithCopyWarning


class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.csv"
    )

    @staticmethod
    def preprocess_raw_file(df, min_ges_nwords=21):
        """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
            dropping duplicates"""
        #TODO in exploration I also played around with Levenhsthein-distance etc!
        #remove those for which the Name (exluding stuff in parantheses) is equal...
        assert isinstance(df, pd.DataFrame)
        df = df.reset_index().drop(columns=["Unnamed: 0", "index"])
        df = df[~df['description'].isna()]
        df.loc[:, 'ges_nwords'] = df["description"].str.count(" ").fillna(0)+df["title"].str.count(" ").fillna(0)+df["subtitle"].str.count(" ").fillna(0)
        df = df[df["ges_nwords"] >= min_ges_nwords]
        with pd.option_context('mode.chained_assignment', None):
            for column in ["title", "description", "subtitle"]:
                df.loc[:, column] = df[column].copy().str.strip()
        df.drop_duplicates(subset=["Name"])
        return df