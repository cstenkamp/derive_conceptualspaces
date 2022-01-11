import re
import pandas as pd
from derive_conceptualspace.load_data.dataset_specifics import BaseDataset

class Dataset(BaseDataset):

    @staticmethod
    def preprocess_raw_file(df, min_desc_len=10):
        """loads the given Siddata-Style CSV into a pandas-dataframe, already performing some processing like
            dropping duplicates"""
        #TODO in exploration I also played around with Levenhsthein-distance etc!
        #remove those for which the Name (exluding stuff in parantheses) is equal...
        assert isinstance(df, pd.DataFrame)
        df['NameNoParanth'] = df['Name'].str.replace(re.compile(r'\([^)]*\)'), '')
        df = df.drop_duplicates(subset='NameNoParanth')
        #remove those with too short a description...
        df = df[~df['Beschreibung'].isna()]
        df.loc[:, 'desc_len'] = [len(i) for i in df['Beschreibung']]
        df = df[df["desc_len"] > min_desc_len]
        df = df.drop(columns=['desc_len','NameNoParanth'])
        #remove those with equal Veranstaltungsnummer...
        df = df.drop_duplicates(subset='VeranstaltungsNummer')
        #TODO instead of dropping, I could append descriptions or smth!!
        for column in ["Name", "Beschreibung", "Untertitel"]:
            df[column] = df[column].str.strip()
        if len((dups := df[df["Name"].duplicated(keep=False)])) > 0:
            print("There are courses with different VeranstaltungsNummer but equal Name!")
            for n, cont in dups.iterrows():
                df.loc[n]["Name"] = f"{cont['Name']} ({cont['VeranstaltungsNummer']})"
        return df