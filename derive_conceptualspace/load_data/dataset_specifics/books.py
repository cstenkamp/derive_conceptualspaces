from derive_conceptualspace.load_data.dataset_specifics import BaseDataset

class Dataset(BaseDataset):
    @staticmethod
    def preprocess_raw_file(df):
        df = df.rename(columns={"Title": "Name", "Description": "Beschreibung"})
        df["Untertitel"] = ""
        return df

#TODO DDC mitschleifen, HTML-Tags aus description entfernen