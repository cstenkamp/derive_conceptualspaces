from derive_conceptualspace.load_data.dataset_specifics import BaseDataset

class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "guttenbergstories.csv",
        all_descriptions_lang = "en",
    )
    additionals = ["Author"]

    @staticmethod
    def preprocess_raw_file(df, pp_components):
        df = df.rename(columns={"Title": "title", "content": "description"})
        df["subtitle"] = ""
        return df
