from derive_conceptualspace.load_data.dataset_specifics import BaseDataset

class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "usable_reviews.csv"
    )

    @staticmethod
    def preprocess_raw_file(df):
        df = df.rename(columns={"CourseId": "Name", "Review": "Beschreibung"})
        df["Untertitel"] = ""
        return df
