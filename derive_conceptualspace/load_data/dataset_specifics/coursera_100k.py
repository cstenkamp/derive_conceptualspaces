from derive_conceptualspace.load_data.dataset_specifics import BaseDataset

class Dataset(BaseDataset):
    configs = dict(
        raw_descriptions_file = "raw_descriptions.csv",
        all_descriptions_lang = "en",
    )

    @staticmethod
    def preprocess_raw_file(df, *args, **kwargs):
        df = df.rename(columns={"CourseId": "title", "Review": "description"})
        df["subtitle"] = ""
        return df

    @staticmethod
    def init(ctx):
        ctx.set_config("language", "en", "force[dsetclass]")
        ctx.set_config("translate_policy", "onlyorig", "force[dsetclass]")
