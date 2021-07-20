import pandas as pd


def load_data(paths: dict):
    data = {key: pd.read_csv(paths[key], index_col=0) for key, val in paths.items()}
    for key, val in data.items():
        if isinstance(val.index.name, str) and len(val.index.name) > 0:
            data[key] = pd.read_csv(paths[key])
    data = {key: remove_unnecesssary_columns(val) for key, val in data.items()}
    return data


def remove_unnecesssary_columns(data):
    """
    cleaning and renaming of columns
    :param data: the raw data as saved in the tsv
    :return: only the classes- and description-columns with meaningful column-names
    """
    data = data.set_index("Name")
    return data["VeranstaltungsNummer"]