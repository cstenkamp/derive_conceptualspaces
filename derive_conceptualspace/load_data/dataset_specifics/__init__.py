from abc import ABC, abstractmethod
import importlib.util
from os.path import join, dirname

def load_dataset_class(dataset_name):
    if dataset_name == "places": dataset_name = "placetypes" #TODO: this mixes code & data
    _spec = importlib.util.spec_from_file_location("dataset_specifics", join(dirname(__file__), dataset_name+".py"))
    specifics_module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(specifics_module)
    return specifics_module.Dataset

class BaseDataset(ABC):

    @staticmethod
    @abstractmethod
    def preprocess_raw_file(df, *args, **kwargs):
        ...