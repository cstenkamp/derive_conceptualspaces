from os.path import join, isdir, isfile, abspath, dirname, splitext
import os
from dotenv import load_dotenv

#TODO use get_setting EVERYWHERE and have only defaults here!!!

################ new stuff #################

ALL_PP_COMPONENTS = ["tcsldp", "tcsdp"]
ALL_TRANSLATE_POLICY = ["translate", "origlan"] # "onlyeng"
ALL_EXTRACTION_METHOD = ["pp_keybert", "keybert"]
ALL_QUANTIFICATION_MEASURE = ["ppmi", "tf-idf"]
ALL_MDS_DIMENSIONS = [3, 100]
ALL_DCM_QUANT_MEASURE = ["tf-idf", "count", "binary"] #TODO check if these and the quantification_measure are interchangeable!! (also: tag-share is missing)

#set default-values for the ALL_... variables
for k, v in {k[4:]: v[0] for k,v in dict(locals()).items() if isinstance(v, list) and k.startswith("ALL_")}.items():
    locals()["DEFAULT_"+k] = v

ENV_PREFIX = "MA"

DEFAULT_DEBUG = False
DEFAULT_DEBUG_N_ITEMS = 50
DEFAULT_CANDIDATETERM_MIN_OCCURSIN_DOCS = 25
################ /new stuff #################



## Paths
ENV_FILE_PATH = os.getenv("ENV_FILE_PATH") or abspath(join(dirname(__file__), "", "..", "docker", ".env"))
#you can specify a custom path to an env-file using ENV_FILE_PATH = xyz python ...
load_dotenv(ENV_FILE_PATH)
DATA_BASE = abspath(join(dirname(__file__), "", "..", "..", "data"))
SPACES_DATA_BASE = join(DATA_BASE, "semanticspaces")
SID_DATA_BASE = join(DATA_BASE, "siddata_semspaces")
DATA_DUMP_DIR = abspath(join(dirname(__file__), "", "..", "data_dump"))
GOOGLE_CREDENTIALS_FILE = join(DATA_BASE, "gcloud_tools_key.json")

## model-updown
SIDDATA_SEAFILE_SERVER = 'https://myshare.uni-osnabrueck.de'
SIDDATA_SEAFILE_REPOID = '0b3948a7-9483-4e26-a7bb-a123496ddfcf' #for modelupdown v2
SIDDATA_SEAFILE_REPOWRITE_ACC = os.getenv("SIDDATA_SEAFILE_REPOWRITE_ACC")
SIDDATA_SEAFILE_REPOWRITE_PASSWORD = os.getenv("SIDDATA_SEAFILE_REPOWRITE_PASSWORD")
SIDDATA_SEAFILE_REPOREAD_ACC = os.getenv("SIDDATA_SEAFILE_REPOREAD_ACC")
SIDDATA_SEAFILE_REPOREAD_PASSWORD = os.getenv("SIDDATA_SEAFILE_REPOREADr_PASSWORD")
SIDDATA_SEAFILE_REPO_BASEPATH = "backend_synced_models"
SIDDATA_SEAFILE_MODEL_VERSIONS = {"siddata_semspaces": 1} #"semanticspaces": 1,

## specifically for courses-dataset
COURSE_TYPES = ["colloquium", "seminar", "internship", "practice", "lecture"]
# DEFAULT_TRANSLATE_POLICY = TRANSL

## other

OVEWRITE_SETTINGS_PREFIX="MA2"
DATA_SET = "movies" # "movies", "places", "wines", "courses"
# DEFAULT_MDS_DIMENSIONS = 20 #20,50,100,200
DEFAULT_DEBUG = False
DEFAULT_VERBOSE = True
DEFAULT_RANDOM_SEED = 1
MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"
# DEFAULT_CANDIDATETERM_MIN_OCCURSIN_DOCS = CANDIDATETERM_MIN_OCCURSIN_DOCS = 25

STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
MDS_DEFAULT_BASENAME = "siddata_names_descriptions_mds_"

########################################################################################################################
# KEEP THIS AT THE BOTTOM!

def get_setting(name, default_none=False):
    if os.getenv(ENV_PREFIX+"_"+name):
        tmp = os.environ[ENV_PREFIX+"_"+name]
        if tmp.isnumeric():
            return int(tmp)
        elif all([i.isdecimal() or i in ".," for i in tmp]):
            return float(tmp)
        return tmp
    elif os.getenv(ENV_PREFIX+"_"+name+"_FALSE"):
        return False
    if "DEFAULT_"+name in globals():
        print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
        return globals()["DEFAULT_"+name]
    if default_none:
        return None
    assert False, f"Couldn't get setting {name}"

#TODO now I can overwrite the env-vars both in click and with this, this is stupid argh

#overwriting env-vars
from types import ModuleType  # noqa: E402
_all_settings = {
    k: v
    for k, v in locals().items()
    if (not k.startswith("_") and not callable(v) and not isinstance(v, ModuleType) and k.isupper())
}
_overwrites = {k: os.getenv(f"{OVEWRITE_SETTINGS_PREFIX}_" + k) for k, v in _all_settings.items() if os.getenv(f"{OVEWRITE_SETTINGS_PREFIX}_" + k)}
for k, v in _overwrites.items():
    if isinstance(_all_settings[k], (list, tuple)):
        locals()[k] = [i.strip("\"' ") for i in v.strip("[]()").split(",")]
    elif isinstance(_all_settings[k], dict):
        assert v.strip().startswith("{") and v.strip().endswith("}")
        locals()[k] = dict([[j.strip("\"' ") for j in i.strip().split(":")] for i in v.strip(" {}").split(",")])
    elif isinstance(_all_settings[k], bool):
        locals()[k] = bool(v)
    elif isinstance(_all_settings[k], int):
        locals()[k] = int(v)
    elif isinstance(_all_settings[k], float):
        locals()[k] = float(v)
    elif not isinstance(_all_settings[k], str):
        raise NotImplementedError(f"I don't understand the type of the setting {k} you want to overwrite with a envvar")
    else:
        print(f"Overwriting setting {k} with {v}")
        locals()[k] = v



#actually make all defined directories (global vars that end in "_PATH")
for key, val in dict(locals()).items():
    if key.endswith('_PATH') and not isfile(val):
        locals()[key] = abspath(val)
        os.makedirs(locals()[key], exist_ok=True)