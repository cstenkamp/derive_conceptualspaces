from os.path import join, isdir, isfile, abspath, dirname, splitext
import os
from dotenv import load_dotenv

from src.main.util.mds_object import ORIGLAN, ONLYENG, TRANSL

## Paths

ENV_FILE_PATH = os.getenv("ENV_FILE_PATH") or abspath(join(dirname(__file__), "..", "..", "docker", ".env"))
#you can specify a custom path to an env-file using ENV_FILE_PATH = xyz python ...
load_dotenv(ENV_FILE_PATH)
DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data"))
SPACES_DATA_BASE = join(DATA_BASE, "semanticspaces")
SID_DATA_BASE = join(DATA_BASE, "siddata_semspaces")
DATA_DUMP_DIR = abspath(join(dirname(__file__), "..", "..", "data_dump"))
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
DEFAULT_TRANSLATE_POLICY = TRANSL

## other
DATA_SET = "movies" # "movies", "places", "wines", "courses"
MDS_DIMENSIONS = 20 #20,50,100,200
DEBUG = False
RANDOM_SEED = None
MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"
CANDIDATETERM_MIN_OCCURSIN_DOCS = 10

STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
MDS_DEFAULT_BASENAME = "siddata_names_descriptions_mds_"

########################################################################################################################
# KEEP THIS AT THE BOTTOM!

#overwriting env-vars
ENV_PREFIX = "MA"
from types import ModuleType  # noqa: E402
_all_settings = {
    k: v
    for k, v in locals().items()
    if (not k.startswith("_") and not callable(v) and not isinstance(v, ModuleType) and k.isupper())
}
_overwrites = {k: os.getenv(f"{ENV_PREFIX}_" + k) for k, v in _all_settings.items() if os.getenv(f"{ENV_PREFIX}_" + k)}
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
        locals()[k] = v


# checkers etc
import os
if not os.environ.get("SETTINGS_WARNINGS_GIVEN"):
    if DEBUG:
        print("DEBUG is activated!!!")
    if RANDOM_SEED:
        print("Using a random seed!!!")
os.environ["SETTINGS_WARNINGS_GIVEN"] = "1"

#actually make all defined directories (global vars that end in "_PATH")
for key, val in dict(locals()).items():
    if key.endswith('_PATH') and not isfile(val):
        locals()[key] = abspath(val)
        os.makedirs(locals()[key], exist_ok=True)