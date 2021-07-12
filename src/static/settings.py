from os.path import join, isdir, isfile, abspath, dirname, splitext
import os
from dotenv import load_dotenv

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

## other
DATA_SET = "movies" # "movies", "places", "wines", "courses"
MDS_DIMENSIONS = 20 #20,50,100,200
DEBUG = True
RANDOM_SEED = None
MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"

STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html



########################################################################################################################
# KEEP THIS AT THE BOTTOM!

#overwriting with env-vars
all_vars = [key for key,val in locals().items() if not callable(val) and not key.startswith("_") and key.isupper()]
for key, val in {i: os.getenv(i) for i in all_vars if os.getenv(i)}.items():
    locals()[key] = val

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