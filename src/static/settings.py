from os.path import join, isdir, isfile, abspath, dirname, splitext
import os
from dotenv import load_dotenv

## Paths
DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data"))
SPACES_DATA_BASE = join(DATA_BASE, "siddata_semspaces")
SID_DATA_BASE = join(DATA_BASE, "siddata")
DATA_DUMP_DIR = abspath(join(dirname(__file__), "..", "..", "data_dump"))
ENV_FILE_PATH = abspath(join(dirname(__file__), "..", "..", "docker", ".env"))
load_dotenv(ENV_FILE_PATH)
GOOGLE_CREDENTIALS_FILE = join(DATA_BASE, "gloud_tools_key.json")

DATA_SET = "movies" # "movies", "places", "wines", "courses"
MDS_DIMENSIONS = 20 #20,50,100,200

DEBUG = True
RANDOM_SEED = None
MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"


#model-updown
SIDDATA_SEAFILE_SERVER = 'https://myshare.uni-osnabrueck.de'
SIDDATA_SEAFILE_REPOID = '0b3948a7-9483-4e26-a7bb-a123496ddfcf' #for modelupdown v2
SIDDATA_SEAFILE_REPOWRITE_ACC = 'cstenkamp@uni-osnabrueck.de'
SIDDATA_SEAFILE_REPOWRITE_PASSWORD = os.environ["SIDDATA_SEAFILE_REPOWRITE_PASSWORD"]
SIDDATA_SEAFILE_REPOREAD_ACC = 'cstenkamp@uni-osnabrueck.de'
SIDDATA_SEAFILE_REPOREAD_PASSWORD = os.environ["SIDDATA_SEAFILE_REPOWRITE_PASSWORD"]
SIDDATA_SEAFILE_REPO_BASEPATH = "backend_synced_models"
SIDDATA_SEAFILE_MODEL_VERSIONS = {"semanticspaces": 1, "siddata_semspaces": 1}





# checkers etc (THIS AT BOTTOM!!!)
import os
if not os.environ.get("SETTINGS_WARNINGS_GIVEN"):
    if DEBUG:
        print("DEBUG is activated!!!")
    if RANDOM_SEED:
        print("Using a random seed!!!")
os.environ["SETTINGS_WARNINGS_GIVEN"] = "1"