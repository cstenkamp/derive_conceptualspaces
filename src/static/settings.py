from os.path import join, isdir, isfile, abspath, dirname, splitext
import os
from dotenv import load_dotenv

## Paths
DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "semanticspaces"))
SID_DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "siddata"))
DATA_DUMP_DIR = abspath(join(dirname(__file__), "..", "..", "data_dump"))
ENV_FILE_PATH = abspath(join(dirname(__file__), "..", "..", "docker", ".env"))
load_dotenv(ENV_FILE_PATH)

DATA_SET = "courses" # "movies", "places", "wines", "courses"
MDS_DIMENSIONS = 20 #20,50,100,200

DEBUG = True
RANDOM_SEED = None
MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"

# checkers etc
import os
if not os.environ.get("SETTINGS_WARNINGS_GIVEN"):
    if DEBUG:
        print("DEBUG is activated!!!")
    if RANDOM_SEED:
        print("Using a random seed!!!")
os.environ["SETTINGS_WARNINGS_GIVEN"] = "1"