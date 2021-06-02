from os.path import join, isdir, isfile, abspath, dirname, splitext

## Paths
DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "semanticspaces"))
SID_DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "siddata"))



DATA_SET = "courses" # "movies", "places", "wines", "courses"
MDS_DIMENSIONS = 20 #20,50,100,200

DEBUG = True
RANDOM_SEED = None


# checkers etc
import os
if not os.environ.get("SETTINGS_WARNINGS_GIVEN"):
    if DEBUG:
        print("DEBUG is activated!!!")
    if RANDOM_SEED:
        print("Using a random seed!!!")
os.environ["SETTINGS_WARNINGS_GIVEN"] = "1"