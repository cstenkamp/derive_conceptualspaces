from os.path import join, isdir, isfile, abspath, dirname, splitext

DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "semanticspaces"))

SID_DATA_BASE = abspath(join(dirname(__file__), "..", "..", "..", "data", "siddata"))

DATA_SET = "movies" # "movies", "places", "wines"
MDS_DIMENSIONS = 100 #20,50,100,200

DEBUG = True
RANDOM_SEED = 1