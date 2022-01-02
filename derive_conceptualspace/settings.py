from os.path import join, isdir, isfile, abspath, dirname, splitext
import os

ENV_PREFIX = "MA"
########################################################################################################################
############################################## the important parameters ################################################
########################################################################################################################

#!! use singular for these (bzw the form you'd use if there wasn't the "ALL_" before)
ALL_PP_COMPONENTS = ["autcsldp", "tcsldp", "au2"] #,"tcsdp"          # If in preprocessing it should add coursetitle, lemmatize, etc
ALL_TRANSLATE_POLICY = ["translate"] #, "origlan", "onlyeng"         # If non-english descriptions should be translated
ALL_QUANTIFICATION_MEASURE = ["ppmi", "tf-idf", "count", "binary"]   # For the dissimiliarity Matrix of the Descripts
ALL_EXTRACTION_METHOD = ["pp_keybert", "tf-idf", "ppmi"]             # How candidate-terms are getting extracted         #TODO keybert
ALL_EMBED_ALGO = ["mds", "tsne", "isomap"]                           # Actual Embedding of the Descriptions
ALL_EMBED_DIMENSIONS = [100, 3] #, 50, 200                           # Actual Embedding of the Descriptions
ALL_DCM_QUANT_MEASURE = ["tf-idf", "count", "ppmi", "binary"]        # Quantification for the Doc-Keyphrase-Matrix       #TODO tag-share

FORBIDDEN_COMBIS = ["tsne_50d", "tsne_100d"]

#set default-values for the ALL_... variables (always the first one)
# `DEFAULT_PP_COMPONENTS = ALL_PP_COMPONENTS[0] \n ...`
for k, v in {k[4:]: v[0] for k,v in dict(locals()).items() if isinstance(v, list) and k.startswith("ALL_")}.items():
    locals()["DEFAULT_"+k] = v

########################################################################################################################
################################################## other default values ################################################
########################################################################################################################


DEFAULT_DEBUG = False
DEFAULT_DEBUG_N_ITEMS = 100

DEFAULT_CANDIDATE_MIN_TERM_COUNT = 25
DEFAULT_FASTER_KEYBERT = False
DEFAULT_PRIM_LAMBDA = 0.45
DEFAULT_SEC_LAMBDA = 0.3

DEFAULT_STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
DEFAULT_COURSE_TYPES = ["colloquium", "seminar", "internship", "practice", "lecture"]
DEFAULT_RANDOM_SEED = 1
DEFAULT_VERBOSE = True
DEFAULT_STRICT_METAINF_CHECKING = True

########################################################################################################################
##################################################### old stuff ########################################################
########################################################################################################################

# DATA_BASE = abspath(join(dirname(__file__), "", "..", "..", "data"))
# SPACES_DATA_BASE = join(DATA_BASE, "semanticspaces")
# SID_DATA_BASE = join(DATA_BASE, "siddata_semspaces")
# DATA_DUMP_DIR = abspath(join(dirname(__file__), "", "..", "data_dump"))
# GOOGLE_CREDENTIALS_FILE = join(DATA_BASE, "gcloud_tools_key.json")
# SIDDATA_SEAFILE_SERVER = 'https://myshare.uni-osnabrueck.de'
# SIDDATA_SEAFILE_REPOID = '0b3948a7-9483-4e26-a7bb-a123496ddfcf' #for modelupdown v2
# SIDDATA_SEAFILE_REPOWRITE_ACC = os.getenv("SIDDATA_SEAFILE_REPOWRITE_ACC")
# SIDDATA_SEAFILE_REPOWRITE_PASSWORD = os.getenv("SIDDATA_SEAFILE_REPOWRITE_PASSWORD")
# SIDDATA_SEAFILE_REPOREAD_ACC = os.getenv("SIDDATA_SEAFILE_REPOREAD_ACC")
# SIDDATA_SEAFILE_REPOREAD_PASSWORD = os.getenv("SIDDATA_SEAFILE_REPOREADr_PASSWORD")
# SIDDATA_SEAFILE_REPO_BASEPATH = "backend_synced_models"
# SIDDATA_SEAFILE_MODEL_VERSIONS = {"siddata_semspaces": 1} #"semanticspaces": 1,
# DATA_SET = "movies" # "movies", "places", "wines", "courses"
# MONGO_URI = f"mongodb://{os.environ['MONGO_INITDB_ROOT_USERNAME']}:{os.environ['MONGO_INITDB_ROOT_PASSWORD']}@127.0.0.1/?authMechanism=SCRAM-SHA-1"

########################################################################################################################
######################################## set and get settings/env-vars #################################################
########################################################################################################################

def set_envvar(envvarname, value):
    if isinstance(value, bool):
        if value:
            os.environ[envvarname] = "1"
        else:
            os.environ[envvarname + "_FALSE"] = "1"
    else:
        os.environ[envvarname] = str(value)


def get_envvar(envvarname):
    if os.getenv(envvarname):
        tmp = os.environ[envvarname]
        if tmp.lower() == "none":
            return "none"
        elif tmp.isnumeric():
            return int(tmp)
        elif all([i.isdecimal() or i in ".," for i in tmp]):
            return float(tmp)
        return tmp
    elif os.getenv(envvarname+"_FALSE"):
        return False
    return None


def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False):
    suppress_further = True if not silent else True if stay_silent else False
    if get_envvar(ENV_PREFIX+"_"+name) is not None:
        return get_envvar(ENV_PREFIX+"_"+name) if get_envvar(ENV_PREFIX+"_"+name) != "none" else None
    if "DEFAULT_"+name in globals():
        if not silent and not get_envvar(ENV_PREFIX+"_"+name+"_shutup"):
            print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
        if suppress_further and not get_envvar(ENV_PREFIX + "_" + name + "_shutup"):
            set_envvar(ENV_PREFIX+"_"+name+"_shutup", True)
        if set_env_from_default:
            set_envvar(ENV_PREFIX+"_"+name, globals()['DEFAULT_'+name])
        return globals()["DEFAULT_"+name]
    if default_none:
        return None
    assert False, f"Couldn't get setting {name}"


########################################################################################################################
########################################### KEEP THIS AT THE BOTTOM! ###################################################
########################################################################################################################

STARTUP_ENVVARS = {k:v for k,v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}

#actually make all defined directories (global vars that end in "_PATH")
for key, val in dict(locals()).items():
    if key.endswith('_PATH') and not isfile(val):
        locals()[key] = abspath(val)
        os.makedirs(locals()[key], exist_ok=True)