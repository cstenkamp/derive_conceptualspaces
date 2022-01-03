from os.path import isfile, abspath
import os

ENV_PREFIX = "MA"
########################################################################################################################
############################################## the important parameters ################################################
########################################################################################################################

#!! use singular for these (bzw the form you'd use if there wasn't the "ALL_" before)
ALL_PP_COMPONENTS = ["autcsldp", "tcsldp", "aucsd2"] #,"tcsdp"       # If in preprocessing it should add coursetitle, lemmatize, etc
ALL_TRANSLATE_POLICY = ["translate"] #, "origlan", "onlyeng"         # If non-english descriptions should be translated
ALL_QUANTIFICATION_MEASURE = ["ppmi", "tf-idf", "count", "binary"]   # For the dissimiliarity Matrix of the Descripts
ALL_EXTRACTION_METHOD = ["pp_keybert", "tf-idf", "ppmi"]             # How candidate-terms are getting extracted         #TODO keybert
ALL_EMBED_ALGO = ["mds", "tsne", "isomap"]                           # Actual Embedding of the Descriptions
ALL_EMBED_DIMENSIONS = [100, 3] #, 50, 200                           # Actual Embedding of the Descriptions
ALL_DCM_QUANT_MEASURE = ["tf-idf", "count", "ppmi", "binary"]        # Quantification for the Doc-Keyphrase-Matrix       #TODO tag-share

FORBIDDEN_COMBIS = ["tsne_50d", "tsne_100d"]
#TODO: add (extraction_method in ["keybert", "pp_keybert"] and "2" in get_setting("PP_COMPONENTS")) to forbidden_combis

#set default-values for the ALL_... variables (always the first one)
# `DEFAULT_PP_COMPONENTS = ALL_PP_COMPONENTS[0] \n ...`
for k, v in {k[4:]: v[0] for k,v in dict(locals()).items() if isinstance(v, list) and k.startswith("ALL_")}.items():
    locals()["DEFAULT_"+k] = v

########################################################################################################################
################################################## other default values ################################################
########################################################################################################################

#DEBUG-Settings
DEFAULT_DEBUG = False
DEFAULT_DEBUG_N_ITEMS = 100
DEFAULT_RANDOM_SEED = 1
DEFAULT_VERBOSE = True

#Settings that influence the algorithm
DEFAULT_CANDIDATE_MIN_TERM_COUNT = 25
DEFAULT_FASTER_KEYBERT = False
DEFAULT_PRIM_LAMBDA = 0.45
DEFAULT_SEC_LAMBDA = 0.3
DEFAULT_STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
DEFAULT_COURSE_TYPES = ["colloquium", "seminar", "internship", "practice", "lecture"]
DEFAULT_MAX_NGRAM = 5
DEFAULT_NGRAMS_IN_EMBEDDING = False #If I should set the ngram-range already in the preprocess_descriptions step (makes the dissimiliarity-matrix a shitton more sparse)
DEFAULT_DISSIM_MAT_ONLY_PARTNERED = True

#Settings regarding the architecture/platform
DEFAULT_STRICT_METAINF_CHECKING = True


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