import contextlib
from os.path import isfile, abspath, dirname, join
import os

ENV_PREFIX = "MA"
########################################################################################################################
############################################## the important parameters ################################################
########################################################################################################################
# ALL_MIN_WORDS_PER_DESC = [10, 50, 100]


#!! use singular for these (bzw the form you'd use if there wasn't the "ALL_" before)
ALL_PP_COMPONENTS = ["faucsd2"]#, "autcsldp"] #,"tcsdp"                 # If in preprocessing it should add coursetitle, lemmatize, etc #TODO "autcsldp", "tcsldp" (gehen gerade nicht weil die nicht mit ngrams klarkommen)
ALL_TRANSLATE_POLICY = ["translate"]#, "origlang", "onlyeng"]          # If non-english descriptions should be translated
ALL_EMBED_ALGO = ["mds"]#, "tsne", "isomap"]                           # Actual Embedding of the Descriptions
ALL_EMBED_DIMENSIONS = [100]#, 3] #, 50, 200                           # Actual Embedding of the Descriptions
ALL_QUANTIFICATION_MEASURE = ["ppmi"]#, "tfidf", "count", "binary"]    # For the dissimiliarity Matrix of the Descripts
ALL_EXTRACTION_METHOD = ["tfidf"]#, "pp_keybert", "ppmi"]              # How candidate-terms are getting extracted         #TODO keybert
ALL_DCM_QUANT_MEASURE = ["count"]#, "tfidf", "count", "binary"]         # Quantification for the Doc-Keyphrase-Matrix       #TODO tag-share
#TODO do I even need the distinction between DCM_QUANT_MEASURE and CLASSIFIER_COMPARETO_RANKING ???

FORBIDDEN_COMBIS = ["tsne_50d", "tsne_100d"]


#set default-values for the ALL_... variables (always the first one)
# `DEFAULT_PP_COMPONENTS = ALL_PP_COMPONENTS[0] \n ...`
for k, v in {k[4:]: v[0] for k,v in dict(locals()).items() if isinstance(v, list) and k.startswith("ALL_")}.items():
    locals()["DEFAULT_"+k] = v

NORMALIFY_PARAMS = ["QUANTIFICATION_MEASURE", "EXTRACTON_METHOD", "EMBED_ALGO", "DCM_QUANT_MEASURE", "CLASSIFIER_COMPARETO_RANKING"]
#for all params that are in this, eg `Tf-IdF` will become `tfidf`
########################################################################################################################
################################################## other default values ################################################
########################################################################################################################

#DEBUG-Settings
DEFAULT_DEBUG = False
DEFAULT_DEBUG_N_ITEMS = 100
DEFAULT_RANDOM_SEED = 1
DEFAULT_VERBOSE = True
DEFAULT_RIG_ASSERTS = True

#Settings that influence the algorithm
DEFAULT_DISSIM_MEASURE = "norm_ang_dist"  #can be: ["cosine", "norm_ang_dist"]
DEFAULT_CANDIDATE_MIN_TERM_COUNT = 25
DEFAULT_FASTER_KEYBERT = False
DEFAULT_PRIM_LAMBDA = 0.45
DEFAULT_SEC_LAMBDA = 0.1
DEFAULT_STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
DEFAULT_COURSE_TYPES = ["colloquium", "seminar", "internship", "practice", "lecture"]
DEFAULT_CUSTOM_STOPWORDS = ("one", "also", "take")
DEFAULT_MAX_NGRAM = 5
DEFAULT_NGRAMS_IN_EMBEDDING = False #If I should set the ngram-range already in the preprocess_descriptions step (makes the dissimiliarity-matrix a shitton more sparse)
DEFAULT_DISSIM_MAT_ONLY_PARTNERED = True
DEFAULT_CANDS_USE_NDOCS_COUNT = True
DEFAULT_MIN_WORDS_PER_DESC = 50

DEFAULT_QUANTEXTRACT_MAXPERDOC_ABS = 20
DEFAULT_QUANTEXTRACT_MAXPERDOC_REL = 0.1
DEFAULT_QUANTEXTRACT_MINVAL = None
DEFAULT_QUANTEXTRACT_MINVAL_PERC = 0.8
DEFAULT_QUANTEXTRACT_MINPERDOC = 0
DEFAULT_QUANTEXTRACT_FORCETAKE_PERC = 0.99
#TODO statt dieser settings kann ich auch ne number an demanded candidateterms vorgeben, und die candidates kriegen alle einn wert wie gut sie sind und der standard wird so lange gelowert bis die #demandedterms ungefähr erreicht sind

DEFAULT_CLASSIFIER_COMPARETO_RANKING = "count"  #so far: one of ["count", "ppmi"]
DEFAULT_CLASSIFIER_SUCCMETRIC = "cohen_kappa"

@contextlib.contextmanager
def set_noninfluentials():
    pre_globals = dict(globals())
    yield
    added_vars = set(i for i in globals().keys() - pre_globals.keys() - {"pre_globals"} if not i.startswith("__py_debug"))
    globals()["NON_INFLUENTIAL_CONFIGS"] += [i[len("DEFAULT_"):] if i.startswith("DEFAULT_") else i for i in added_vars]

#Settings regarding the architecture/platform
NON_INFLUENTIAL_CONFIGS = ["CONF_FILE"]
with set_noninfluentials(): #this context-manager adds all settings from here to the NON_INFLUENTIAL_CONFIGS variable
    CONF_PRIORITY = ["dependency", "cmd_args", "env_vars", "conf_file", "dataset_class", "defaults"] #no distinction between env_file and env_var bc load_dotenv is executed eagerly and just overwrites envvars from envfile
    DEFAULT_BASE_DIR = abspath(join(dirname(__file__), "..", "..", ENV_PREFIX+"_data"))
    DEFAULT_NOTIFY_TELEGRAM = False

    DEFAULT_RAW_DESCRIPTIONS_FILE = "kurse-beschreibungen.csv"
    DEFAULT_LANGUAGES_FILE = "languages.json"
    DEFAULT_TRANSLATIONS_FILE = "translated_descriptions.json"
    DEFAULT_TITLE_LANGUAGES_FILE = "title_languages.json"
    DEFAULT_TITLE_TRANSLATIONS_FILE = "translated_titles.json"
    DEFAULT_LOG = "Info"
    DEFAULT_LOGFILE = ""
    DEFAULT_CONF_FILE = None

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
        elif tmp == "True":
            return True
        elif tmp == "False":
            return False
        elif tmp.isnumeric() and "DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):] in globals() and isinstance(globals()["DEFAULT_"+envvarname[len(ENV_PREFIX+"_"):]], bool) and tmp in [0, 1, "0", "1"]:
            return bool(int(tmp))
        elif tmp.isnumeric():
            return int(tmp)
        elif all([i.isdecimal() or i in ".," for i in tmp]):
            return float(tmp)
        return tmp
    elif os.getenv(envvarname+"_FALSE"):
        return False
    return None


from functools import wraps
import sys

# def notify_jsonpersister(fn):
#     @wraps(fn)
#     def wrapped(*args, **kwargs):
#         res = fn(*args, **kwargs)
#         if not kwargs.get("fordefault"):
#             if hasattr(sys.stdout, "ctx") and "json_persister" in sys.stdout.ctx.obj:  # TODO getting the json_serializer this way is dirty as fuck!
#                 sys.stdout.ctx.obj["json_persister"].add_config(args[0].lower(), res)
#         return res
#     return wrapped


# @notify_jsonpersister
def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False, fordefault=True):
    #!!! diese funktion darf NICHTS machen außer sys.stdout.ctx.get_config(name) returnen!!! alles an processing gehört in die get_config!!!
    if hasattr(sys.stdout, "ctx"):
        return sys.stdout.ctx.get_config(name)
    #TODO overhaul 16.01.2022: einige Dinge von der old version waren schon sinnvoll, zum beispiel das bescheid sagen wenn von default, gucken
    # was ich davon wieder haben möchte
    # if fordefault: #fordefault is used for click's default-values. In those situations, it it should NOT notify the json-persister!
    #     silent = True
    #     stay_silent = False
    #     set_env_from_default = False
    #     default_none = True
    #     # return "default" #("default", globals().get("DEFAULT_"+name, "NO_DEFAULT"))
    # suppress_further = True if not silent else True if stay_silent else False
    # if get_envvar(get_envvarname(name, assert_hasdefault=False)) is not None:
    #     return get_envvar(get_envvarname(name, assert_hasdefault=False)) if get_envvar(get_envvarname(name, assert_hasdefault=False)) != "none" else None
    # if "DEFAULT_"+get_envvarname(name, assert_hasdefault=False, without_prefix=True) in globals():
    #     if not silent and not get_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup"):
    #         print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
    #     if suppress_further and not get_envvar(get_envvarname(name, assert_hasdefault=False) + "_shutup"):
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup", True)
    #     if set_env_from_default:
    #         set_envvar(get_envvarname(name, assert_hasdefault=False)+name, globals()['DEFAULT_'+name])
    #     return globals()["DEFAULT_"+name]
    # if default_none:
    #     return None
    # raise ValueError(f"There is no default-value for setting {name}, you have to explicitly pass it!")


def get_envvarname(config, assert_hasdefault=True, without_prefix=False):
    config = config.upper()
    if assert_hasdefault:
        assert "DEFAULT_"+config in globals(), f"there is no default value for {config}!"
    if without_prefix:
        return config
    return ENV_PREFIX+"_"+config

########################################################################################################################
# as of 16.01.2022

def cast_config(k, v):
    if isinstance(v, str) and v.isnumeric():
        v = int(v)
    if "DEFAULT_" + k in globals() and isinstance(globals()["DEFAULT_" + k], bool) and v in [0, 1]:
        v = bool(v)
    if v == "True":
        v = True
    if v == "False":
        v = False
    if isinstance(v, list):
        v = tuple(v)
    if "DEFAULT_" + k in globals() and type(globals()["DEFAULT_" + k]) != type(v) and (v != None and globals()["DEFAULT_" + k] != None):
        assert False, "TODO overhaul 16.01.2022"
    return v

def standardize_config_name(configname):
    return configname.upper().replace("-","_")

def standardize_config_val(configname, configval):
    if configname in NORMALIFY_PARAMS and configval is not None:
        configval = "".join([i for i in configval.lower() if i.isalpha() or i in "_"])
    configval = cast_config(configname, configval)
    return configval

def standardize_config(configname, configval):
    configname = standardize_config_name(configname)
    return configname, standardize_config_val(configname, configval)

def get_defaultsetting(key):
    if "DEFAULT_" + key not in globals():
        raise ValueError(f"You didn't provide a value for {key} and there is no default-value!")
    default = globals()["DEFAULT_"+key]
    if key not in NON_INFLUENTIAL_CONFIGS:
        print(f"returning setting for {key} from default value: {default}")
    return default

########################################################################################################################
########################################### KEEP THIS AT THE BOTTOM! ###################################################
########################################################################################################################


#TODO overhaul 16.01.2022: do this in Context.pre_actualcommand_ops
STARTUP_ENVVARS = {k:v for k,v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}

#actually make all defined directories (global vars that end in "_PATH")
for key, val in dict(locals()).items():
    if key.endswith('_PATH') and not isfile(val):
        locals()[key] = abspath(val)
        os.makedirs(locals()[key], exist_ok=True)