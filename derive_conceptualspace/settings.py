from os.path import isfile, abspath
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
DEFAULT_CUSTOM_STOPWORDS = ["one", "also", "take"]
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

DEFAULT_CLASSIFIER_COMPARETO_RANKING = "count"  #so far: one of ["count", "ppmi"]
DEFAULT_CLASSIFIER_SUCCMETRIC = "cohen_kappa"

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

def notify_jsonpersister(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not kwargs.get("fordefault"):
            if hasattr(sys.stdout, "ctx") and "json_persister" in sys.stdout.ctx.obj:  # TODO getting the json_serializer this way is dirty as fuck!
                sys.stdout.ctx.obj["json_persister"].add_config(args[0].lower(), res)
        return res
    return wrapped


@notify_jsonpersister
def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False, fordefault=True):
    if fordefault: #fordefault is used for click's default-values. In those situations, it it should NOT notify the json-persister!
        silent = True
        stay_silent = False
        set_env_from_default = False
    suppress_further = True if not silent else True if stay_silent else False
    if get_envvar(get_envvarname(name, assert_hasdefault=False)) is not None:
        return get_envvar(get_envvarname(name, assert_hasdefault=False)) if get_envvar(get_envvarname(name, assert_hasdefault=False)) != "none" else None
    if "DEFAULT_"+get_envvarname(name, assert_hasdefault=False, without_prefix=True) in globals():
        if not silent and not get_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup"):
            print(f"returning setting for {name} from default value: {globals()['DEFAULT_'+name]}")
        if suppress_further and not get_envvar(get_envvarname(name, assert_hasdefault=False) + "_shutup"):
            set_envvar(get_envvarname(name, assert_hasdefault=False)+"_shutup", True)
        if set_env_from_default:
            set_envvar(get_envvarname(name, assert_hasdefault=False)+name, globals()['DEFAULT_'+name])
        return globals()["DEFAULT_"+name]
    if default_none:
        return None
    assert False, f"Couldn't get setting {name}"


def get_envvarname(config, assert_hasdefault=True, without_prefix=False):
    config = config.upper()
    if assert_hasdefault:
        assert "DEFAULT_"+config in globals(), f"there is no default value for {config}!"
    if without_prefix:
        return config
    return ENV_PREFIX+"_"+config

########################################################################################################################
########################################### KEEP THIS AT THE BOTTOM! ###################################################
########################################################################################################################

STARTUP_ENVVARS = {k:v for k,v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}

#actually make all defined directories (global vars that end in "_PATH")
for key, val in dict(locals()).items():
    if key.endswith('_PATH') and not isfile(val):
        locals()[key] = abspath(val)
        os.makedirs(locals()[key], exist_ok=True)