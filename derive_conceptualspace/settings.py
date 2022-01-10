from os.path import isfile, abspath
import os


ENV_PREFIX = "MA"
########################################################################################################################
################################ stuff necessary for loading & saving & architecture ###################################
########################################################################################################################

#I'm re-doing how a few things work regarding settings & the jsonPersistor. Previously, there was `all_params`, `forward_meta_inf` and `dir_struct`,
# and `all_params` was simply all the settings that start with "ALL_" in settings.py. From now on, I have to explicitly list params. And what ARE params?
# params = relevant configuration for a run, passed forward through the respective files, also making up filepath & filename.
# -> stimmt nicht, es gibt nur ASSERT_PARAMS und DIR_STRUCT
# und in snakemake generiert er die automatisch, und woimmer ein ALL_ existiert kann er das nehmen, ez

STRICT_METAINF_CHECKING = True

ASSERT_PARAMS = ["FASTER_KEYBERT", "CANDIDATE_MIN_TERM_COUNT"]
DIR_STRUCT = ["debug_{debug}",
              "{pp_components}_{translate_policy}_minwords{min_words_per_desc}",
              "{quantification_measure}_{embed_algo}_{embed_dimensions}d",
              "{extraction_method}_{dcm_quant_measure}"]
FNAME_PARAMS = ["DEBUG", "PP_COMPONENTS", "TRANSLATE_POLICY", "MIN_WORDS_PER_DESC", "QUANTIFICATION_MEASURE", "EMBED_ALGO",
                "EMBED_DIMENSIONS", "EXTRACTION_METHOD", "DCM_QUANT_MEASURE"]


NORMALIFY_PARAMS = ["QUANTIFICATION_MEASURE", "EXTRACTON_METHOD", "EMBED_ALGO", "DCM_QUANT_MEASURE"] #for all params that are in this, eg `Tf-IdF` will become `tfidf`
########################################################################################################################
############################################## the important parameters ################################################
########################################################################################################################

ALL_MIN_WORDS_PER_DESC = [50, 20]

#!! use singular for these (bzw the form you'd use if there wasn't the "ALL_" before)
ALL_PP_COMPONENTS = ["aucsd2", "autcsldp"] #,"tcsdp"                 # If in preprocessing it should add coursetitle, lemmatize, etc #TODO "autcsldp", "tcsldp" (gehen gerade nicht weil die nicht mit ngrams klarkommen)
ALL_TRANSLATE_POLICY = ["translate", "origlang", "onlyeng"]          # If non-english descriptions should be translated
ALL_QUANTIFICATION_MEASURE = ["ppmi", "tfidf", "count", "binary"]    # For the dissimiliarity Matrix of the Descripts
ALL_EXTRACTION_METHOD = ["tfidf", "pp_keybert", "ppmi"]              # How candidate-terms are getting extracted         #TODO keybert
ALL_EMBED_ALGO = ["mds", "tsne", "isomap"]                           # Actual Embedding of the Descriptions
ALL_EMBED_DIMENSIONS = [100, 3] #, 50, 200                           # Actual Embedding of the Descriptions
ALL_DCM_QUANT_MEASURE = ["ppmi", "tfidf", "count", "binary"]         # Quantification for the Doc-Keyphrase-Matrix       #TODO tag-share


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
DEFAULT_RIG_ASSERTS = True

#Settings that influence the algorithm
DEFAULT_CANDIDATE_MIN_TERM_COUNT = 25
DEFAULT_FASTER_KEYBERT = False
DEFAULT_PRIM_LAMBDA = 0.45
DEFAULT_SEC_LAMBDA = 0.3
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
DEFAULT_QUANTEXTRACT_FORCETAKE_PERC = 0.98

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


from functools import wraps
import sys

def notify_jsonpersister(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        res = fn(*args, **kwargs)
        if hasattr(sys.stdout, "ctx"):  # TODO getting the json_serializer this way is dirty as fuck!
            if "json_persister" in sys.stdout.ctx.obj:
                sys.stdout.ctx.obj["json_persister"].add_config(args[0], res)
        return res
    return wrapped


normalify = lambda txt: "".join([i for i in txt.lower() if i.isalpha() or i in "_"])

def cast_config(k, v):
    if k.upper() in NORMALIFY_PARAMS:
        v = normalify(v)
    if isinstance(v, str) and v.isnumeric():
        v = int(v)
    if "DEFAULT_" + k.upper() in globals() and isinstance(globals()["DEFAULT_" + k.upper()], bool) and v in [0, 1]:
        v = bool(v)
    return v


@notify_jsonpersister
def get_setting(name, default_none=False, silent=False, set_env_from_default=False, stay_silent=False):
    suppress_further = True if not silent else True if stay_silent else False
    if get_envvar(ENV_PREFIX+"_"+name) is not None:
        return cast_config(get_envvar(ENV_PREFIX+"_"+name) if get_envvar(ENV_PREFIX+"_"+name) != "none" else None)
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