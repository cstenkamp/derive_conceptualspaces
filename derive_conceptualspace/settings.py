import psutil
import contextlib
from os.path import abspath, dirname, join
import os
import sys

from misc_util.pretty_print import pretty_print as print

ENV_PREFIX = "MA"
########################################################################################################################
############################################## the important parameters ################################################
########################################################################################################################

# PP-Components:
# sentwise_merge = "m",  add_additionals = "f", add_title = "a", add_subtitle = "u", remove_htmltags = "h", sent_tokenize = "t", convert_lower = "c",
# remove_stopwords = "s", lemmatize = "l", remove_diacritics = "d", remove_punctuation = "p", use_skcountvec = "2",
# m,f,a,u,h,c should be default for now -> mfauhc
# if not sk-countvec then t             -> mfauhc2 & mfauhtc
# d & p make for more merging of kw's   -> mfauhcd2 & mfauhtcdp (d not possible with sklearn)
# TODO: maybe try without remove-stopwords eventually? TODO: better german stopwords?
# left are: lemmatize                   -> mfauhcsd2, mfauhtcdp, mfauhtcldp
# maybe we try without title once       -> mfhcsd2, mfauhcsd2, mfauhtcsdp, mfauhtcsldp
# TODO: synsetisize as soon as GermaNet
# old candidates were: mfacsd2  (sentwisemerge, add-additionals, add-title, convert-lower, remove-stopwords, remove-diacritics, use-skcount)
#                      autcsldp (add-title, add-subtitle, sent-tokenize, convert-lower, remove-stopwords, lemmatize, remove-diacritics, remove-punctuation)
#                      tcsdp    (sent-tokenize, convert-lower, remove-stopwords, remove-diacritics, remove-punctuation)


#!! use singular for these (bzw the form you'd use if there wasn't the "ALL_" before)
ALL_PP_COMPONENTS = ["mfhcsd2", "mfauhcsd2", "mfauhtcsdp", "mfauhtcsldp"]     # If in preprocessing it should add title, lemmatize, etc
ALL_TRANSLATE_POLICY = ["onlyorig"] #["onlyorig", "translate", "origlang"]                    # If non-english/non-german/... descriptions should be translated
ALL_EMBED_ALGO = ["mds", "tsne", "isomap"]                                      # Actual Embedding of the Descriptions
ALL_EMBED_DIMENSIONS = [100, 3, 20, 50, 200]                                    # Actual Embedding of the Descriptions
ALL_QUANTIFICATION_MEASURE = ["count", "tfidf", "ppmi", "binary", "tf"]         # For the dissimiliarity Matrix of the Descripts
ALL_EXTRACTION_METHOD = ["keybert", "pp_keybert", "tfidf", "tf", "all", "ppmi"] # How candidate-terms are getting extracted
ALL_DCM_QUANT_MEASURE = ["count", "tfidf", "ppmi", "binary", "tf"]              # Quantification for the Doc-Keyphrase-Matrix (=CLASSIFIER_COMPARETO_RANKING)    #TODO tag-share
ALL_CLASSIFIER = ["SVM", "SVM_square", "SVM"]
ALL_KAPPA_WEIGHTS = ["quadratic", "linear", None]

#set default-values for the ALL_... variables (always the first one) `DEFAULT_PP_COMPONENTS = ALL_PP_COMPONENTS[0] \n ...`
for k, v in {k[4:]: v[0] for k,v in dict(locals()).items() if isinstance(v, list) and k.startswith("ALL_")}.items():
    locals()["DEFAULT_"+k] = v

########################################################################################################################
################################################## other default values ################################################
########################################################################################################################

#DEBUG-Settings
DEFAULT_DEBUG = False
DEFAULT_DEBUG_N_ITEMS = 200
DEFAULT_RANDOM_SEED = 1
DEFAULT_SEED_ONLY_IN_DEBUG = True
DEFAULT_VERBOSE = True
IS_INTERACTIVE = "PYCHARM_HOSTED" in os.environ
DEFAULT_N_CPUS = max(psutil.cpu_count(logical=False), psutil.cpu_count(logical=True)-2)


#Settings that influence the algorithm
DEFAULT_LANGUAGE = "de"
DEFAULT_DISSIM_MEASURE = "norm_ang_dist"  #can be: ["cosine", "norm_ang_dist"]
DEFAULT_CANDIDATE_MIN_TERM_COUNT = 25
DEFAULT_FASTER_KEYBERT = False
DEFAULT_PRIM_LAMBDA = 0.45
DEFAULT_SEC_LAMBDA = 0.1
DEFAULT_STANFORDNLP_VERSION = "4.2.2" #whatever's newest at https://stanfordnlp.github.io/CoreNLP/history.html
DEFAULT_COURSE_TYPES = ("colloquium", "seminar", "internship", "practice", "lecture")
DEFAULT_CUSTOM_STOPWORDS = ("one", "also", "take")
DEFAULT_MAX_NGRAM = 5
DEFAULT_NGRAMS_IN_EMBEDDING = False # if the dissimilarity-matrix should already consider n-grams (makes it a LOT more sparse)
DEFAULT_DISSIM_MAT_ONLY_PARTNERED = True
DEFAULT_CANDS_USE_NDOCS_COUNT = True
DEFAULT_MIN_WORDS_PER_DESC = 50
DEFAULT_USE_STANZA = False #for now only for sentence-tokenization, god I hate nltk.

DEFAULT_QUANTEXTRACT_MAXPERDOC_ABS = 40
DEFAULT_QUANTEXTRACT_MAXPERDOC_REL = 0.2
DEFAULT_QUANTEXTRACT_MINVAL = None
DEFAULT_QUANTEXTRACT_MINVAL_PERC = 0.6
DEFAULT_QUANTEXTRACT_MINPERDOC = 0
DEFAULT_QUANTEXTRACT_FORCETAKE_PERC = 0.99
#TODO statt dieser settings kann ich auch ne number an demanded candidateterms vorgeben, und die candidates kriegen alle einn wert wie gut sie sind und der standard wird so lange gelowert bis die #demandedterms ungefähr erreicht sind

DEFAULT_CLASSIFIER_SUCCMETRIC = "cohen_kappa"

########################################################################################################################
######################################### settings regarding the architecture ##########################################
########################################################################################################################

@contextlib.contextmanager
def set_noninfluentials():
    pre_globals = dict(globals())
    yield
    added_vars = set(i for i in globals().keys() - pre_globals.keys() - {"pre_globals"} if not i.startswith("__py_debug"))
    globals()["NON_INFLUENTIAL_CONFIGS"] += [i[len("DEFAULT_"):] if i.startswith("DEFAULT_") else i for i in added_vars]


NON_INFLUENTIAL_CONFIGS = ["CONF_FILE", "GOOGLE_CREDENTIALS_FILE", "VERBOSE", "STARTUP_ENVVARS", "IS_INTERACTIVE", "ENV_PREFIX", "DEFAULT_SEED_ONLY_IN_DEBUG", "N_CPUS"]
with set_noninfluentials(): #this context-manager adds all settings from here to the NON_INFLUENTIAL_CONFIGS variable

    STANDARD_HOSTNAME = 'chris-ThinkPad-E480'
    DEFAULT_BASE_DIR = abspath(join(dirname(__file__), "..", "..", ENV_PREFIX+"_data"))
    DEFAULT_NOTIFY_TELEGRAM = False

    DEFAULT_RAW_DESCRIPTIONS_FILE = "raw_descriptions.csv"
    DEFAULT_LOG = "Info"
    DEFAULT_LOGFILE = ""
    DEFAULT_CONF_FILE = None
    SMK_DEFAULT_CONFIG = True #If for the "default" rule, snakemake should read the config-file like click does.

    DIR_STRUCT = ["{dataset}",
                  "{language}_debug_{debug}",
                  "{pp_components}_{translate_policy}_minwords{min_words_per_desc}",
                  "embedding_{quantification_measure}",
                  "{embed_algo}_{embed_dimensions}d",
                  "{extraction_method}_{dcm_quant_measure}_{kappa_weights}"]

    FORBIDDEN_COMBIS = ["tsne_50d", "tsne_100d"]
    NORMALIFY_PARAMS = ["QUANTIFICATION_MEASURE", "EXTRACTON_METHOD", "EMBED_ALGO", "DCM_QUANT_MEASURE"]  #for all params that are in this, eg `Tf-IdF` will become `tfidf`
    CONF_PRIORITY = ["force", "smk_wildcard", "dependency", "cmd_args", "env_vars", "smk_args", "conf_file", "dataset_class", "defaults"] #no distinction between env_file and env_var bc load_dotenv is executed eagerly and just overwrites envvars from envfile
    #note that snakemake reads the conf_file differently and sets env-vars (that however apply force) from the configurations
    MAY_DIFFER_IN_DEPENDENCIES = ["DEBUG", "RANDOM_SEED", "CANDIDATE_MIN_TERM_COUNT", "BASE_DIR", "DEBUG_N_ITEMS", "CONF_FILE"]+NON_INFLUENTIAL_CONFIGS
    DEFAULT_DEP_PREFERS_NONDEBUG = True
    DEFAULT_DO_SANITYCHECKS = False  #sanity-checks check for code-correctness and can increase code-runtime by a lot. Running them once on each dataset&parameter-combination after changes is recommended.

########################################################################################################################
######################################## set and get settings/env-vars #################################################
########################################################################################################################

def get_setting(name, **kwargs):
    #!!! diese funktion darf NICHTS machen außer sys.stdout.ctx.get_config(name) returnen!!! alles an processing gehört in die get_config!!!
    if hasattr(sys.stdout, "ctx"):
        return sys.stdout.ctx.get_config(name, **kwargs)
    if os.path.basename(sys.argv[0]) == 'run_fb_classifier.py':
        return False #TODO pass ich das auf das neue an? Ja? nein?
    raise Exception("Unexpected!")

def forbid_setting(name, **kwargs):
    if hasattr(sys.stdout, "ctx"):
        return sys.stdout.ctx.forbid_config(name, **kwargs)
    raise Exception("Unexpected!")


def get_ncpu(ram_per_core=None, ignore_debug=False):
    import psutil
    if not ignore_debug and get_setting("DEBUG"):
        return 1
    ncpus = get_setting("N_CPUS")
    if os.getenv("NSLOTS"):
        if not os.getenv(f"{ENV_PREFIX}shutups_nslots"):
            print("This machine has been given NSLOTS and it is", os.getenv("NSLOTS"))
        os.environ[f"{ENV_PREFIX}shutups_nslots"] = "1"
        ncpus = max(int(os.environ["NSLOTS"]) - 1, 1)
        # "To ensure that your job is scheduled on a host you are advised not to have request more  than $NCPU -1 parallel environments."
    if "GOTO_NUM_THREADS" in os.environ:  # see https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#threads
        print(f"Snakemake restricts the #Threads to {os.environ['GOTO_NUM_THREADS']}")
        ncpus = min(ncpus, int(os.environ["GOTO_NUM_THREADS"]))
    if ram_per_core: #TODO if I'm on the grid, I should have an env-var with the assigned ram and use that instead!!
        ncpus = min(ncpus, round(psutil.virtual_memory().total / 1024 / 1024 / 1024 / ram_per_core))
        if "SGE_SMK_mem" in os.environ and os.environ["SGE_SMK_mem"].endswith("G"):
            ncpus = min(ncpus, round(int(os.environ["SGE_SMK_mem"][:-1]) / ram_per_core)) # max. 1 thread per XGB RAM
    return ncpus


########################################################################################################################

def cast_config(k, v):
    if isinstance(v, str) and v.isnumeric():
        v = int(v)
    if isinstance(v, str) and all([i.isdecimal() or i in ".," for i in v]):
        v = float(v)
    if "DEFAULT_" + k in globals() and isinstance(globals()["DEFAULT_" + k], bool) and v in [0, 1]:
        v = bool(v)
    if v == "True":
        v = True
    if v == "False":
        v = False
    if isinstance(v, list):
        v = tuple(v)
    if "DEFAULT_" + k in globals() and type(globals()["DEFAULT_" + k]) != type(v) and (v != None and globals()["DEFAULT_" + k] != None) and not all(i in (int, float) for i in (type(globals()["DEFAULT_" + k]), type(v))):
        raise Exception(f"Default {k}: {globals()['DEFAULT_' + k] }, should-be: {v}")
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

def get_defaultsetting(key, silent=False, default_false=False):
    if "DEFAULT_" + key not in globals():
        if not default_false:
            raise ValueError(f"You didn't provide a value for {key} and there is no default-value!")
        else:
            return False
    default = globals()["DEFAULT_"+key]
    if key not in NON_INFLUENTIAL_CONFIGS and not silent:
        if not os.getenv(f"{ENV_PREFIX}shutups_{key}"):
            print(f"returning {key} from default: *b*{default}*b*")
        os.environ[f"{ENV_PREFIX}shutups_{key}"] = "1"
    return default

########################################################################################################################
########################################### KEEP THIS AT THE BOTTOM! ###################################################
########################################################################################################################

STARTUP_ENVVARS = {k:v for k,v in os.environ.items() if k.startswith(ENV_PREFIX+"_")}