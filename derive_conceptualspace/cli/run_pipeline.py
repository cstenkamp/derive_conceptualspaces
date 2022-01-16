import warnings
import inspect
import logging
import sys
from datetime import datetime
from functools import wraps
from os.path import join, dirname, basename, abspath
import yaml

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))

from dotenv import load_dotenv
import click

from misc_util.telegram_notifier import telegram_notify
from misc_util.logutils import setup_logging
from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace import settings
from derive_conceptualspace.pipeline import init_context
from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.settings import (
    ALL_TRANSLATE_POLICY, ALL_QUANTIFICATION_MEASURE, ALL_EXTRACTION_METHOD, ALL_EMBED_ALGO, ALL_DCM_QUANT_MEASURE,
    ENV_PREFIX, NORMALIFY_PARAMS,
    get_setting, set_envvar, get_envvar, get_envvarname, standardize_config, standardize_config_name
)
from derive_conceptualspace.create_spaces.translate_descriptions import (
    full_translate_titles as translate_titles_base,
    full_translate_descriptions as translate_descriptions_base,
    create_languages_file as create_languages_file_base,
    # count_translations as count_translations_base
)
from derive_conceptualspace.extract_keywords.postprocess_candidates import (
    postprocess_candidateterms as postprocess_candidateterms_base,
)
from derive_conceptualspace.extract_keywords.keywords_main import (
    extract_candidateterms as extract_candidateterms_base,
    create_filtered_doc_cand_matrix as create_filtered_doc_cand_matrix_base,
)
from derive_conceptualspace.create_spaces.preprocess_descriptions import (
    preprocess_descriptions_full as preprocess_descriptions_base,
)
from derive_conceptualspace.create_spaces.spaces_main import (
    create_dissim_mat as create_dissim_mat_base,
)
from derive_conceptualspace.create_spaces.create_embedding import (
    create_embedding as create_embedding_base,
)
from derive_conceptualspace.semantic_directions.create_candidate_svm import (
    create_candidate_svms as create_candidate_svms_base
)
from derive_conceptualspace.unfinished_commands import (
    rank_courses_saldirs as rank_courses_saldirs_base
)
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader
from derive_conceptualspace.pipeline import print_settings, cluster_loader
from misc_util.object_wrapper import ObjectWrapper

logger = logging.getLogger(basename(__file__))
flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli helpers & main
#TODO putt the stuff from here in ../pipeline.py


# def incorporate_settings(source_dict, ctx, source):
#     """yes there is click's auto_envvar_prefix, but that messes with what I want to do (and doesn't work for arguments).
#        So this function overwrites ctx.params & ctx.obj from env-vars (if they have the correct prefix), and also SETS
#        these env-vars from the cmd-args such that they can be accessed using get_setting(). ALSO, we don't use the
#        ctx.auto_envvar_prefix which builds up the prefix hierachically for subcommands."""
#     for param, val in source_dict.items():
#         param, val = standardize_config(param, val)
#         ctx.set_config(param, val, source) #does basically ctx.obj[param] = val
#         # envvarname = env_prefix+"_"+param.upper().replace("-","_")
#         # # https://github.com/pallets/click/issues/714#issuecomment-651389598
#         # eagers = set(i.human_readable_name for i in ctx.command.params if i.is_eager)
#         # if (envvar := get_envvar(envvarname)) is not None and envvar != ctx.params[param] and param not in eagers:
#         #     print(f"The param {param} used to be *r*{ctx.params[param]}*r*, but is overwritten by an env-var to *b*{envvar}*b*")
#         #     ctx.params[param] = envvar
#         #     ctx.obj[param] = envvar
#         # else:
#         #     set_envvar(envvarname, ctx.obj[param])


class CustomContext(ObjectWrapper):
    def __init__(self, click_ctx):
        assert isinstance(click_ctx, click.Context)
        super(CustomContext, self).__init__(click_ctx)
        self.toset_configs = []
        self.used_configs = {}

    def set_config(self, key, val, source): #this is only a suggestion, it will only be finally set once it's accessed!
        key, val = standardize_config(key, val)
        self.toset_configs.append([key, val, source])

    def pre_actualcommand_ops(self):
        warnings.warn("pre_actualcommand_ops TODO overhaul 16.1.2022")
        # import derive_conceptualspace.settings
        # # ensure that those configs that can be set/overwritten in click are all added as config (it's important to know in which file they were first introduced)
        # for key, val in {k: v for k, v in ctx.obj.items() if "DEFAULT_" + k.upper() in derive_conceptualspace.settings.__dict__}.items():
        #     ctx.obj["json_persister"].add_config(key, val)
        # print_settings()

    def get_config(self, key):
        key = standardize_config_name(key)
        if key not in self.used_configs:
            conf_suggestions = [i[1:] for i in self.toset_configs if i[0] == key]
            assert len(conf_suggestions) > 0, "TODO what now?!"
            final_conf = min([i for i in conf_suggestions], key=lambda x: settings.CONF_PRIORITY.index(x[1]))
            if final_conf[1] == "defaults": #TODO overhaul 16.01.2022: does that mean I can get rid of get_setting?!
                if "DEFAULT_"+key not in settings.__dict__:
                    raise ValueError(f"You didn't provide a value for {key} and there is no default-value!")
                final_conf[0] = settings.__dict__["DEFAULT_"+key]
            self.used_configs[key] = final_conf[0]
        return self.used_configs[key]




def click_pass_add_context(fn):
    @click.pass_context
    @wraps(fn)
    def wrapped(ctx, *args, **kwargs):  #kwargs are cli-args, default-vals None
        if not isinstance(ctx, CustomContext): #ensure I always have my CustomContext
            if "actual_context" in ctx.obj:
                ctx = ctx.obj["actual_context"]
            else:
                ctx = CustomContext(ctx)
                ctx._wrapped.obj["actual_context"] = ctx
        eager_params = set(i.name for i in ctx.command.params if i.is_eager)
        click_args = {k: v for k, v in kwargs.items() if k not in eager_params}
        for param, val in click_args.items():
            ctx.set_config(param, val, "cmd_args" if val is not None else "defaults")
            #TODO what about env-vars, env-file and conf-file now? Did they already overwrite somethign?!
        if isinstance(ctx.command, click.Command) and not isinstance(ctx.command, click.Group):
            ctx.pre_actualcommand_ops()
        nkw = {k: ctx.get_config(k) for k in kwargs.keys() if k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}} #only give the function those args that it lists
        nkw.update({k: ctx.obj[k] for k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}-nkw.keys()}) #this adds the OBJECTS, the line above the CONFs
        res = fn(ctx, *args, **nkw)
        if ctx.get_config("notify_telegram") and isinstance(ctx.command, click.Group) and not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)
        return res
    return wrapped


def read_config(path):
    if path:
        with open(path, "r") as rfile:
            config = yaml.load(rfile, Loader=yaml.SafeLoader)
        config = {k: v if not isinstance(v, list) else v[0] for k,v in config.items()}
        #TODO statt v[0], wenn v eine liste ist und wenn ein cmd-arg bzw env-var einen wert hat der damit consistent ist, nimm das arg
        for k, v in config.items():
            set_envvar(get_envvarname(k), v)


@click.group()
#TODO even prior to evaluating this, save the old env-vars somewhere.
@click.option("--env-file", callback=lambda ctx, param, value: load_dotenv(value) if (param.name == "env_file" and value) else None,
              default=lambda: get_setting("ENV_FILE", default_none=True, fordefault=True), type=click.Path(exists=True), is_eager=True,
              help="If you want to provide environment-variables using .env-files you can provide the path to a .env-file here.")
@click.option("--conf-file", callback=lambda ctx, param, value: read_config(value) if (param.name == "conf_file" and value) else None,
              default=lambda: get_setting("CONF_FILE", fordefault=True, default_none=True), type=click.Path(exists=True), is_eager=True,
              help="You can also pass a yaml-file containing values for some of the settings")
@click.option("--base-dir", type=str, default=None)
@click.option("--dataset", type=str, default=None,
              help="The dataset you're solving here. Makes for the subfolder in base_dir where your data is stored and which of the classes in `load_data/dataset_specifics` will be used.")
@click.option("--verbose/--no-verbose", default=None)
@click.option("--debug/--no-debug", default=None, help=f"If True, many functions will only run on a few samples, such that everything should run really quickly. Default: {get_setting('DEBUG', silent=True)}")
@click.option("--log", type=str, default=None, help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
@click.option("--logfile", type=str, default=None, help="logfile to log to. If not set, it will be logged to standard stdout/stderr")
@click.option("--notify-telegram/--no-notify-telegram", default=None, help="If you want to get telegram-notified of start & end of the command")
@click_pass_add_context
def cli(ctx):
    """
    You can call this pipeline in many ways: With correct env-vars already set, with a provided `--env-file`, with a
    provided `--conf-file`, with command-line args (at the appropriate sub-command), or with default-values. If a multiple
    values for settings are given, precedence-order is command-line-args > env-vars (--env-file > pre-existing) > conf-file > dataset_class > defaults
    """
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    # setup_logging(ctx.obj["log"], ctx.obj["logfile"])  #TODO overhaul 16.01.2022 add this back
    init_context(ctx)


@cli.resultcallback()
def process_result(*args, **kwargs):
    """gets executed after the actual code. Prints time for bookkeeping"""
    print("Done at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare descriptions

# @cli.command()
# @click_pass_add_context
# def count_translations(ctx, ...):
#     return count_translations_base(ctx.obj["base_dir"], mds_basename=mds_basename, descriptions_basename=descriptions_basename)

@cli.command()
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--languages-file", type=str, default="languages.json")
@click_pass_add_context
def check_languages(ctx, raw_descriptions_file, languages_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    create_languages_file_base(languages_file, ctx.obj["json_persister"], raw_descriptions, ctx.obj["dataset_class"])
    #no need to save, that's done inside the function.



@cli.command()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", fordefault=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", fordefault=True))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--title-languages-file", type=str, default="title_languages.json")
@click.option("--title-translations-file", type=str, default="translated_titles.json")
@click_pass_add_context
def translate_titles(ctx, pp_components, translate_policy, raw_descriptions_file, title_languages_file, title_translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    translate_titles_base(raw_descriptions, pp_components, translate_policy, title_languages_file, title_translations_file, ctx.obj["json_persister"])
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", fordefault=True))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--languages-file", type=str, default="languages.json")
@click.option("--translations-file", type=str, default="translated_descriptions.json")
@click_pass_add_context
def translate_descriptions(ctx, translate_policy, raw_descriptions_file, languages_file, translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    translate_descriptions_base(raw_descriptions, translate_policy, languages_file, translations_file, ctx.obj["json_persister"])
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click.option("--languages-file", type=str, default=None)
@click.option("--translations-file", type=str, default=None)
@click.option("--title-languages-file", type=str, default=None)
@click.option("--title-translations-file", type=str, default=None)
@click.option("--max-ngram", type=int, default=None)
@click_pass_add_context
def preprocess_descriptions(ctx, json_persister, dataset_class, raw_descriptions_file, languages_file, translations_file, title_languages_file, title_translations_file):
    raw_descriptions = json_persister.load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    languages = json_persister.load(languages_file, "languages", ignore_params=["pp_components", "translate_policy"], loader=lambda langs: langs)
    try:
        title_languages = ctx.obj["json_persister"].load(title_languages_file, "title_languages", ignore_params=["pp_components", "translate_policy"], loader=lambda title_langs: title_langs)
    except FileNotFoundError:
        title_languages = languages
    if ctx.obj["translate_policy"] == "translate":
        translations = ctx.obj["json_persister"].load(translations_file, "translated_descriptions", ignore_params=["pp_components", "translate_policy"], force_overwrite=True, loader=lambda **kwargs: kwargs["translations"])
        title_translations = ctx.obj["json_persister"].load(title_translations_file, "translated_titles", ignore_params=["pp_components", "translate_policy"], force_overwrite=True, loader=lambda **kwargs: kwargs["title_translations"])
        # TODO[e] depending on pp_compoments, title_languages etc may still allowed to be empty
    else:
        translations, title_translations = None, None
    descriptions, metainf = preprocess_descriptions_base(raw_descriptions, dataset_class, ctx.obj["pp_components"], ctx.obj["translate_policy"], languages, translations, title_languages, title_translations)
    ctx.obj["json_persister"].save("pp_descriptions.json", descriptions=descriptions, relevant_metainf=metainf)


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", fordefault=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", fordefault=True))
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=lambda: get_setting("QUANTIFICATION_MEASURE", fordefault=True))
@click_pass_add_context
def create_spaces(ctx):
    """[group] CLI base to create the spaces from texts"""
    pass


@create_spaces.command()
@click_pass_add_context
def create_dissim_mat(ctx):
    pp_descriptions = ctx.obj["json_persister"].load(None, "pp_descriptions", ignore_params=["quantification_measure"], loader=DescriptionList.from_json)
    quant_dtm, dissim_mat, metainf = create_dissim_mat_base(pp_descriptions, ctx.obj["quantification_measure"], ctx.obj["verbose"])
    ctx.obj["json_persister"].save("dissim_mat.json", quant_dtm=quant_dtm, dissim_mat=dissim_mat, relevant_metainf=metainf)


@create_spaces.command()
@click.option("--embed-dimensions", type=int, default=lambda: get_setting("EMBED_DIMENSIONS", fordefault=True))
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=lambda: get_setting("EMBED_ALGO", fordefault=True))
@click_pass_add_context
def create_embedding(ctx):
    dissim_mat = ctx.obj["json_persister"].load(None, "dissim_mat", ignore_params=["embed_dimensions"], loader=dtm_dissimmat_loader)
    pp_descriptions = ctx.obj["json_persister"].load(None, "pp_descriptions", ignore_params=["quantification_measure"], loader=DescriptionList.from_json, silent=True) \
                      if ctx.obj["verbose"] else None
    embedding = create_embedding_base(dissim_mat, ctx.obj["embed_dimensions"], ctx.obj["embed_algo"], ctx.obj["verbose"], pp_descriptions=pp_descriptions)
    ctx.obj["json_persister"].save("embedding.json", embedding=embedding)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", fordefault=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", fordefault=True))
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=lambda: get_setting("EXTRACTION_METHOD", fordefault=True))
@click_pass_add_context
def prepare_candidateterms(ctx):
    """[group] CLI base to extract candidate-terms from texts"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "pp_descriptions", loader=DescriptionList.from_json)


@prepare_candidateterms.command()
@click_pass_add_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=lambda: get_setting("FASTER_KEYBERT", fordefault=True))
@click.option("--max-ngram", default=lambda: get_setting("MAX_NGRAM", fordefault=True))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms(ctx, max_ngram):
    #TODO if not NGRAMS_IN_EMBEDDING and extraction_method in tfidf/ppmi, you have to re-extract, otherwise you won't get n-grams
    candidateterms, relevant_metainf = extract_candidateterms_base(ctx.obj["pp_descriptions"], ctx.obj["extraction_method"], max_ngram, ctx.obj["faster_keybert"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("candidate_terms.json", candidateterms=candidateterms, relevant_metainf=relevant_metainf)


@prepare_candidateterms.command()
@click_pass_add_context
def postprocess_candidateterms(ctx):
    ctx.obj["candidate_terms"] = ctx.obj["json_persister"].load(None, "candidate_terms")
    postprocessed_candidates = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.obj["extraction_method"])
    ctx.obj["json_persister"].save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates)


@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", fordefault=True))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE", fordefault=True))
@click.option("--use-ndocs-count/--no-use-ndocs-count", default=lambda: get_setting("CANDS_USE_NDOCS_COUNT", fordefault=True))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_filtered_doc_cand_matrix(ctx, candidate_min_term_count, use_ndocs_count):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12])
    # TODO Do I need pp_descriptions except for verbosity? -> if not, make loading silent
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates", loader = lambda **args: args["postprocessed_candidates"])
    filtered_dcm = create_filtered_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count,
                                                    dcm_quant_measure=ctx.obj["dcm_quant_measure"], use_n_docs_count=use_ndocs_count, verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("filtered_dcm.json", relevant_metainf={"candidate_min_term_count": candidate_min_term_count}, doc_term_matrix=filtered_dcm)


########################################################################################################################
########################################################################################################################
########################################################################################################################
# generate-conceptualspace group


@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS", fordefault=True))
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=lambda: get_setting("TRANSLATE_POLICY", fordefault=True))
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=lambda: get_setting("QUANTIFICATION_MEASURE", fordefault=True))
@click.option("--embed-dimensions", type=int, default=lambda: get_setting("EMBED_DIMENSIONS", fordefault=True))
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=lambda: get_setting("EMBED_ALGO", fordefault=True))
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=lambda: get_setting("EXTRACTION_METHOD", fordefault=True))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE", fordefault=True))
@click_pass_add_context
def generate_conceptualspace(ctx):
    """[group] CLI base to create the actual conceptual spaces"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "pp_descriptions", loader=DescriptionList.from_json, ignore_params=["quantification_measure", "embed_dimensions"])
    ctx.obj["filtered_dcm"] = ctx.obj["json_persister"].load(None, "filtered_dcm", loader=dtm_loader, ignore_params=["quantification_measure", "embed_dimensions"])
    ctx.obj["embedding"] = ctx.obj["json_persister"].load(None, "embedding", ignore_params=["extraction_method", "dcm_quant_measure"], loader=lambda **args: args["embedding"])
    assert ctx.obj["embedding"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your embedding has {ctx.obj["embedding"].embedding_.shape[0] } descriptions!'


@generate_conceptualspace.command()
@click_pass_add_context
def create_candidate_svm(ctx):
    #TODO here, the descriptions are also only needed for visualization
    decision_planes, metrics = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["embedding"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("clusters.json", decision_planes=decision_planes, metrics=metrics)
    #TODO hier war dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION"))), krieg ich das wieder hin?

#TODO I can do something like autodetect_relevant_params und metainf im json_persister


@generate_conceptualspace.command()
@click_pass_add_context
def show_data_info(ctx):
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters")
    show_data_info_base(ctx)


@generate_conceptualspace.command()
# @click.option("--prim-lambda", type=float, default=lambda: get_setting("PRIM_LAMBDA", fordefault=True))
# @click.option("--sec-lambda", type=float, default=lambda: get_setting("SEC_LAMBDA", fordefault=True))
@click_pass_add_context
def rank_courses_saldirs(ctx):
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters", loader=cluster_loader)
    rank_courses_saldirs_base(ctx.obj["pp_descriptions"], ctx.obj["embedding"], ctx.obj["clusters"], ctx.obj["filtered_dcm"])
    #relevant_metainf={"prim_lambda": ctx.obj["prim_lambda"], "sec_lambda": ctx.obj["sec_lambda"]}


@generate_conceptualspace.command()
@click_pass_add_context
def run_lsi_gensim(ctx):
    run_lsi_gensim_base(ctx.obj["pp_descriptions"], ctx.obj["filtered_dcm"], verbose=ctx.obj["verbose"])


@generate_conceptualspace.command()
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def run_lsi(ctx):
    run_lsi_base(ctx.obj["pp_descriptions"], ctx.obj["filtered_dcm"], ctx.obj["verbose"])


#
#
#
#
# def rank_courses_saldirs_base(descriptions, clusters):
#     assert hasattr(descriptions[0], "embedding")
#     print()



########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(auto_envvar_prefix=ENV_PREFIX, obj={})



