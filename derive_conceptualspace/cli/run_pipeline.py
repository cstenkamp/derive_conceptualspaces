import inspect
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from os.path import join, dirname, basename, abspath, isfile

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))


from dotenv import load_dotenv
import click

from misc_util.telegram_notifier import telegram_notify
from misc_util.pretty_print import pretty_print as print
from misc_util.logutils import setup_logging


from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.settings import (
    ALL_TRANSLATE_POLICY, ALL_QUANTIFICATION_MEASURE, ALL_EXTRACTION_METHOD, ALL_EMBED_ALGO, ALL_DCM_QUANT_MEASURE,
    ENV_PREFIX,
    IS_INTERACTIVE,
    standardize_config_name,
)
from derive_conceptualspace.create_spaces.translate_descriptions import (
    full_translate_column as translate_column_base,
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
    create_candidate_svms as create_candidate_svms_base,
    select_salient_terms as select_salient_terms_base,
)
from derive_conceptualspace.unfinished_commands import (
    rank_saldirs as rank_saldirs_base,
    show_data_info as show_data_info_base,
)
from derive_conceptualspace.evaluate.shallow_trees import (
    classify_shallowtree as classify_shallowtree_base,
    classify_shallowtree_multi as classify_shallowtree_multi_base,
)
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader
from derive_conceptualspace.util.interruptible_funcs import InterruptibleLoad
from derive_conceptualspace.pipeline import featureaxes_loader, CustomContext, load_lang_translate_files, apply_dotenv_vars, cluster_loader

logger = logging.getLogger(basename(__file__))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli helpers & main

#TODO overhaul 16.01.2022: I used to set the env-vars from the configs, did that make sense or is it fine to be rid of it?

def click_pass_add_context(fn):
    @click.pass_context
    @wraps(fn)
    def wrapped(ctx, *args, **kwargs):  #kwargs are cli-args, default-vals None
        if not isinstance(ctx, CustomContext): #ensure I always have my CustomContext
            if "actual_context" in ctx.obj:
                ctx = ctx.obj["actual_context"].reattach(ctx)
            else:
                ctx = CustomContext(ctx)
                ctx._wrapped.obj["actual_context"] = ctx
        eager_params = set(i.name for i in ctx.command.params if i.is_eager)
        click_args = {k: v for k, v in kwargs.items() if k not in eager_params}
        for param, val in click_args.items():
            ctx.set_config(param, val, "cmd_args" if val is not None else "defaults")
        if isinstance(ctx.command, click.Command) and not isinstance(ctx.command, click.Group):
            ctx.pre_actualcommand_ops(fn)
        nkw = {k: ctx.get_config(k) for k in kwargs.keys() if k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}} #only give the function those args that it lists
        nkw.update({k: ctx.obj[k] for k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}-nkw.keys()}) #this adds the OBJECTS, the line above the CONFs
        res = fn(ctx, *args, **nkw)
        if ctx.get_config("notify_telegram") and isinstance(ctx.command, click.Group) and not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)
        return res
    return wrapped

def load_envvfile(ctx, param, value):
    apply_dotenv_vars()
    if param.name == "env_file":
        if value is None and os.getenv(ENV_PREFIX+"_"+standardize_config_name(param.name)):
            value = os.getenv(ENV_PREFIX+"_"+standardize_config_name(param.name)) #default-value, but evaluated after reading the SELECT_ENV_FILE
        if value:
            if not isfile(value):
                raise click.exceptions.FileError(value)
            load_dotenv(value)

@click.group()
@click.option("--env-file", callback=load_envvfile, type=click.Path(), is_eager=True, help="If you want to provide environment-variables using .env-files you can provide the path to a .env-file here.")
@click.option("--conf-file", default=None, type=click.Path(exists=True), help="You can also pass a yaml-file containing values for some of the settings")
@click.option("--base-dir", type=click.Path(exists=True), default=None)
@click.option("--dataset", type=str, default=None, help="The dataset you're solving here. Makes for the subfolder in base_dir where your data is stored and which of the classes in `load_data/dataset_specifics` will be used.")
@click.option("--verbose/--no-verbose", default=None)
@click.option("--debug/--no-debug", default=None, help=f"If True, many functions will only run on a few samples, such that everything should run really quickly.")
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
    setup_logging(ctx.get_config("log"), ctx.get_config("logfile"))
    ctx.init_context() #after this point, no new env-vars should be set anymore (are not considered)

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
@click.option("--pp-components", type=str, default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click_pass_add_context
def check_languages(ctx, raw_descriptions_file, pp_components):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions")
    create_languages_file_base(raw_descriptions, ["description", "title", "subtitle"], ctx.obj["json_persister"], ctx.obj["dataset_class"], declare_silent=True, pp_components=pp_components)
    #no need to save, that's done inside the function.




# @cli.command()
# @click.option("--pp-components", type=str, default=None)
# @click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
# @click.option("--raw-descriptions-file", type=str, default=None)
# @click_pass_add_context
# def translate_titles(ctx, pp_components, translate_policy, raw_descriptions_file, title_languages_file, title_translations_file):
#     raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions")
#     translate_titles_base(raw_descriptions, "titles", title_languages_file, title_translations_file, ctx.obj["json_persister"], ctx.obj["dataset_class"], pp_components=ctx.get_config("pp_components"), translate_policy=translate_policy)
#     #no need to save, that's done inside the function.


@cli.command()
@click.argument("column", type=click.Choice(["description", "title", "subtitle"], case_sensitive=True))
@click.option("--language", type=str, default=None)
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click_pass_add_context
def translate_descriptions(ctx, translate_policy, raw_descriptions_file, language, pp_components):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions")
    translate_column_base(raw_descriptions, translate_policy, language, ctx.get_config("column"), ctx.obj["json_persister"], ctx.obj["dataset_class"], pp_components=pp_components)
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--language", type=str, default=None)
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click_pass_add_context
def preprocess_descriptions(ctx, json_persister, dataset_class, raw_descriptions_file, pp_components, language):
    raw_descriptions = json_persister.load(raw_descriptions_file, "raw_descriptions")
    if ctx.has_config("all_descriptions_lang") and ctx.get_config("all_descriptions_lang"):
        languages, translations = ctx.get_config("all_descriptions_lang"), None
    else:
        languages, translations = load_lang_translate_files(ctx, json_persister, pp_components)
    descriptions, metainf = preprocess_descriptions_base(raw_descriptions, dataset_class, pp_components, language, ctx.get_config("translate_policy"), languages, translations)
    json_persister.save("pp_descriptions.json", descriptions=descriptions, metainf=metainf)


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
@click.option("--language", type=str, default=None)
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=None)
@click_pass_add_context
def create_spaces(ctx, json_persister):
    """[group] CLI base to create the spaces from texts"""
    pass


@create_spaces.command()
@click_pass_add_context
def create_dissim_mat(ctx, json_persister):
    pp_descriptions = json_persister.load(None, "pp_descriptions", loader=DescriptionList.from_json)
    with InterruptibleLoad(ctx, "dissim_mat.json", metainf_ignorevarnames=["is_dissim"], loader=(lambda **kw: [kw["dissim_mat"]])) as mgr:
        quant_dtm, dissim_mat, metainf = create_dissim_mat_base(pp_descriptions, ctx.get_config("quantification_measure"), ctx.get_config("verbose"), **mgr.kwargs)
    mgr.save(quant_dtm=quant_dtm, dissim_mat=dissim_mat, metainf=metainf)


@create_spaces.command()
@click.option("--embed-dimensions", type=int, default=None)
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=None)
@click_pass_add_context
def create_embedding(ctx, json_persister):
    dissim_mat = json_persister.load(None, "dissim_mat", loader=dtm_dissimmat_loader)
    pp_descriptions = json_persister.load(None, "pp_descriptions", loader=DescriptionList.from_json, silent=True) if ctx.get_config("verbose") else None
    embedding = create_embedding_base(dissim_mat, ctx.get_config("embed_dimensions"), ctx.get_config("embed_algo"), ctx.get_config("verbose"), pp_descriptions=pp_descriptions)
    json_persister.save("embedding.json", embedding=embedding)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group


@cli.group()
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=None)
@click_pass_add_context
def prepare_candidateterms(ctx, json_persister):
    """[group] CLI base to extract candidate-terms from texts"""
    ctx.obj["pp_descriptions"] = json_persister.load(None, "pp_descriptions", loader=DescriptionList.from_json)


@prepare_candidateterms.command()
@click_pass_add_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=None)
@click.option("--max-ngram", default=None)
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms(ctx, max_ngram):
    with InterruptibleLoad(ctx, "candidate_terms.json", metainf_countervarnames=["n_candidateterms", "n_immediateworking", "n_fixed", "n_errs"]) as mgr:
        candidate_terms, metainf = extract_candidateterms_base(ctx.obj["pp_descriptions"], ctx.get_config("extraction_method"), max_ngram, verbose=ctx.get_config("verbose"), **mgr.kwargs)
    mgr.save(candidate_terms=candidate_terms, metainf=metainf)


@prepare_candidateterms.command()
@click_pass_add_context
def postprocess_candidateterms(ctx, json_persister):
    ctx.obj["candidate_terms"] = json_persister.load(None, "candidate_terms", loader=lambda **args: args["candidate_terms"])
    postprocessed_candidates, changeds = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.get_config("extraction_method"))
    json_persister.save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates, changeds=changeds)


@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=None)
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=None)
@click.option("--cands-use-ndocs-count/--no-cands-use-ndocs-count", default=None)
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_filtered_doc_cand_matrix(ctx, candidate_min_term_count, cands_use_ndocs_count):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12])
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates")
    filtered_dcm = create_filtered_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count,
                                                    dcm_quant_measure=ctx.get_config("dcm_quant_measure"), use_n_docs_count=cands_use_ndocs_count, verbose=ctx.get_config("verbose"))
    ctx.obj["json_persister"].save("filtered_dcm.json", doc_term_matrix=filtered_dcm)


########################################################################################################################
########################################################################################################################
########################################################################################################################
# generate-conceptualspace group


@cli.group()
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--quantification-measure", type=click.Choice(ALL_QUANTIFICATION_MEASURE, case_sensitive=False), default=None)
@click.option("--embed-dimensions", type=int, default=None)
@click.option("--embed-algo", type=click.Choice(ALL_EMBED_ALGO, case_sensitive=False), default=None)
@click.option("--extraction-method", type=click.Choice(ALL_EXTRACTION_METHOD, case_sensitive=False), default=None)
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=None)
@click_pass_add_context
def generate_conceptualspace(ctx, json_persister):
    """[group] CLI base to create the actual conceptual spaces"""
    ctx.obj["filtered_dcm"] = json_persister.load(None, "filtered_dcm", loader=dtm_loader)
    ctx.obj["embedding"] = json_persister.load(None, "embedding", loader=lambda **args: args["embedding"])
    if not ctx.get_config("DEBUG"):
        assert ctx.obj["embedding"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your embedding has {ctx.obj["embedding"].embedding_.shape[0] } descriptions!'


@generate_conceptualspace.command()
@click_pass_add_context
def create_candidate_svm(ctx):
    ctx.obj["pp_descriptions"] = ctx.p.load(None, "pp_descriptions", loader=DescriptionList.from_json, silent=True)
    with InterruptibleLoad(ctx, "featureaxes.json", loader=lambda x:x) as mgr:
        quants_s, decision_planes, metrics, metainf = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["embedding"], ctx.obj["pp_descriptions"], verbose=ctx.get_config("verbose"), **mgr.kwargs)
    mgr.save(quants_s=quants_s, decision_planes=decision_planes, metrics=metrics, metainf=metainf)


@generate_conceptualspace.command()
@click.option("--prim-lambda", type=float)
@click.option("--sec-lambda", type=float)
@click.option("--classifier-succmetric", type=str)
@click_pass_add_context
def cluster_candidates(ctx):
    #TODO decide on ONE for the func! "cluster_candidates", "cluster_feature_axes", "select_salient_terms"
    ctx.obj["featureaxes"] = ctx.obj["json_persister"].load(None, "featureaxes", loader=featureaxes_loader)
    decision_planes, metrics = ctx.obj["featureaxes"].values()
    clusters, directions = select_salient_terms_base(metrics, decision_planes, ctx.obj["filtered_dcm"], ctx.obj["embedding"],
                                                     prim_lambda=ctx.get_config("prim_lambda"), sec_lambda=ctx.get_config("sec_lambda"),
                                                     metricname=ctx.get_config("classifier_succmetric"), verbose=ctx.get_config("verbose"))
    ctx.obj["json_persister"].save("clusters.json", clusters=clusters, directions=directions)
    #TODO this needs WAY MORE Parameters & ways-how-to-do-it, see bottom of select_salient_terms_base
    #TODO im JsonPersister abbilden können wenn bestimmte dependencies or configs nur für gewisse parameter relevant sind (filtered_dcm & embedding brauch ich nur bei reclassify)

@generate_conceptualspace.command()
@click_pass_add_context
def show_data_info(ctx):
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters") #TODO use LAST_RESULT (and make it ONE THING used also in Snakefile and args_from_filename)
    show_data_info_base(ctx)
    print()


@generate_conceptualspace.command()
# @click.option("--one-vs-rest/--no-one-vs-rest", type=bool)
# @click.option("--dt-depth", type=int)
# @click.option("--test-percentage-crossval", type=float)
# @click.option("--classes", type=str)
@click_pass_add_context
def decision_trees(ctx):
    descriptions = ctx.p.load(None, "pp_descriptions", loader=DescriptionList.from_json)
    clusters = ctx.p.load(None, "clusters", loader=cluster_loader)
    # classify_shallowtree_base(clusters, ctx.obj["embedding"].embedding_, descriptions,
    #                           one_vs_rest=ctx.get_config("DT_ONE_VS_REST"), dt_depth=ctx.get_config("DT_DEPTH"),
    #                           test_percentage_crossval=ctx.get_config("TEST_PERCENTAGE_CROSSVAL"), classes=ctx.get_config("CLASSIFY_CLASSES"), verbose=ctx.get_config("VERBOSE"))
    classify_shallowtree_multi_base(clusters, ctx.obj["embedding"].embedding_, descriptions, verbose=ctx.get_config("VERBOSE"))



@generate_conceptualspace.command()
@click_pass_add_context
def rank_saldirs(ctx):
    ctx.obj["pp_descriptions"] = ctx.p.load(None, "pp_descriptions", loader=DescriptionList.from_json, silent=True) #TODO really silent?
    ctx.obj["featureaxes"] = ctx.p.load(None, "featureaxes", loader=featureaxes_loader)
    ctx.obj["clusters"] = ctx.p.load(None, "clusters")
    #TODO this should rather contain the code from run_pipeline.decision_trees
    rank_saldirs_base(ctx.obj["pp_descriptions"], ctx.obj["embedding"], ctx.obj["featureaxes"], ctx.obj["filtered_dcm"],
                      prim_lambda=ctx.get_config("prim_lambda"), sec_lambda=ctx.get_config("sec_lambda"), metricname=ctx.get_config("classifier_succmetric"))


@cli.command()
@click_pass_add_context
def list_paramcombis(ctx):
    # TODO get rid of this entirely.
    # TODO this should ONLY consider command-line-args as config to compare the candidates to
    candidates = [join(path, name)[len(ctx.p.in_dir):] for path, subdirs, files in os.walk(join(ctx.p.in_dir, "")) for
                  name in files if name.startswith("featureaxes.json")] #TODO use LAST_RESULT
    candidates = [i for i in candidates if i.startswith(ctx.p.get_subdir({i: ctx.get_config(i) for i in ["DEBUG", "DATASET", "LANGUAGE"]})[0])]
    for cand in candidates:
        print(cand)




@generate_conceptualspace.command()
@click_pass_add_context
def run_lsi_gensim(ctx):
    run_lsi_gensim_base(ctx.obj["pp_descriptions"], ctx.obj["filtered_dcm"], verbose=ctx.get_config("verbose"))


@generate_conceptualspace.command()
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def run_lsi(ctx):
    run_lsi_base(ctx.obj["pp_descriptions"], ctx.obj["filtered_dcm"], ctx.get_config("verbose"))



########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(obj={})