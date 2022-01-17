import inspect
import logging
import os
import sys
from datetime import datetime
from functools import wraps
from os.path import join, dirname, basename, abspath

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))

from dotenv import load_dotenv
import click

from misc_util.telegram_notifier import telegram_notify
from misc_util.pretty_print import pretty_print as print

from derive_conceptualspace.util.desc_object import DescriptionList
from derive_conceptualspace.settings import (
    ALL_TRANSLATE_POLICY, ALL_QUANTIFICATION_MEASURE, ALL_EXTRACTION_METHOD, ALL_EMBED_ALGO, ALL_DCM_QUANT_MEASURE,
    ENV_PREFIX,
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
    rank_courses_saldirs as rank_courses_saldirs_base,
    show_data_info as show_data_info_base,
)
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader
from derive_conceptualspace.pipeline import cluster_loader
from derive_conceptualspace.pipeline import CustomContext

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
            ctx.pre_actualcommand_ops()
        nkw = {k: ctx.get_config(k) for k in kwargs.keys() if k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}} #only give the function those args that it lists
        nkw.update({k: ctx.obj[k] for k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}-nkw.keys()}) #this adds the OBJECTS, the line above the CONFs
        res = fn(ctx, *args, **nkw)
        if ctx.get_config("notify_telegram") and isinstance(ctx.command, click.Group) and not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)
        return res
    return wrapped


@click.group()
@click.option("--env-file", callback=lambda ctx, param, value: load_dotenv(value) if (param.name == "env_file" and value) else None,
              default=os.environ.get(ENV_PREFIX+"_"+"ENV_FILE"), type=click.Path(exists=True), is_eager=True,
              help="If you want to provide environment-variables using .env-files you can provide the path to a .env-file here.")
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
@click.option("--raw-descriptions-file", type=str, default=None)
@click.option("--languages-file", type=str, default=None)
@click.option("--title-languages-file", type=str, default=None)
@click_pass_add_context
def check_languages(ctx, raw_descriptions_file, languages_file, title_languages_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions")
    create_languages_file_base(languages_file, "languages", "Beschreibung", ctx.obj["json_persister"], raw_descriptions, ctx.obj["dataset_class"], declare_silent=True)
    create_languages_file_base(title_languages_file, "title_languages", "Name", ctx.obj["json_persister"], raw_descriptions, ctx.obj["dataset_class"], declare_silent=True)
    #no need to save, that's done inside the function.



@cli.command()
@click.option("--pp-components", type=str, default=None)
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click.option("--title-languages-file", type=str, default=None)
@click.option("--title-translations-file", type=str, default=None)
@click_pass_add_context
def translate_titles(ctx, pp_components, translate_policy, raw_descriptions_file, title_languages_file, title_translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions")
    translate_titles_base(raw_descriptions, pp_components, translate_policy, title_languages_file, title_translations_file, ctx.obj["json_persister"], ctx.obj["dataset_class"])
    #no need to save, that's done inside the function.


@cli.command()
@click.option("--translate-policy", type=click.Choice(ALL_TRANSLATE_POLICY, case_sensitive=False), default=None)
@click.option("--raw-descriptions-file", type=str, default=None)
@click.option("--languages-file", type=str, default=None)
@click.option("--translations-file", type=str, default=None)
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
    raw_descriptions = json_persister.load(raw_descriptions_file, "raw_descriptions")
    languages = json_persister.load(languages_file, "languages", loader=lambda langs: langs)
    try:
        title_languages = json_persister.load(title_languages_file, "title_languages", loader=lambda langs: langs)
    except FileNotFoundError:
        title_languages = languages
    if ctx.get_config("translate_policy") == "translate":
        translations = json_persister.load(translations_file, "translated_descriptions")
        title_translations = json_persister.load(title_translations_file, "translated_titles", loader=lambda **kw: kw["title_translations"])
        # TODO[e] depending on pp_compoments, title_languages etc may still allowed to be empty
    else:
        translations, title_translations = None, None
    descriptions, metainf = preprocess_descriptions_base(raw_descriptions, dataset_class, ctx.get_config("pp_components"), ctx.get_config("translate_policy"), languages, translations, title_languages, title_translations)
    json_persister.save("pp_descriptions.json", descriptions=descriptions)# , relevant_metainf=metainf) #TODO overhaul 16.01.2022: can I do wihtout relevant_metainf?!


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
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
    quant_dtm, dissim_mat, metainf = create_dissim_mat_base(pp_descriptions, ctx.get_config("quantification_measure"), ctx.get_config("verbose"))
    ctx.obj["json_persister"].save("dissim_mat.json", quant_dtm=quant_dtm, dissim_mat=dissim_mat)


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
    #TODO if not NGRAMS_IN_EMBEDDING and extraction_method in tfidf/ppmi, you have to re-extract, otherwise you won't get n-grams
    candidateterms, relevant_metainf = extract_candidateterms_base(ctx.obj["pp_descriptions"], ctx.get_config("extraction_method"), max_ngram, ctx.get_config("faster_keybert"), verbose=ctx.get_config("verbose"))
    ctx.obj["json_persister"].save("candidate_terms.json", candidateterms=candidateterms)


@prepare_candidateterms.command()
@click_pass_add_context
def postprocess_candidateterms(ctx, json_persister):
    ctx.obj["candidate_terms"] = json_persister.load(None, "candidate_terms")
    postprocessed_candidates = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.get_config("extraction_method"))
    json_persister.save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates)


@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=None)
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=None)
@click.option("--cands-use-ndocs-count/--no-cands-use-ndocs-count", default=None)
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_filtered_doc_cand_matrix(ctx, candidate_min_term_count, cands_use_ndocs_count):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12])
    # TODO Do I need pp_descriptions except for verbosity? -> if not, make loading silent
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates", loader = lambda **args: args["postprocessed_candidates"])
    filtered_dcm = create_filtered_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count,
                                                    dcm_quant_measure=ctx.get_config("dcm_quant_measure"), use_n_docs_count=cands_use_ndocs_count, verbose=ctx.get_config("verbose"))
    ctx.obj["json_persister"].save("filtered_dcm.json", doc_term_matrix=filtered_dcm) #, relevant_metainf={"candidate_min_term_count": candidate_min_term_count}


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
    assert ctx.obj["embedding"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your embedding has {ctx.obj["embedding"].embedding_.shape[0] } descriptions!'


@generate_conceptualspace.command()
@click_pass_add_context
def create_candidate_svm(ctx):
    ctx.obj["pp_descriptions"] = ctx.p.load(None, "pp_descriptions", loader=DescriptionList.from_json, silent=True)
    decision_planes, metrics = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["embedding"], ctx.obj["pp_descriptions"], verbose=ctx.get_config("verbose"))
    ctx.p.save("clusters.json", decision_planes=decision_planes, metrics=metrics)
    #TODO hier war dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION"))), krieg ich das wieder hin?


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