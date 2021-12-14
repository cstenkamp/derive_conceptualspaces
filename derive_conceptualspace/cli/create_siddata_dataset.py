"""The purpose of this file is to create a dataset from the Siddata-data that looks like the three datasets used in [DESC15],
available at http://www.cs.cf.ac.uk/semanticspaces/. Meaning: MDS, ..."""

#TODO make (snakemake?) Pipeline that runs start to finish and creates the complete directory
import os
from os.path import join, dirname, basename, abspath
import random
import logging
from datetime import datetime
from time import sleep

import click

import sys

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))

from misc_util.telegram_notifier import telegram_notify
from misc_util.logutils import setup_logging
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.settings import (
    RANDOM_SEED,
    MDS_DEFAULT_BASENAME,
    DEFAULT_TRANSLATE_POLICY,
    CANDIDATETERM_MIN_OCCURSIN_DOCS,
    DEFAULT_DEBUG,
    ENV_PREFIX,
    get_setting,
)
from derive_conceptualspace.util.jsonloadstore import json_dump, Struct, json_load
from derive_conceptualspace.create_spaces.translate_descriptions import (
    translate_descriptions as translate_descriptions_base,
    count_translations as count_translations_base
)
from derive_conceptualspace.extract_keywords.keywords_main import (
    extract_candidateterms_keybert as extract_candidateterms_keybert_base,
    postprocess_candidateterms as postprocess_candidateterms_base,
    create_doc_cand_matrix as create_doc_cand_matrix_base,
    filter_keyphrases as filter_keyphrases_base,
    extract_candidateterms_keybert_preprocessed
)
from derive_conceptualspace.create_spaces.spaces_main import (
    load_translate_mds,
    preprocess_descriptions_full,
    create_dissim_mat as create_dissim_mat_base,
    create_mds_json as create_mds_json_base,
    load_preprocessed_descriptions
)
from derive_conceptualspace.semantic_directions.create_candidate_svm import (
    create_candidate_svms as create_candidate_svms_base
)
from derive_conceptualspace.util.dtm_object import DocTermMatrix

logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli main

@click.group()
@click.argument("base-dir", type=str)
@click.option("--verbose/--no-verbose", default=True, help="default: True")
@click.option("--debug/--no-debug", default=DEFAULT_DEBUG, help=f"If True, many functions will only run on a few samples, such that everything should run really quickly. Default: {DEFAULT_DEBUG}")
@click.option(
    "--log",
    type=str,
    default="INFO",
    help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]",
)
@click.option(
    "--logfile",
    type=str,
    default="",
    help="logfile to log to. If not set, it will be logged to standard stdout/stderr",
)
@click.option("--notify-telegram/--no-notify-telegram", default=False, help="If you want to get telegram-notified of start & end of the command")
@click.pass_context
def cli(ctx, base_dir, verbose=False, debug=DEFAULT_DEBUG, log="INFO", logfile=None, notify_telegram=False):
    for param, val in ctx.params.items():
        ctx.obj[param] = val
        envvarname = ctx.auto_envvar_prefix+"_"+param.upper().replace("-","_")
        if (envvar := os.environ.get(envvarname)): #https://github.com/pallets/click/issues/714#issuecomment-651389598
            ctx.params[param] = envvar
            ctx.obj[param] = envvar
        else:
            if isinstance(ctx.obj[param], bool):
                if ctx.obj[param]:
                    os.environ[envvarname] = "1"
                else:
                    os.environ[envvarname+"_FALSE"] = "1"
            else:
                os.environ[envvarname] = ctx.obj[param]
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    if debug: print("debug is active!")
    setup_logging(ctx.obj["log"], ctx.obj["logfile"])
    random.seed(RANDOM_SEED)
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)



@cli.resultcallback()
def process_result(*args, **kwargs):
    """gets executed after the actual code. Prints time for bookkeeping"""
    print("Done at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))


########################################################################################################################
########################################################################################################################
########################################################################################################################
#translation-commands

@cli.command()
@click.option("--mds-basename", type=str, default=MDS_DEFAULT_BASENAME)
@click.pass_context
def translate_descriptions(ctx, mds_basename=MDS_DEFAULT_BASENAME):
    return translate_descriptions_base(ctx.obj["base_dir"], ctx.obj["mds_basename"])


@cli.command()
@click.option("--mds-basename", type=str, default=None)
@click.option("--descriptions-basename", type=str, default=None)
@click.pass_context
def count_translations(ctx, mds_basename=None, descriptions_basename=None):
    return count_translations_base(ctx.obj["base_dir"], mds_basename=mds_basename, descriptions_basename=descriptions_basename)


########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.argument("mds-dimensions", type=int)
@click.option("--translate-policy", type=str, default=DEFAULT_TRANSLATE_POLICY)
@click.pass_context
def create_spaces(ctx, mds_dimensions, translate_policy=DEFAULT_TRANSLATE_POLICY):
    """[group] CLI base to create the spaces from texts"""
    ctx.obj["mds_dimensions"] = mds_dimensions
    ctx.obj["translate_policy"] = translate_policy
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)


@create_spaces.command()
@click.option("--sent-tokenize/--no-sent-tokenize", default=True)
@click.option("--convert-lower/--no-convert-lower", default=True)
@click.option("--remove-stopwords/--no-remove-stopwords", default=True)
@click.option("--lemmatize/--no-lemmatize", default=True)
@click.option("--remove-diacritics/--no-remove-diacritics", default=True)
@click.option("--remove-punctuation/--no-remove-punctuation", default=True)
@click.pass_context
def preprocess_descriptions(ctx, **pp_components):
    vocab, descriptions = preprocess_descriptions_full(ctx.obj["base_dir"], ctx.obj["translate_policy"], pp_components)
    descriptions = [Struct(**desc.__dict__) for desc in descriptions]
    json_dump({"vocab": vocab, "descriptions": descriptions, "pp_components": pp_components}, join(ctx.obj["base_dir"], "preprocessed_descriptions.json"))


@create_spaces.command()
@click.argument("pp-descriptions-filename")
@click.argument("quantification-measure")
@click.pass_context
def create_dissim_mat(ctx, quantification_measure, pp_descriptions_filename):
    quantification, dissim_mat, pp_components = create_dissim_mat_base(ctx.obj["base_dir"], pp_descriptions_filename, quantification_measure)
    quantification = Struct(**{k:v for k,v in quantification.__dict__.items() if not k.startswith("_") and k not in ["csr_matrix", "doc_freqs", "reverse_term_dict"]})
    json_dump({"quantification": quantification, "dissim_mat": dissim_mat, "pp_components": pp_components, "quant_measure": quantification_measure},
              join(ctx.obj["base_dir"], f"dissim_matrix_{quantification_measure}.json"))

#TODO die info über translate-policy, pp_components etc muss durchgehend durchgeschleift werden, dass muss auch im MDS-JSON noch da sein... außerdem
# wäre es smart wenn im MDS nen Hash der das Descriptions-File eindeutig referenziert dabei ist damit ich weiß dass das zusammengeört

@create_spaces.command()
@click.argument("dissim-mat-filename")
@click.pass_context
def create_mds_json(ctx, dissim_mat_filename):
    mds = create_mds_json_base(ctx.obj["base_dir"], dissim_mat_filename, ctx.obj["mds_dimensions"])
    mds["mds"] = Struct(**mds["mds"].__dict__) #let's return the dict of the MDS such that we can load it from json and its equal
    json_dump(mds, join(ctx.obj["base_dir"], dissim_mat_filename.replace("dissim_matrix", f"mds_{ctx.obj['mds_dimensions']}d")))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

# @cli.group()
# @click.option("--mds-basename", type=str, default=MDS_DEFAULT_BASENAME)
# @click.option("--ndm-file", type=click.Path(exists=True), default=None)
# @click.option("--translate-policy", type=str, default=DEFAULT_TRANSLATE_POLICY)
# @click.pass_context
# def prepare_candidateterms(ctx, mds_basename=MDS_DEFAULT_BASENAME, ndm_file=None, translate_policy=DEFAULT_TRANSLATE_POLICY):
#     """[group] CLI base to create candidate-terms (everything in here needs the mds_obj)"""
#     if not ndm_file:
#         ndm_file = next(i for i in os.listdir(ctx.obj["base_dir"]) if i.startswith(mds_basename) and i.endswith(".json"))
#         load_basedir = ctx.obj["base_dir"]
#     else:
#         load_basedir = dirname(ndm_file)
#         ndm_file = basename(ndm_file)
#     ctx.obj["ndm_file"] = ndm_file
#     ctx.obj["mds_obj"] = load_translate_mds(load_basedir, ndm_file, translate_policy=translate_policy)
#     ctx.obj["translate_policy"] = translate_policy
#     if ctx.obj["notify_telegram"] == True:
#         if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
#             ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)

@cli.group()
@click.argument("pp-descriptions-filename")
@click.pass_context
def prepare_candidateterms(ctx, pp_descriptions_filename):
    """[group] CLI base to create candidate-terms"""
    ctx.obj["pp_descriptions_filename"] = pp_descriptions_filename
    ctx.obj["vocab"], ctx.obj["descriptions"], ctx.obj["pp_components"] = load_preprocessed_descriptions(join(ctx.obj["base_dir"], pp_descriptions_filename))
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)


@prepare_candidateterms.command()
@click.pass_context
def test_command(ctx):
    print(ctx.obj)
    sleep(2)


@prepare_candidateterms.command()
@click.pass_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=False)
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert(ctx, faster_keybert=False):
    candidateterms, extractor, (n_immediateworking_ges, n_fixed_ges, n_errs_ges) = extract_candidateterms_keybert_base(ctx.obj["vocab"], ctx.obj["descriptions"], faster_keybert)
    print(f"Immediately working: {n_immediateworking_ges}")
    print(f"Fixed: {n_fixed_ges}")
    print(f"Errors: {n_errs_ges}")
    json_dump({"model": extractor.model_name, "candidate_terms": [list(i) for i in candidateterms], "pp_txt_for_cands": False}, join(ctx.obj["base_dir"], "candidate_terms.json"))
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=False)
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert_2(ctx, faster_keybert=False):
    candidates, model_name = extract_candidateterms_keybert_preprocessed(ctx.obj["vocab"], ctx.obj["descriptions"], faster_keybert)
    json_dump({**{"model": model_name, "candidate_terms":candidates, "pp_txt_for_cands": True}, **{i: ctx.obj[i] for i in ["pp_components", "pp_descriptions_filename"]}}, join(ctx.obj["base_dir"], "candidate_terms.json"))


@prepare_candidateterms.command()
@click.option("--postfix", type=str, default="_postprocessed")
@click.pass_context
def postprocess_candidateterms(ctx, postfix="_postprocessed"):
    model, candidate_terms = postprocess_candidateterms_base(ctx.obj["base_dir"], ctx.obj["descriptions"])
    candidate_terms["postprocessed"] = True
    json_dump(candidate_terms, join(ctx.obj["base_dir"], f"candidate_terms{postfix}.json"))
    print(f"Saved the post-processed model under candidate_terms{postfix}.json!")
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.option("--json-filename", type=str, default="candidate_terms_postprocessed.json")
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_doc_cand_matrix(ctx, json_filename="candidate_terms_postprocessed.json"):
    doc_term_matrix = create_doc_cand_matrix_base(ctx.obj["base_dir"], ctx.obj["descriptions"], json_filename, verbose=ctx.obj["verbose"])
    json_dump({"all_terms": doc_term_matrix.all_terms, "doc_term_matrix": doc_term_matrix.dtm}, join(ctx.obj["base_dir"], "doc_cand_matrix.json"))
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.option("--min-term-count", type=int, default=CANDIDATETERM_MIN_OCCURSIN_DOCS)
@click.option("--matrix-val", type=click.Choice(["count", "binary", "tf-idf"], case_sensitive=False), default="count")
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def filter_keyphrases(ctx, min_term_count=10, matrix_val="count"):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12]), PPMI,
    doc_term_matrix, all_terms = filter_keyphrases_base(ctx.obj["base_dir"], ctx.obj["descriptions"], min_term_count=min_term_count,
                                                        matrix_val=matrix_val, verbose=ctx.obj["verbose"])
    json_dump({"all_terms": all_terms, "doc_term_matrix": doc_term_matrix}, join(ctx.obj["base_dir"], f"doc_cand_matrix_{matrix_val}.json"))
    # TODO filename depends on params


@prepare_candidateterms.command()
@click.pass_context
@click.argument("dtm-filename", type=str)
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def run_lsi(ctx, dtm_filename):
    """as in [VISR12: 4.2.1]"""
    # TODO options here:
    # * if it should filter AFTER the LSI
    from derive_conceptualspace.util.jsonloadstore import json_load
    import numpy as np
    from derive_conceptualspace.util.dtm_object import DocTermMatrix
    from os.path import splitext
    from gensim import corpora
    from gensim.models import LsiModel

    metric = splitext(dtm_filename)[0].split('_')[-1]
    print(f"Using the DocTermMatrix with *b*{metric}*b* as metric!")
    dtm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dtm_filename), assert_meta=("CANDIDATETERM_MIN_OCCURSIN_DOCS", "STANFORDNLP_VERSION")))
    if ctx.obj["verbose"]:
        print("The 25 candidate_terms that occur in the most descriptions (incl the #descriptions they occur in):", {i[0]: len(i[1]) for i in sorted(dtm.term_existinds(use_index=False).items(), key=lambda x: len(x[1]), reverse=True)[:25]})
        if metric.lower() not in ["binary"]:
            max_ind = np.unravel_index(dtm.as_csr().argmax(), dtm.as_csr().shape)
            print(f"Max-{metric}: Term `*b*{dtm.all_terms[max_ind[0]]}*b*` has value *b*{dict(dtm.dtm[max_ind[1]])[max_ind[0]]}*b* for doc `*b*{ctx.obj['mds_obj'].names[max_ind[1]]}*b*`")

    dtm.add_pseudo_keyworddocs()

    #ok so as much for the preprocessing, now let's actually go for the LSI
    dictionary = corpora.Dictionary([list(dtm.all_terms.values())])
    # print("Start creating the LSA-Model with MORE topics than terms...")
    # lsamodel_manytopics = LsiModel(doc_term_matrix, num_topics=len(all_terms) * 2, id2word=dictionary)
    print("Start creating the LSA-Model with FEWER topics than terms...")
    lsamodel_lesstopics = LsiModel(dtm.dtm, num_topics=len(dtm.all_terms)//10, id2word=dictionary)
    print()
    import matplotlib.cm; import matplotlib.pyplot as plt
    #TODO use the mpl_tools here as well to also save plot!
    plt.imshow(lsamodel_lesstopics.get_topics()[:100,:200], vmin=lsamodel_lesstopics.get_topics().min(), vmax=lsamodel_lesstopics.get_topics().max(), cmap=matplotlib.cm.get_cmap("coolwarm")); plt.show()
    print()


# TODO ensure this can be done in snakemake instead
# @cli.command()
# def create_all_datasets():
#     # for n_dims in [20,50,100,200]:
#     #     create_dataset(n_dims, "courses")
#     create_descstyle_dataset(20, "courses")

########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-candidate-svm



@cli.command()
@click.argument("pp-descriptions-filename")
@click.argument("mds-filename")
@click.argument("dcm-filename")
@click.pass_context
def create_candidate_svm(ctx, pp_descriptions_filename, mds_filename, dcm_filename):
    ctx.obj["pp_descriptions_filename"] = pp_descriptions_filename
    ctx.obj["vocab"], ctx.obj["descriptions"], ctx.obj["pp_components"] = load_preprocessed_descriptions(join(ctx.obj["base_dir"], pp_descriptions_filename))
    ctx.obj["mds_filename"] = mds_filename
    tmp = json_load(join(ctx.obj["base_dir"], mds_filename))
    assert tmp["pp_components"] == ctx.obj["pp_components"]
    print(f"Using the MDS which had {tmp['quant_measure']} as quantification measure!")
    mds = tmp["mds"]
    dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATETERM_MIN_OCCURSIN_DOCS", "STANFORDNLP_VERSION")))
    descriptions = ctx.obj["descriptions"]
    assert len(dcm.dtm) == len(descriptions)
    create_candidate_svms_base(dcm, mds, ctx.obj["descriptions"], ctx.obj["verbose"])

########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(auto_envvar_prefix=ENV_PREFIX, obj={})



