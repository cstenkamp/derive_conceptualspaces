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
    ENV_PREFIX
)
from derive_conceptualspace.util.jsonloadstore import json_dump, Struct
from derive_conceptualspace.create_spaces.translate_descriptions import (
    translate_descriptions as translate_descriptions_base,
    count_translations as count_translations_base
)
from derive_conceptualspace.extract_keywords.keywords_main import (
    extract_candidateterms_keybert as extract_candidateterms_keybert_base,
    postprocess_candidateterms as postprocess_candidateterms_base,
    create_doc_term_matrix as create_doc_term_matrix_base,
    filter_keyphrases as filter_keyphrases_base,
)
from derive_conceptualspace.create_spaces.spaces_main import (
    load_translate_mds,
    preprocess_descriptions_full,
    create_dissim_mat as create_dissim_mat_base,
)

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
    json_dump({"vocab": vocab, "descriptions": descriptions, "pp_components": pp_components}, join(ctx.obj["base_dir"], f"preprocessed_descriptions.json"))


@create_spaces.command()
@click.argument("pp-descriptions-filename")
@click.argument("quantification-measure")
@click.pass_context
def create_dissim_mat(ctx, quantification_measure, pp_descriptions_filename):
    quantification, dissim_mat, pp_components = create_dissim_mat_base(ctx.obj["base_dir"], pp_descriptions_filename, quantification_measure)
    json_dump({"quantification": quantification, "dissim_mat": dissim_mat, "pp_components": pp_components}, join(to_data_path, to_data_name))


# @create_spaces.command()
# @click.pass_context
# def create_mds_json(ctx):
#     names, descriptions, mds = create_mds_json_base(ctx.obj["base_dir"], ctx.obj["mds_dimensions"], ctx.obj["translate_policy"])
#     json_dump({"names": names, "descriptions": descriptions, "mds": mds}, join(to_data_path, to_data_name))
#     print(ctx.obj)
#     sleep(2)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.option("--mds-basename", type=str, default=MDS_DEFAULT_BASENAME)
@click.option("--ndm-file", type=click.Path(exists=True), default=None)
@click.option("--translate-policy", type=str, default=DEFAULT_TRANSLATE_POLICY)
@click.pass_context
def prepare_candidateterms(ctx, mds_basename=MDS_DEFAULT_BASENAME, ndm_file=None, translate_policy=DEFAULT_TRANSLATE_POLICY):
    """[group] CLI base to create candidate-terms (everything in here needs the mds_obj)"""
    if not ndm_file:
        ndm_file = next(i for i in os.listdir(ctx.obj["base_dir"]) if i.startswith(mds_basename) and i.endswith(".json"))
        load_basedir = ctx.obj["base_dir"]
    else:
        load_basedir = dirname(ndm_file)
        ndm_file = basename(ndm_file)
    ctx.obj["ndm_file"] = ndm_file
    ctx.obj["mds_obj"] = load_translate_mds(load_basedir, ndm_file, translate_policy=translate_policy)
    ctx.obj["translate_policy"] = translate_policy
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
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert(ctx):
    candidateterms, extractor, (n_immediateworking_ges, n_fixed_ges, n_errs_ges) = extract_candidateterms_keybert_base(ctx.obj["mds_obj"])
    print(f"Immediately working: {n_immediateworking_ges}")
    print(f"Fixed: {n_fixed_ges}")
    print(f"Errors: {n_errs_ges}")
    json_dump({"model": extractor.model_name, "candidate_terms": [list(i) for i in candidateterms]}, join(ctx.obj["base_dir"], "candidate_terms.json"))
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.argument("--postfix", type=str, default="")
@click.pass_context
def postprocess_candidateterms(ctx, postfix=""):
    model, candidate_terms = postprocess_candidateterms_base(ctx.obj["base_dir"], ctx.obj["mds_obj"])
    json_dump({"model": model, "candidate_terms": [list(i) for i in candidate_terms], "postprocessed": True}, join(ctx.obj["base_dir"], f"candidate_terms{postfix}.json"))
    print(f"Saved the post-processed model under candidate_terms{postfix}.json!")
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.option("--json-filename", type=str, default="candidate_terms_postprocessed.json")
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_doc_term_matrix(ctx, json_filename="candidate_terms_postprocessed.json"):
    all_terms, doc_term_matrix = create_doc_term_matrix_base(ctx.obj["base_dir"], ctx.obj["mds_obj"], json_filename, verbose=True)
    json_dump({"all_terms": all_terms, "doc_term_matrix": doc_term_matrix}, join(ctx.obj["base_dir"], "doc_term_matrix.json"))
    #TODO filename depends on params


@prepare_candidateterms.command()
@click.option("--min-term-count", type=int, default=CANDIDATETERM_MIN_OCCURSIN_DOCS)
@click.option("--matrix-val", type=click.Choice(["count", "binary", "tf-idf"], case_sensitive=False), default="count")
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def filter_keyphrases(ctx, min_term_count=10, matrix_val="count"):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12]), PPMI,
    doc_term_matrix, all_terms = filter_keyphrases_base(ctx.obj["base_dir"], ctx.obj["mds_obj"], min_term_count=min_term_count,
                                                        matrix_val=matrix_val, json_filename="candidate_terms_postprocessed.json",
                                                        verbose=ctx.obj["verbose"]) #TODO make sure json_filename always means either input or output, at best rename them
    json_dump({"all_terms": all_terms, "doc_term_matrix": doc_term_matrix}, join(ctx.obj["base_dir"], f"doc_term_matrix_{matrix_val}.json"))
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
@click.argument("ndm-filename", type=str)
@click.argument("dtm-filename", type=str)
@click.option("--translate-policy", type=str, default=DEFAULT_TRANSLATE_POLICY)
@click.pass_context
def create_candidate_svm(ctx, ndm_filename, dtm_filename, translate_policy=DEFAULT_TRANSLATE_POLICY):
    from src.main.util.jsonloadstore import json_load
    from tqdm import tqdm
    import sklearn.svm
    import numpy as np
    from src.main.util.dtm_object import DocTermMatrix

    mds_obj = load_translate_mds(ctx.obj["base_dir"], ndm_filename, translate_policy=translate_policy)
    dtm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dtm_filename), assert_meta=("CANDIDATETERM_MIN_OCCURSIN_DOCS", "STANFORDNLP_VERSION")))
    dtm.as_csr()

    correct_percentages = {}
    for term, exist_indices in tqdm(dtm.term_existinds(use_index=False).items()):
        labels = [False] * len(mds_obj.names)
        for i in exist_indices:
            labels[i] = True
        # TODO figure out if there's a reason to choose LinearSVC over SVC(kernel=linear) or vice versa!
        svm = sklearn.svm.LinearSVC(dual=False, class_weight="balanced")
        svm.fit(mds_obj.mds.embedding_, np.array(labels, dtype=int))
        svm_results = svm.decision_function(mds_obj.mds.embedding_)
        correct_preds = [labels[i] == (svm_results[i] > 0) for i in range(len(labels))]
        correct_percentage = round(sum(correct_preds)/len(correct_preds), 4)*100
        correct_percentages[term] = correct_percentage
        # print(f"Correct Percentage: {correct_percentage}%")

        # decision_plane = Plane(*svm.coef_[0], svm.intercept_[0])
        # with ThreeDFigure() as fig:
        #     X = mds_obj.mds.embedding_
        #     y = np.array(labels, dtype=int)
        #     fig.add_markers(X, color=y, size=1)  # samples
        #
        #     trafo, back_trafo = make_base_changer(decision_plane)
        #     onto_plane = np.array([back_trafo([0, trafo(point)[1], trafo(point)[2]]) for point, side in zip(X, y)])
        #     minx, miny = onto_plane.min(axis=0)[:2]
        #     maxx, maxy = onto_plane.max(axis=0)[:2]
        #     xx, yy = make_meshgrid(minx=minx, miny=miny, maxx=maxx, maxy=maxy, margin=0.1)
        #
        #     fig.add_surface(xx, yy, decision_plane.z)  # decision hyperplane
        #     fig.add_line(X.mean(axis=0) - decision_plane.normal*50, X.mean(axis=0) + decision_plane.normal*10, width=50)  # orthogonal of decision hyperplane through mean of points
        #     fig.add_markers([0, 0, 0], size=10)  # coordinate center
        #     # fig.add_line(-decision_plane.normal * 5, decision_plane.normal * 5)  # orthogonal of decision hyperplane through [0,0,0]
        #     # fig.add_sample_projections(X, decision_plane.normal)  # orthogonal lines from the samples onto the decision hyperplane orthogonal
        #     fig.show()
        # print()
    print(f"Average Correct Percentages: {round(sum(list(correct_percentages.values()))/len(list(correct_percentages.values())), 2)}%")
    sorted_percentages = sorted([[k,round(v,2)] for k,v in correct_percentages.items()], key=lambda x:x[1], reverse=True)
    best_ones = list(dict(sorted_percentages).keys())[:50]
    best_dict = {i: [f"{round(correct_percentages[i], 2)}%", f"{len(dtm.term_existinds(use_index=False)[i])} samples"] for i in best_ones}
    for k, v in best_dict.items():
        print(f"  {k}: {'; '.join(v)}")

########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(auto_envvar_prefix=ENV_PREFIX, obj={})



