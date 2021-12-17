import os
from os.path import join, dirname, basename, abspath
import random
import logging
from datetime import datetime
from time import sleep
import sys

if abspath(join(dirname(__file__), "../..")) not in sys.path:
    sys.path.append(abspath(join(dirname(__file__), "../..")))

import click

from misc_util.telegram_notifier import telegram_notify
from misc_util.logutils import setup_logging
from misc_util.pretty_print import pretty_print as print
from derive_conceptualspace.settings import (
    DEFAULT_TRANSLATE_POLICY,
    DEFAULT_PP_COMPONENTS,
    DEFAULT_QUANTIFICATION_MEASURE,
    DEFAULT_MDS_DIMENSIONS,
    DEFAULT_DEBUG,
    ENV_PREFIX,
    get_setting,
)
from derive_conceptualspace.util.desc_object import pp_descriptions_loader
from derive_conceptualspace.util.jsonloadstore import json_dump, Struct, json_load, JsonPersister
from derive_conceptualspace.create_spaces.translate_descriptions import (
    translate_descriptions as translate_descriptions_base,
    count_translations as count_translations_base
)
from derive_conceptualspace.extract_keywords.postprocess_candidates import (
    postprocess_candidateterms as postprocess_candidateterms_base,
)
from derive_conceptualspace.extract_keywords.keywords_main import (
    extract_candidateterms_keybert as extract_candidateterms_keybert_base,
    create_doc_cand_matrix as create_doc_cand_matrix_base,
    filter_keyphrases as filter_keyphrases_base,
)
from derive_conceptualspace.create_spaces.spaces_main import (
    preprocess_descriptions_full as preprocess_descriptions_base,
    create_dissim_mat as create_dissim_mat_base,
    create_mds_json as create_mds_json_base
)
from derive_conceptualspace.semantic_directions.create_candidate_svm import (
    create_candidate_svms as create_candidate_svms_base
)
from derive_conceptualspace.util.dtm_object import DocTermMatrix, dtm_dissimmat_loader, dtm_loader

logger = logging.getLogger(basename(__file__))


########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli main

def loadstore_settings_envvars(ctx):
    """auto_envvar_prefix only works for options, not for arguments. So this function overwrites ctx.params & ctx.obj
       from env-vars (if they have the correct prefix), and also SETS these env-vars from the cmd-args such that they
       can be accessed using get_setting() """
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

def setup_json_persister(ctx):
    json_persister = JsonPersister(ctx.obj["base_dir"], ctx.obj["base_dir"], ctx, ctx.obj.get("add_relevantparams_to_filename", True))
    json_persister.default_metainf_getters = {"n_samples": lambda: get_setting("DEBUG_N_ITEMS"), "faster_keybert": lambda: get_setting("FASTER_KEYBERT"), "candidate_min_term_count": lambda: get_setting("CANDIDATE_MIN_TERM_COUNT")}
    return json_persister


def set_debug(ctx):
    if get_setting("DEBUG") and not os.getenv(ctx.auto_envvar_prefix+"_DEBUG_SET"):
        print(f"Debug is active! #Items for Debug: {get_setting('DEBUG_N_ITEMS')}")
        if get_setting("RANDOM_SEED", default_none=True):
            random.seed(get_setting("RANDOM_SEED"))
        assert ctx.auto_envvar_prefix+"_CANDIDATE_MIN_TERM_COUNT" not in os.environ
        os.environ[ctx.auto_envvar_prefix+"_CANDIDATE_MIN_TERM_COUNT"] = "1"
        os.environ[ctx.auto_envvar_prefix+"_DEBUG_SET"] = "1"



@click.group()
@click.argument("base-dir", type=str)
@click.option("--verbose/--no-verbose", default=True, help="default: True")
@click.option("--debug/--no-debug", default=DEFAULT_DEBUG, help=f"If True, many functions will only run on a few samples, such that everything should run really quickly. Default: {DEFAULT_DEBUG}")
@click.option("--log", type=str, default="INFO", help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
@click.option("--logfile", type=str, default="", help="logfile to log to. If not set, it will be logged to standard stdout/stderr")
@click.option("--notify-telegram/--no-notify-telegram", default=False, help="If you want to get telegram-notified of start & end of the command")
@click.pass_context
def cli(ctx, base_dir, verbose=False, debug=DEFAULT_DEBUG, log="INFO", logfile=None, notify_telegram=False):
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    loadstore_settings_envvars(ctx)
    del base_dir, verbose, debug, log, logfile, notify_telegram #after the loadstore you cannot use the original ones anymore
    setup_logging(ctx.obj["log"], ctx.obj["logfile"])
    set_debug(ctx)
    ctx.obj["json_persister"] = setup_json_persister(ctx)
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
#prepare descriptions

#TODO: create languages-file from titles & descriptions, see
# create_load_languages_file(from_path, names, descriptions)
# derive_conceptualspace.create_spaces.translate_descriptions.create_load_languages_file

# @cli.command()
# @click.option("--mds-basename", type=str, default=MDS_DEFAULT_BASENAME)
# @click.pass_context
# def translate_descriptions(ctx, mds_basename=MDS_DEFAULT_BASENAME):
#     return translate_descriptions_base(ctx.obj["base_dir"], ctx.obj["mds_basename"])
#
#
# @cli.command()
# @click.option("--mds-basename", type=str, default=None)
# @click.option("--descriptions-basename", type=str, default=None)
# @click.pass_context
# def count_translations(ctx, mds_basename=None, descriptions_basename=None):
#     return count_translations_base(ctx.obj["base_dir"], mds_basename=mds_basename, descriptions_basename=descriptions_basename)


@cli.command()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--raw-descriptions-file", type=str, default="kurse-beschreibungen.csv")
@click.option("--languages-file", type=str, default="languages.json")
@click.option("--translations-file", type=str, default="translated_descriptions.json")
@click.pass_context
def preprocess_descriptions(ctx, pp_components, translate_policy, raw_descriptions_file, languages_file, translations_file):
    ctx.obj["pp_components"] = pp_components
    ctx.obj["translate_policy"] = translate_policy
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    languages = ctx.obj["json_persister"].load(languages_file, "languages", ignore_params=["pp_components", "translate_policy"])
    translations = ctx.obj["json_persister"].load(translations_file, "translations", ignore_params=["pp_components", "translate_policy"])
    vocab, descriptions = preprocess_descriptions_base(raw_descriptions, pp_components, translate_policy, languages, translations)
    ctx.obj["json_persister"].save("preprocessed_descriptions.json", vocab=vocab, descriptions=descriptions, relevant_metainf={"n_samples": len(descriptions)})


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--quantification-measure", type=str, default=lambda: get_setting("QUANTIFICATION_MEASURE"))
@click.pass_context
def create_spaces(ctx, pp_components, translate_policy, quantification_measure):
    """[group] CLI base to create the spaces from texts"""
    ctx.obj["pp_components"] = pp_components
    ctx.obj["translate_policy"] = translate_policy
    ctx.obj["quantification_measure"] = quantification_measure
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)


@create_spaces.command()
@click.pass_context
def create_dissim_mat(ctx):
    pp_descriptions = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", by_config=True, ignore_params=["quantification_measure"], loader=pp_descriptions_loader)
    quant_dtm, dissim_mat = create_dissim_mat_base(pp_descriptions, ctx.obj["quantification_measure"])
    ctx.obj["json_persister"].save("dissim_matrix.json", quant_dtm=quant_dtm, dissim_mat=dissim_mat)


@create_spaces.command()
@click.option("--mds-dimensions", type=int, default=lambda: get_setting("MDS_DIMENSIONS"))
@click.pass_context
def create_mds_json(ctx, mds_dimensions):
    ctx.obj["mds_dimensions"] = mds_dimensions
    dissim_mat = ctx.obj["json_persister"].load(None, "dissim_matrix", by_config=True, ignore_params=["mds_dimensions"], loader=dtm_dissimmat_loader)
    mds = create_mds_json_base(dissim_mat, mds_dimensions)
    ctx.obj["json_persister"].save("mds.json", mds=mds)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--extraction-method", type=str, default=lambda: get_setting("EXTRACTION_METHOD"))
@click.pass_context
def prepare_candidateterms(ctx, pp_components, translate_policy, extraction_method):
    """[group] CLI base to extract candidate-terms from texts"""
    ctx.obj["pp_components"] = pp_components
    ctx.obj["translate_policy"] = translate_policy
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", by_config=True, loader=pp_descriptions_loader)
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)


@prepare_candidateterms.command()
@click.pass_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=lambda: get_setting("FASTER_KEYBERT"))
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert(ctx, faster_keybert):
    candidateterms = extract_candidateterms_keybert_base(ctx.obj["pp_descriptions"], ctx.obj["extraction_method"], faster_keybert, verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("candidate_terms.json", candidateterms=candidateterms, relevant_metainf={"faster_keybert": faster_keybert})


@prepare_candidateterms.command()
@click.pass_context
def postprocess_candidateterms(ctx):
    ctx.obj["candidate_terms"] = ctx.obj["json_persister"].load(None, "candidate_terms", by_config=True)
    postprocessed_candidates = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.obj["extraction_method"])
    ctx.obj["json_persister"].save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates)


@prepare_candidateterms.command()
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_doc_cand_matrix(ctx):
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates", by_config=True)
    doc_term_matrix = create_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("doc_cand_matrix.json", doc_term_matrix=doc_term_matrix)
    #TODO can't I go for a quantification_measure in this doc-cand-matrix as well?!
    #TODO the create_doc_cand_matrix_base function used to have a "assert_postprocessed" arguement, can I still do this?!


#TODO the click.Choice muss als argument settings.ALL_... haben
@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=lambda: get_setting("CANDIDATE_MIN_TERM_COUNT"))
@click.option("--dcm-quant-measure", type=click.Choice(["count", "binary", "tf-idf"], case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE"))
@click.pass_context
@telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def filter_keyphrases(ctx, candidate_min_term_count, dcm_quant_measure):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12]), PPMI,
    ctx.obj["candidate_min_term_count"] = candidate_min_term_count
    ctx.obj["dcm_quant_measure"] = dcm_quant_measure
    ctx.obj["doc_cand_matrix"] = ctx.obj["json_persister"].load(None, "doc_cand_matrix", by_config=True, loader=dtm_loader)
    filtered_dcm = filter_keyphrases_base(ctx.obj["doc_cand_matrix"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count, dcm_quant_measure=dcm_quant_measure, verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("filtered_dcm.json", relevant_metainf={"candidate_min_term_count": candidate_min_term_count}, doc_term_matrix=filtered_dcm)
    #TODO: use_n_docs_count as argument


########################################################################################################################
########################################################################################################################
########################################################################################################################
# generate-conceptualspace group

#TODO the click.Choice muss als argument settings.ALL_... haben
@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--quantification-measure", type=str, default=lambda: get_setting("QUANTIFICATION_MEASURE"))
@click.option("--mds-dimensions", type=int, default=lambda: get_setting("MDS_DIMENSIONS"))
@click.option("--extraction-method", type=str, default=lambda: get_setting("EXTRACTION_METHOD"))
@click.option("--dcm-quant-measure", type=click.Choice(["count", "binary", "tf-idf"], case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE"))
@click.pass_context
def generate_conceptualspace(ctx, pp_components, translate_policy, quantification_measure, mds_dimensions, extraction_method, dcm_quant_measure):
    """[group] CLI base to create the actual conceptual spaces"""
    ctx.obj["pp_components"] = pp_components
    ctx.obj["translate_policy"] = translate_policy
    ctx.obj["quantification_measure"] = quantification_measure
    ctx.obj["mds_dimensions"] = mds_dimensions
    ctx.obj["extraction_method"] = extraction_method
    ctx.obj["dcm_quant_measure"] = dcm_quant_measure
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", by_config=True, loader=pp_descriptions_loader, ignore_params=["quantification_measure", "mds_dimensions"])
    ctx.obj["filtered_dcm"] = ctx.obj["json_persister"].load(None, "filtered_dcm", by_config=True, loader=dtm_loader, ignore_params=["quantification_measure", "mds_dimensions"])
    ctx.obj["mds"] = ctx.obj["json_persister"].load(None, "mds", by_config=True, loader=lambda **args: args["mds"])
    assert ctx.obj["mds"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your MDS has {ctx.obj["mds"].embedding_.shape[0] } descriptions!'
    if ctx.obj["notify_telegram"] == True:
        if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
            ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)


@generate_conceptualspace.command()
@click.pass_context
def create_candidate_svm(ctx):
    clusters, cluster_directions, kappa_scores, decision_planes = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["mds"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("clusters.json", clusters=clusters, cluster_directions=cluster_directions, kappa_scores=kappa_scores, decision_planes=decision_planes)
    #TODO hier war dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION"))), krieg ich das wieder hin?

#
#
# @prepare_candidateterms.command()
# @click.pass_context
# @click.argument("dtm-filename", type=str)
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
# def run_lsi(ctx, dtm_filename):
#     """as in [VISR12: 4.2.1]"""
#     # TODO options here:
#     # * if it should filter AFTER the LSI
#     from derive_conceptualspace.util.jsonloadstore import json_load
#     import numpy as np
#     from derive_conceptualspace.util.dtm_object import DocTermMatrix
#     from os.path import splitext
#     from gensim import corpora
#     from gensim.models import LsiModel
#
#     metric = splitext(dtm_filename)[0].split('_')[-1]
#     print(f"Using the DocTermMatrix with *b*{metric}*b* as metric!")
#     dtm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dtm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION")))
#     if ctx.obj["verbose"]:
#         print("The 25 candidate_terms that occur in the most descriptions (incl the #descriptions they occur in):", {i[0]: len(i[1]) for i in sorted(dtm.term_existinds(use_index=False).items(), key=lambda x: len(x[1]), reverse=True)[:25]})
#         if metric.lower() not in ["binary"]:
#             max_ind = np.unravel_index(dtm.as_csr().argmax(), dtm.as_csr().shape)
#             print(f"Max-{metric}: Term `*b*{dtm.all_terms[max_ind[0]]}*b*` has value *b*{dict(dtm.dtm[max_ind[1]])[max_ind[0]]}*b* for doc `*b*{ctx.obj['mds_obj'].names[max_ind[1]]}*b*`")
#
#     dtm.add_pseudo_keyworddocs()
#
#     #ok so as much for the preprocessing, now let's actually go for the LSI
#     dictionary = corpora.Dictionary([list(dtm.all_terms.values())])
#     # print("Start creating the LSA-Model with MORE topics than terms...")
#     # lsamodel_manytopics = LsiModel(doc_term_matrix, num_topics=len(all_terms) * 2, id2word=dictionary)
#     print("Start creating the LSA-Model with FEWER topics than terms...")
#     lsamodel_lesstopics = LsiModel(dtm.dtm, num_topics=len(dtm.all_terms)//10, id2word=dictionary)
#     print()
#     import matplotlib.cm; import matplotlib.pyplot as plt
#     #TODO use the mpl_tools here as well to also save plot!
#     plt.imshow(lsamodel_lesstopics.get_topics()[:100,:200], vmin=lsamodel_lesstopics.get_topics().min(), vmax=lsamodel_lesstopics.get_topics().max(), cmap=matplotlib.cm.get_cmap("coolwarm")); plt.show()
#     print()
#
#
#
# @cli.command()
# @click.argument("pp-descriptions-filename")
# @click.argument("mds-filename")
# @click.argument("dcm-filename")
# @click.argument("clusteredcands-filename")
# @click.pass_context
# def rank_courses_saldirs(ctx, pp_descriptions_filename, mds_filename, dcm_filename, clusteredcands_filename):
#     from derive_conceptualspace.util.base_changer import NDPlane
#     import numpy as np
#     ctx.obj["pp_descriptions_filename"] = pp_descriptions_filename
#     ctx.obj["vocab"], ctx.obj["descriptions"], ctx.obj["pp_components"] = load_preprocessed_descriptions(join(ctx.obj["base_dir"], pp_descriptions_filename))
#     ctx.obj["mds_filename"] = mds_filename
#     tmp = json_load(join(ctx.obj["base_dir"], mds_filename))
#     assert tmp["pp_components"] == ctx.obj["pp_components"]
#     print(f"Using the MDS which had {tmp['quant_measure']} as quantification measure!")
#     mds = tmp["mds"]
#     dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION")))
#     descriptions = ctx.obj["descriptions"]
#     assert len(dcm.dtm) == len(descriptions), f"The Doc-Candidate-Matrix contains {len(dcm.dtm)} items and you have {len(descriptions)} descriptions!"
#     for desc, embedding in zip(descriptions, list(mds.embedding_)):
#         desc.embedding = embedding
#     tmp = json_load(join(ctx.obj["base_dir"], clusteredcands_filename))
#     clusters, cluster_directions, kappa_scores, decision_planes = tmp["clusters"], tmp["cluster_directions"], tmp["kappa_scores"], tmp["decision_planes"]
#     clusters = {k: {"components": [k]+v, "direction": cluster_directions[k], "kappa_scores": {i: kappa_scores[i] for i in [k]+v}, "decision_planes": {i: NDPlane(np.array(decision_planes[i][1]), decision_planes[i][0]) for i in [k]+v}} for k,v in clusters.items()}
#     rank_courses_saldirs_base(descriptions, clusters)
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



