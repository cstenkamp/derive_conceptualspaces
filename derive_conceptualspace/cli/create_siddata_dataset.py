import inspect
from functools import wraps
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

from derive_conceptualspace.settings import ALL_DCM_QUANT_MEASURE, ENV_PREFIX, get_setting, set_envvar, get_envvar
from derive_conceptualspace.util.desc_object import pp_descriptions_loader
from derive_conceptualspace.util.jsonloadstore import JsonPersister
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
from derive_conceptualspace.create_spaces.preprocess_descriptions import (
    preprocess_descriptions_full as preprocess_descriptions_base,
)
from derive_conceptualspace.create_spaces.spaces_main import (
    create_dissim_mat as create_dissim_mat_base,
)
from derive_conceptualspace.create_spaces.create_mds import (
    create_mds as create_mds_json_base,
)
from derive_conceptualspace.semantic_directions.create_candidate_svm import (
    create_candidate_svms as create_candidate_svms_base
)
from derive_conceptualspace.util.dtm_object import dtm_dissimmat_loader, dtm_loader

logger = logging.getLogger(basename(__file__))

########################################################################################################################
########################################################################################################################
########################################################################################################################
#cli helpers & main

def loadstore_settings_envvars(ctx, use_auto_envvar_prefix=False):
    """auto_envvar_prefix only works for options, not for arguments. So this function overwrites ctx.params & ctx.obj
       from env-vars (if they have the correct prefix), and also SETS these env-vars from the cmd-args such that they
       can be accessed using get_setting() """
    env_prefix = ctx.auto_envvar_prefix if use_auto_envvar_prefix else ENV_PREFIX
    #the auto_envvar_prefix always gets appended the subcommand, I don't want that generally though.
    for param, val in ctx.params.items():
        ctx.obj[param] = val
        envvarname = env_prefix+"_"+param.upper().replace("-","_")
        # https://github.com/pallets/click/issues/714#issuecomment-651389598
        if (envvar := get_envvar(envvarname)) is not None and envvar != ctx.params[param]:
            print(f"The param {param} used to be {ctx.params[param]}, but is overwritten by an env-var to {envvar}")
            ctx.params[param] = envvar
            ctx.obj[param] = envvar
        else:
            set_envvar(envvarname, ctx.obj[param])


def setup_json_persister(ctx, ignore_nsamples=False):
    json_persister = JsonPersister(ctx.obj["base_dir"], ctx.obj["base_dir"], ctx, ctx.obj.get("add_relevantparams_to_filename", True))
    if ignore_nsamples:
        n_samples_getter = lambda: "ANY"
        cand_ntc_getter = lambda: "ANY"
    else:
        n_samples_getter = lambda: get_setting("DEBUG_N_ITEMS", silent=True) if get_setting("DEBUG", silent=True) else 7588 #TODO don't hard-code this!
        cand_ntc_getter = lambda: get_setting("CANDIDATE_MIN_TERM_COUNT", silent=True)
    json_persister.default_metainf_getters = {"n_samples": n_samples_getter,
                                              "faster_keybert": lambda: get_setting("FASTER_KEYBERT", silent=True),
                                              "candidate_min_term_count": cand_ntc_getter}
    return json_persister


def set_debug(ctx, use_auto_envvar_prefix=False):
    env_prefix = ctx.auto_envvar_prefix if use_auto_envvar_prefix else ENV_PREFIX
    if get_setting("DEBUG"):
        if not os.getenv(env_prefix+"_DEBUG_SET"):
            assert env_prefix+"_CANDIDATE_MIN_TERM_COUNT" not in os.environ
            os.environ[env_prefix + "_CANDIDATE_MIN_TERM_COUNT"] = "1"
            print(f"Debug is active! #Items for Debug: {get_setting('DEBUG_N_ITEMS')}")
            if get_setting("RANDOM_SEED", default_none=True): print("Using a random seed!")
        if get_setting("RANDOM_SEED", default_none=True):
            random.seed(get_setting("RANDOM_SEED"))
        assert os.environ[env_prefix + "_CANDIDATE_MIN_TERM_COUNT"] == "1"
        os.environ[env_prefix+"_DEBUG_SET"] = "1"



def click_pass_add_context(fn):
    @click.pass_context
    @wraps(fn)
    def wrapped(*args, **kwargs):
        assert isinstance(args[0], click.Context)
        for k, v in kwargs.items():
            assert k not in args[0].obj
            args[0].obj[k] = v
        ctx = args[0]
        nkw = {k:v for k,v in kwargs.items() if k in set(inspect.getfullargspec(fn).args)-{"ctx", "context"}}
        loadstore_settings_envvars(ctx)
        res = fn(*args, **nkw)
        if isinstance(ctx.command, click.Group):
            if ctx.obj["notify_telegram"] == True:
                print("wtfff")
                if not isinstance(cli.get_command(ctx, ctx.invoked_subcommand), click.Group):
                    ctx.command.get_command(ctx, ctx.invoked_subcommand).callback = telegram_notify(only_terminal=False, only_on_fail=False, log_start=True)(ctx.command.get_command(ctx, ctx.invoked_subcommand).callback)
        return res
    return wrapped


@click.group()
@click.argument("base-dir", type=str)
@click.option("--verbose/--no-verbose", default=True, help="default: True")
@click.option("--debug/--no-debug", default=get_setting("DEBUG"), help=f"If True, many functions will only run on a few samples, such that everything should run really quickly. Default: {get_setting('DEBUG')}")
@click.option("--log", type=str, default="INFO", help="log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
@click.option("--logfile", type=str, default="", help="logfile to log to. If not set, it will be logged to standard stdout/stderr")
@click.option("--notify-telegram/--no-notify-telegram", default=False, help="If you want to get telegram-notified of start & end of the command")
@click_pass_add_context
def cli(ctx):
    print("Starting up at", datetime.now().strftime("%d.%m.%Y, %H:%M:%S"))
    setup_logging(ctx.obj["log"], ctx.obj["logfile"])
    set_debug(ctx)
    ctx.obj["json_persister"] = setup_json_persister(ctx)


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
@click_pass_add_context
def preprocess_descriptions(ctx, raw_descriptions_file, languages_file, translations_file):
    raw_descriptions = ctx.obj["json_persister"].load(raw_descriptions_file, "raw_descriptions", ignore_params=["pp_components", "translate_policy"])
    languages = ctx.obj["json_persister"].load(languages_file, "languages", ignore_params=["pp_components", "translate_policy"])
    translations = ctx.obj["json_persister"].load(translations_file, "translations", ignore_params=["pp_components", "translate_policy"])
    vocab, descriptions = preprocess_descriptions_base(raw_descriptions, ctx.obj["pp_components"], ctx.obj["translate_policy"], languages, translations)
    ctx.obj["json_persister"].save("preprocessed_descriptions.json", vocab=vocab, descriptions=descriptions, relevant_metainf={"n_samples": len(descriptions)})


########################################################################################################################
########################################################################################################################
########################################################################################################################
#create-spaces group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--quantification-measure", type=str, default=lambda: get_setting("QUANTIFICATION_MEASURE"))
@click_pass_add_context
def create_spaces(ctx):
    """[group] CLI base to create the spaces from texts"""
    pass


@create_spaces.command()
@click_pass_add_context
def create_dissim_mat(ctx):
    pp_descriptions = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", ignore_params=["quantification_measure"], loader=pp_descriptions_loader)
    quant_dtm, dissim_mat = create_dissim_mat_base(pp_descriptions, ctx.obj["quantification_measure"])
    ctx.obj["json_persister"].save("dissim_matrix.json", quant_dtm=quant_dtm, dissim_mat=dissim_mat)


@create_spaces.command()
@click.option("--mds-dimensions", type=int, default=lambda: get_setting("MDS_DIMENSIONS"))
@click_pass_add_context
def create_mds_json(ctx):
    dissim_mat = ctx.obj["json_persister"].load(None, "dissim_matrix", ignore_params=["mds_dimensions"], loader=dtm_dissimmat_loader)
    mds = create_mds_json_base(dissim_mat, ctx.obj["mds_dimensions"])
    ctx.obj["json_persister"].save("mds.json", mds=mds)

########################################################################################################################
########################################################################################################################
########################################################################################################################
#prepare-candidateterms group

@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--extraction-method", type=str, default=lambda: get_setting("EXTRACTION_METHOD"))
@click_pass_add_context
def prepare_candidateterms(ctx):
    """[group] CLI base to extract candidate-terms from texts"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", loader=pp_descriptions_loader)


@prepare_candidateterms.command()
@click_pass_add_context
def extract_candidateterms_stanfordlp(ctx):
    raise NotImplementedError()
    # names, descriptions, _, _ = load_mds(join(SID_DATA_BASE, f"siddata_names_descriptions_mds_20.json"))
    # nlp = download_activate_stanfordnlp(DATA_BASE, ["english", "german"])
    # print(stanford_extract_nounphrases(nlp, descriptions[1]))


@prepare_candidateterms.command()
@click.option("--faster-keybert/--no-faster-keybert", default=lambda: get_setting("FASTER_KEYBERT"))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def extract_candidateterms_keybert(ctx):
    candidateterms = extract_candidateterms_keybert_base(ctx.obj["pp_descriptions"], ctx.obj["extraction_method"], ctx.obj["faster_keybert"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("candidate_terms.json", candidateterms=candidateterms, relevant_metainf={"faster_keybert": ctx.obj["faster_keybert"]})


@prepare_candidateterms.command()
@click_pass_add_context
def postprocess_candidateterms(ctx):
    ctx.obj["candidate_terms"] = ctx.obj["json_persister"].load(None, "candidate_terms")
    postprocessed_candidates = postprocess_candidateterms_base(ctx.obj["candidate_terms"], ctx.obj["pp_descriptions"], ctx.obj["extraction_method"])
    ctx.obj["json_persister"].save("postprocessed_candidates.json", postprocessed_candidates=postprocessed_candidates)


@prepare_candidateterms.command()
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def create_doc_cand_matrix(ctx):
    ctx.obj["postprocessed_candidates"] = ctx.obj["json_persister"].load(None, "postprocessed_candidates")
    doc_term_matrix = create_doc_cand_matrix_base(ctx.obj["postprocessed_candidates"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("doc_cand_matrix.json", doc_term_matrix=doc_term_matrix)
    #TODO can't I go for a quantification_measure in this doc-cand-matrix as well?!
    #TODO the create_doc_cand_matrix_base function used to have a "assert_postprocessed" arguement, can I still do this?!


@prepare_candidateterms.command()
@click.option("--candidate-min-term-count", type=int, default=lambda: get_setting("CANDIDATE_MIN_TERM_COUNT"))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE"))
@click_pass_add_context
# @telegram_notify(only_terminal=True, only_on_fail=False, log_start=True)
def filter_keyphrases(ctx, candidate_min_term_count):
    # TODO missing options here: `tag-share` (chap. 4.2.1 of [VISR12]), PPMI,
    ctx.obj["doc_cand_matrix"] = ctx.obj["json_persister"].load(None, "doc_cand_matrix", loader=dtm_loader)
    filtered_dcm = filter_keyphrases_base(ctx.obj["doc_cand_matrix"], ctx.obj["pp_descriptions"], min_term_count=candidate_min_term_count, dcm_quant_measure=ctx.obj["dcm_quant_measure"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("filtered_dcm.json", relevant_metainf={"candidate_min_term_count": candidate_min_term_count}, doc_term_matrix=filtered_dcm)
    #TODO: use_n_docs_count as argument


########################################################################################################################
########################################################################################################################
########################################################################################################################
# generate-conceptualspace group


@cli.group()
@click.option("--pp-components", type=str, default=lambda: get_setting("PP_COMPONENTS"))
@click.option("--translate-policy", type=str, default=lambda: get_setting("TRANSLATE_POLICY"))
@click.option("--quantification-measure", type=str, default=lambda: get_setting("QUANTIFICATION_MEASURE"))
@click.option("--mds-dimensions", type=int, default=lambda: get_setting("MDS_DIMENSIONS"))
@click.option("--extraction-method", type=str, default=lambda: get_setting("EXTRACTION_METHOD"))
@click.option("--dcm-quant-measure", type=click.Choice(ALL_DCM_QUANT_MEASURE, case_sensitive=False), default=lambda: get_setting("DCM_QUANT_MEASURE"))
@click_pass_add_context
def generate_conceptualspace(ctx):
    """[group] CLI base to create the actual conceptual spaces"""
    ctx.obj["pp_descriptions"] = ctx.obj["json_persister"].load(None, "preprocessed_descriptions", loader=pp_descriptions_loader, ignore_params=["quantification_measure", "mds_dimensions"])
    ctx.obj["filtered_dcm"] = ctx.obj["json_persister"].load(None, "filtered_dcm", loader=dtm_loader, ignore_params=["quantification_measure", "mds_dimensions"])
    ctx.obj["mds"] = ctx.obj["json_persister"].load(None, "mds", ignore_params=["extraction_method", "dcm_quant_measure"], loader=lambda **args: args["mds"])
    assert ctx.obj["mds"].embedding_.shape[0] == len(ctx.obj["filtered_dcm"].dtm), f'The Doc-Candidate-Matrix contains {len(ctx.obj["filtered_dcm"].dtm)} items But your MDS has {ctx.obj["mds"].embedding_.shape[0] } descriptions!'


@generate_conceptualspace.command()
@click_pass_add_context
def create_candidate_svm(ctx):
    clusters, cluster_directions, kappa_scores, decision_planes = create_candidate_svms_base(ctx.obj["filtered_dcm"], ctx.obj["mds"], ctx.obj["pp_descriptions"], verbose=ctx.obj["verbose"])
    ctx.obj["json_persister"].save("clusters.json", clusters=clusters, cluster_directions=cluster_directions, kappa_scores=kappa_scores, decision_planes=decision_planes)
    #TODO hier war dcm = DocTermMatrix(json_load(join(ctx.obj["base_dir"], dcm_filename), assert_meta=("CANDIDATE_MIN_TERM_COUNT", "STANFORDNLP_VERSION"))), krieg ich das wieder hin?

#TODO I can do something like autodetect_relevant_params und metainf im json_persister

@generate_conceptualspace.command()
@click_pass_add_context
def rank_courses_saldirs(ctx):
    ctx.obj["clusters"] = ctx.obj["json_persister"].load(None, "clusters")
    print()

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
#
# def rank_courses_saldirs_base(descriptions, clusters):
#     assert hasattr(descriptions[0], "embedding")
#     print()



########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == '__main__':
    cli(auto_envvar_prefix=ENV_PREFIX, obj={})



