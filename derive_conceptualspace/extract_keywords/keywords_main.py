import sys
from collections import Counter
from os.path import join, basename
import logging
import random

from tqdm import tqdm
from gensim import corpora
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

from .get_candidates_keybert import KeyBertExtractor
from .get_candidates_rules import extract_coursetype

from derive_conceptualspace.util.text_tools import tf_idf, get_stopwords, ppmi


from misc_util.pretty_print import pretty_print as print
from ..settings import get_setting
from ..util.dtm_object import DocTermMatrix, csr_to_list

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################
import os
from datetime import datetime, timedelta
from collections import deque
if os.getenv("WALL_SECS"):
    wall_secs = int(os.environ['WALL_SECS'])
    killed_at = datetime.now() + timedelta(seconds=wall_secs)
    os.environ["KILLED_AT"] = killed_at.strftime("%d.%m.%Y, %H:%M:%S")
    print(f"Walltime: {wall_secs}s. Will be killed at {os.environ['KILLED_AT']}")


def interruptible(iterable, shutdown_time=60, timeavg_samples=10, continue_from=None, pgbar=None, append_var=None, metainf_var=None, kb_int_ref=None):
    #pgbar is either None or a string. If it's not None, we will wrap in tqdm and use the var-value as desc
    #kb_int is a dictionary containing one singe boolean, to be set to True from the OUTSIDE with a keyboard-interrupt
    interrupt_time = None
    if os.getenv("KILLED_AT"):
        interrupt_time = datetime.strptime(os.environ["KILLED_AT"], "%d.%m.%Y, %H:%M:%S")-timedelta(seconds=shutdown_time)
        print(f"This loop will interrupt at {interrupt_time.strftime('%d.%m.%Y, %H:%M:%S')}")
    last_times = deque([datetime.now()], maxlen=timeavg_samples)
    if continue_from is not None:
        old_results, old_metainf, countervarnames = continue_from
        assert append_var is not None
        if old_metainf is not None: assert metainf_var is not None
        print(f"Continuing a previously interrupted loop ({old_metainf['N_RUNS']} runs already). Starting at element {old_metainf['INTERRUPTED_AT'] + 1}/{len(iterable)}")
        assert all(v == old_metainf[k] for k, v in metainf_var.items()), "The metainf changed!"
        for elem in old_results:
            append_var.append(elem)
        full_len = len(iterable)
        iterable = iterable[old_metainf['INTERRUPTED_AT']:]
        sys.stderr.flush(); sys.stdout.flush()
    n_elems = len(iterable)  #TODO only if it has a len!
    iter = enumerate(iterable) if pgbar is None else enumerate(tqdm(iterable, desc=pgbar, total=n_elems))
    for n, elem in iter:
        yield elem
        last_times.append(datetime.now())
        timedeltas = [last_times[i]-last_times[i-1] for i in range(1, len(last_times))]
        average_timedelta = sum(timedeltas, timedelta(0)) / len(timedeltas) # https://stackoverflow.com/a/3617540/5122790
        if (interrupt_time and datetime.now()+average_timedelta*0.9 >= interrupt_time) or kb_int_ref["kb_int"]:
            if kb_int_ref["kb_int"]:
                metainf_var["KEYBORD_INTERRUPTED"] = True
            if continue_from is not None:
                print(f"Interrupted at iteration {n}/{n_elems} ({old_metainf['INTERRUPTED_AT']+n}/{full_len}) because we will hit the wall-time!")
                metainf_var["INTERRUPTED_AT"] = n+old_metainf["INTERRUPTED_AT"]
            else:
                print(f"Interrupted at iteration {n}/{n_elems} because we will hit the wall-time!")
                metainf_var["INTERRUPTED_AT"] = n
            metainf_var["NEWLY_INTERRUPTED"] = True
            raise InterruptedError()



def extract_candidateterms(pp_descriptions, extraction_method, max_ngram, verbose=False, continue_from=None):
    if extraction_method == "keybert":
        candidateterms, metainf = extract_candidateterms_keybert_nopp(pp_descriptions, max_ngram, get_setting("faster_keybert"), verbose=verbose, continue_from=continue_from)
    elif extraction_method == "pp_keybert":
        candidateterms, metainf = extract_candidateterms_keybert_preprocessed(pp_descriptions, max_ngram, get_setting("faster_keybert"), verbose=verbose, continue_from=continue_from)
    elif extraction_method in ["tfidf", "tf", "all", "ppmi"]:
        candidateterms, metainf = extract_candidateterms_quantific(pp_descriptions, max_ngram, quantific=extraction_method, verbose=verbose, continue_from=continue_from)
    else:
        raise NotImplementedError()
    flattened = set(flatten(candidateterms))
    print("Terms I found: ", ", ".join([f"{k+1}-grams: {v}" for k, v in sorted(Counter([i.count(" ") for i in flattened]).items(), key=lambda x:x[0])]), "| sum:", len(flattened))
    metainf["n_candidateterms"] = len(flattened)
    return candidateterms, metainf

def extract_candidateterms_keybert_nopp(descriptions, max_ngram, faster_keybert=False, verbose=False, continue_from=None):
    is_multilan = get_setting("TRANSLATE_POLICY") == "origlang" or get_setting("LANGUAGE") != "en"
    extractor = KeyBertExtractor(is_multilan=is_multilan, faster=faster_keybert, max_ngram=max_ngram)
    metainf = {"keybertextractor_modelname": extractor.model_name}
    if get_setting("DEBUG"):
        assert continue_from is None, "This is too complex for me right now"
        descriptions._descriptions = descriptions._descriptions[:get_setting("DEBUG_N_ITEMS")]
    candidateterms = []
    n_immediateworking_ges, n_fixed_ges, n_errs_ges = 0, 0, 0
    kb_int = {"kb_int": False}
    try:
        for desc in interruptible(descriptions._descriptions, continue_from=continue_from, pgbar="Extracting Keywords...", append_var=candidateterms, metainf_var=metainf, kb_int_ref=kb_int):
            try:
                keyberts, origextracts, (n_immediateworking, n_fixed, n_errs) = extractor(desc.text, desc.lang)
                #TODO best post-processing for candidates is to run the same processing as for the descriptions! => afterwards 100% of them should be in the processed_text (started doing that in postprocess_candidates!)
                if (ct := extract_coursetype(desc)) and ct not in keyberts:
                    keyberts += [ct]
                candidateterms.append(keyberts)
                n_immediateworking_ges += n_immediateworking
                n_fixed_ges += n_fixed
                n_errs_ges += n_errs
            except KeyboardInterrupt:
                kb_int["kb_int"] = True
    except InterruptedError:
        assert "NEWLY_INTERRUPTED" in metainf
    metainf = {**metainf, "n_immediateworking": n_immediateworking_ges, "n_fixed": n_fixed_ges, "n_errs": n_errs_ges}
    return candidateterms, metainf


def extract_candidateterms_keybert_preprocessed(descriptions, max_ngram, faster_keybert=False, verbose=False, **kwargs):
    from keybert import KeyBERT  # lazily loaded as it needs tensorflow/torch which takes some time to init
    model_name = "paraphrase-MiniLM-L6-v2" if faster_keybert else "paraphrase-mpnet-base-v2"
    print(f"Using model {model_name}")
    candidateterms = []
    kw_model = KeyBERT(model_name)
    descs = descriptions._descriptions if not get_setting("DEBUG") else descriptions._descriptions[:get_setting("DEBUG_N_ITEMS")]
    for desc in tqdm(descs, desc="Running KeyBERT on descriptions"):
        stopwords = get_stopwords(desc.lang)
        candidates = set()
        for nwords in range(1, max_ngram):
            n_candidates = kw_model.extract_keywords(desc.processed_as_string(), keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)
        if (ct := extract_coursetype(desc)) and ct not in candidates:
            candidates += [ct]
        candidateterms.append(candidates)
    return candidateterms, {"keybertextractor_modelname": model_name, "kw_max_ngram": max_ngram}

#TODO play around with the parameters here!
def extract_candidateterms_quantific(descriptions, max_ngram, quantific, verbose=False, **kwargs):
    print(f"Loading Doc-Term-Matrix with min-term-count {get_setting('CANDIDATE_MIN_TERM_COUNT')}")
    dtm, mta = descriptions.generate_DocTermMatrix(min_df=get_setting("CANDIDATE_MIN_TERM_COUNT"), max_ngram=max_ngram, do_tfidf=quantific if quantific in ["tfidf", "tf"] else None)
    #Now I'm filtering here, I originally didn't want to do that but it makes the processing incredibly much faster
    if quantific in ["tfidf", "tf"]:
        quant = dtm.dtm if not mta.get("sklearn_tfidf") else csr_to_list(TfidfTransformer(use_idf=(quantific=="tfidf")).fit_transform(dtm.as_csr().T)) # tf_idf(dtm, verbose=verbose, descriptions=descriptions)
    elif quantific == "ppmi":
        quant = ppmi(dtm, verbose=verbose, descriptions=descriptions)
    elif quantific == "all":
        return dtm.terms_per_doc(), {"all_terms_are_candidates": True}
    else:
        raise NotImplementedError()
    metainf = dict()
    min_val = get_setting("QUANTEXTRACT_MINVAL"); min_val_percentile = get_setting("QUANTEXTRACT_MINVAL_PERC")
    assert not (min_val and min_val_percentile)
    if min_val_percentile:
        min_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), min_val_percentile * 100)
        metainf.update(kw_min_val_percentile=min_val_percentile, kw_calculated_minval=min_val)
    else:
        metainf.update(kw_min_val=min_val)
    forcetake_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), get_setting("QUANTEXTRACT_FORCETAKE_PERC") * 100)
    candidates = [ [ sorted(i, key=lambda x:x[1], reverse=True),
                     min(round(len(i)*get_setting("QUANTEXTRACT_MAXPERDOC_REL")), get_setting("QUANTEXTRACT_MAXPERDOC_ABS")),
                     set(j[0] for j in i if j[1] >= forcetake_val)]
                 for i in quant]
    all_candidates = [set([k[0] for k in i[0][:get_setting("QUANTEXTRACT_MINPERDOC")]])|set([j[0] for j in i[0] if j[1] >= min_val][:i[1]])|i[2] for i in candidates]
    if len(set(flatten(all_candidates))) == len(dtm.all_terms):
        metainf["all_terms_are_candidates"] = True
    return [[dtm.all_terms[j] for j in i] for i in all_candidates], metainf


########################################################################################################################
########################################################################################################################
########################################################################################################################

def create_filtered_doc_cand_matrix(postprocessed_candidates, descriptions, min_term_count, dcm_quant_measure, use_n_docs_count=True, verbose=False):
    assert min_term_count >= 2, "It doesn't make sense to consider candidates occuring only once, and if I don't, I can rely on the DTM from the dissim_mat (for 1-grams at least)!"
    doc_cand_matrix = create_doc_cand_matrix(postprocessed_candidates, descriptions, verbose=verbose)
    return filter_keyphrases(doc_cand_matrix, descriptions, min_term_count=min_term_count, dcm_quant_measure=dcm_quant_measure, use_n_docs_count=use_n_docs_count, verbose=verbose)


def create_doc_cand_matrix(postprocessed_candidates, descriptions, verbose=False):
    assert len(postprocessed_candidates) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions._descriptions) for cand in postprocessed_candidates[ndesc])
    all_phrases = list(set(flatten(postprocessed_candidates)))
    if get_setting("DEBUG"):
        all_phrases = all_phrases[:get_setting("DEBUG_N_ITEMS")]
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    # dictionary = corpora.Dictionary([all_phrases])
    dtm = [sorted([(nphrase, desc.count_phrase(phrase)) for nphrase, phrase in enumerate(all_phrases) if phrase in desc], key=lambda x:x[0]) for ndesc, desc in enumerate(tqdm(descriptions._descriptions, desc="Creating Doc-Cand-Matrix"))]
    #TODO statt dem ^ kann ich wieder SkLearn nehmen
    assert all([n for n,i in enumerate(descriptions._descriptions) if term in i] == [n for n, i in enumerate(dtm) if all_phrases.index(term) in [j[0] for j in i]] for term in random.sample(all_phrases, 5))
    doc_term_matrix = DocTermMatrix(dtm=dtm, all_terms=all_phrases, verbose=verbose, quant_name="count")
    assert all(len([i for i in descriptions._descriptions if term in i]) == len([i for i in doc_term_matrix.term_quants(term) if i > 0]) for term in random.sample(all_phrases, 5))
    #TODO why do I even need to filter this uhm err
    if verbose:
        print("The 25 terms that are most often detected as candidate terms (incl. their #detections):",
              ", ".join(f"{k} ({v})" for k, v in sorted(dict(Counter(flatten(postprocessed_candidates))).items(), key=lambda x: x[1], reverse=True)[:25]))
    return doc_term_matrix


def filter_keyphrases(doc_cand_matrix, descriptions, min_term_count, dcm_quant_measure, verbose=False, use_n_docs_count=True):
    """name is a bit wrong, this first filters the doc-keyphrase-matrix and afterwards applies a quantification-measure on the dcm"""
    assert len(doc_cand_matrix.dtm) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions._descriptions) for cand in doc_cand_matrix.terms_per_doc()[ndesc])
    assert all(i[1] > 0 for doc in doc_cand_matrix.dtm for i in doc)
    filtered_dcm = DocTermMatrix.filter(doc_cand_matrix, min_count=min_term_count, use_n_docs_count=use_n_docs_count, verbose=verbose, descriptions=descriptions)
    #TODO: drop those documents without any keyphrase?!
    assert all([n for n, i in enumerate(descriptions._descriptions) if term in i] == [n for n, i in enumerate(filtered_dcm.term_quants(term)) if i > 0] for term in random.sample(list(filtered_dcm.all_terms.values()), 5))
    filtered_dcm = filtered_dcm.apply_quant(dcm_quant_measure)
    return filtered_dcm
