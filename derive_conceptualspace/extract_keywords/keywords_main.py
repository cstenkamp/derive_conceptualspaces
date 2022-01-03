from collections import Counter
from os.path import join, basename
import logging

from tqdm import tqdm
from gensim import corpora
import numpy as np

from .get_candidates_keybert import KeyBertExtractor
from .get_candidates_rules import extract_coursetype

from derive_conceptualspace.util.text_tools import tf_idf, get_stopwords, ppmi
from derive_conceptualspace.util.mpl_tools import show_hist


from misc_util.pretty_print import pretty_print as print
from ..settings import get_setting
from ..util.dtm_object import DocTermMatrix

logger = logging.getLogger(basename(__file__))

flatten = lambda l: [item for sublist in l for item in sublist]

########################################################################################################################
########################################################################################################################
########################################################################################################################

def extract_candidateterms(pp_descriptions, extraction_method, faster_keybert=False, verbose=False):
    assert not (extraction_method in ["keybert", "pp_keybert"] and "2" in get_setting("PP_COMPONENTS"))
    if extraction_method.lower() == "keybert":
        candidateterms, metainf = extract_candidateterms_keybert_nopp(pp_descriptions, faster_keybert, verbose=verbose)
    elif extraction_method.lower() == "pp_keybert":
        candidateterms, metainf = extract_candidateterms_keybert_preprocessed(pp_descriptions, faster_keybert, verbose=verbose)
    elif "".join([i for i in extraction_method.lower() if i.isalpha()]) == "tfidf":
        candidateterms, metainf = extract_candidateterms_quantific(pp_descriptions, quantific="tfidf", verbose=verbose)
    elif extraction_method.lower() == "ppmi":
        candidateterms, metainf = extract_candidateterms_quantific(pp_descriptions, quantific="ppmi", verbose=verbose)
    else:
        raise NotImplementedError()
    return candidateterms, metainf

def extract_candidateterms_keybert_nopp(pp_descriptions, faster_keybert=False, verbose=False):
    vocab, descriptions = pp_descriptions.values()
    extractor = KeyBertExtractor(False, faster=faster_keybert)
    candidateterms = []
    n_immediateworking_ges, n_fixed_ges, n_errs_ges = 0, 0, 0
    for desc in tqdm(descriptions):
        keyberts, origextracts, (n_immediateworking, n_fixed, n_errs) = extractor(desc.text, desc.lang)
        #TODO maybe the best post-processing for all candidates is to run the exact processing I ran for the descriptions for all the results...?!
        # Because theoretically afterwards 100% of them should be in the processed_text
        if (ct := extract_coursetype(desc)) and ct not in keyberts:
            keyberts += [ct]
        candidateterms.append(keyberts)
        n_immediateworking_ges += n_immediateworking
        n_fixed_ges += n_fixed
        n_errs_ges += n_errs
    if verbose:
        print(f"Immediately working: {n_immediateworking_ges}")
        print(f"Fixed: {n_fixed_ges}")
        print(f"Errors: {n_errs_ges}")
    return candidateterms, {"keybertextractor_modelname": extractor.model_name}


def extract_candidateterms_keybert_preprocessed(pp_descriptions, faster_keybert=False, verbose=False):
    vocab, descriptions = pp_descriptions.values()
    from keybert import KeyBERT  # lazily loaded as it needs tensorflow which takes some time to init
    model_name = "paraphrase-MiniLM-L6-v2" if faster_keybert else "paraphrase-mpnet-base-v2"
    print(f"Using model {model_name}")
    candidateterms = []
    kw_model = KeyBERT(model_name)
    for desc in tqdm(descriptions):
        stopwords = get_stopwords(desc.lang)
        candidates = set()
        for nwords in range(1, 4):
            txt = ". ".join([" ".join(sent) for sent in desc.processed_text])
            n_candidates = kw_model.extract_keywords(txt, keyphrase_ngram_range=(1, nwords), stop_words=stopwords)
            candidates |= set(i[0] for i in n_candidates)
        candidates = list(candidates)
        if (ct := extract_coursetype(desc)) and ct not in candidates:
            candidates += [ct]
        candidateterms.append(candidates)
    return candidateterms, {"keybertextractor_modelname": model_name}

#TODO put the parameters here in settings.py
#TODO play around with the parameters here!
def extract_candidateterms_quantific(pp_descriptions, quantific, max_per_doc_abs = 10, max_per_doc_rel = 0.1, min_val = None, min_val_percentile = 0.8, min_per_doc = 2, forcetake_percentile = 0.98, verbose=False):
    assert not (min_val and min_val_percentile)
    vocab, descriptions = pp_descriptions.values()
    dtm = DocTermMatrix(all_terms=vocab, descriptions=descriptions)
    if quantific == "tfidf":
        quant = tf_idf(dtm, verbose=verbose, descriptions=descriptions)
    elif quantific == "ppmi":
        quant = ppmi(dtm, verbose=verbose, descriptions=descriptions)
    else:
        raise NotImplementedError()
    if min_val_percentile:
        min_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), min_val_percentile * 100)
        metainf = dict(kw_max_per_doc_abs=max_per_doc_abs, kw_max_per_doc_rel=max_per_doc_rel, kw_min_val_percentile=min_val_percentile, kw_min_per_doc=min_per_doc, kw_forcetake_percentile=forcetake_percentile)
    else:
        metainf = dict(kw_max_per_doc_abs=max_per_doc_abs, kw_max_per_doc_rel=max_per_doc_rel, kw_min_val=min_val, kw_min_per_doc=min_per_doc, kw_forcetake_percentile=forcetake_percentile)
    forcetake_val = np.percentile(np.array(flatten([[j[1] for j in i] for i in quant])), forcetake_percentile * 100)
    candidates = [ [ sorted(i, key=lambda x:x[1], reverse=True),
                     min(round(len(i)*max_per_doc_rel), max_per_doc_abs),
                     set(j[0] for j in i if j[1] >= forcetake_val)]
                 for i in quant]
    all_candidates = [set([k[0] for k in i[0][:min_per_doc]])|set([j[0] for j in i[0] if j[1] >= min_val][:i[1]])|i[2] for i in candidates]
    return [[dtm.all_terms[j] for j in i] for i in all_candidates], metainf


########################################################################################################################
########################################################################################################################
########################################################################################################################
#TODO kann ich diese gesamte methode mit create_dissim_mat vermischen?!
def filter_keyphrases(doc_cand_matrix, pp_descriptions, min_term_count, dcm_quant_measure, verbose=False, use_n_docs_count=True):
    """name is a bit wrong, this first filters the doc-keyphrase-matrix and afterwards applies a quantification-measure on the dcm"""
    _, descriptions = pp_descriptions.values()
    assert len(doc_cand_matrix.dtm) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions) for cand in doc_cand_matrix.terms_per_doc()[ndesc])
    assert all(i[1] > 0 for doc in doc_cand_matrix.dtm for i in doc)
    if use_n_docs_count:
        occurences = [set(i[0] for i in doc) for doc in doc_cand_matrix.dtm]
        term_counts = {term: sum([term in i for i in occurences]) for term in doc_cand_matrix.all_terms.keys()}
    else:
        flat_terms = [flatten([[i[0]] * i[1] for i in doc]) for doc in doc_cand_matrix.dtm]
        term_counts = Counter(flatten(flat_terms))
    used_terms = {k: v for k, v in term_counts.items() if v >= min_term_count}
    if verbose:
        print(f"Using only terms that occur "+(f"in at least {min_term_count} documents" if use_n_docs_count else f"at least {min_term_count} times")+f", which are {len(used_terms)} of {len(term_counts)} terms.")
        most_used = sorted(list(used_terms.items()), key=lambda x: x[1], reverse=True)[:10]
        print("The most used terms are: " + ", ".join([f"{doc_cand_matrix.all_terms[ind]} ({count})" for ind, count in most_used]))
        show_hist(list(used_terms.values()), "Occurences per Keyword", xlabel="Occurences per Keyword", cutoff_percentile=93)
    doc_term_matrix = [[[doc_cand_matrix.all_terms[ind], num] for ind, num in doc] for doc in doc_cand_matrix.dtm]
    all_terms = {doc_cand_matrix.all_terms[elem]: i for i, elem in enumerate(used_terms.keys())}; del used_terms
    doc_term_matrix = [[[all_terms[ind], num] for ind, num in doc if ind in all_terms] for doc in doc_term_matrix]
    assert set(i[0] for doc in doc_term_matrix for i in doc) == set(all_terms.values())
    all_terms = {v: k for k, v in all_terms.items()}
    assert all(all_terms[ind] in descriptions[ndoc] for ndoc, doc in enumerate(tqdm(doc_term_matrix)) for ind, count in doc)
    if verbose:
        print("Documents without any keyphrase:", [descriptions[i] for i, e in enumerate(doc_term_matrix) if len(e) < 1][:5])
        print("Documents with just 1 keyphrase:", [[descriptions[i], all_terms[e[0][0]]] for i, e in enumerate(doc_term_matrix) if len(e) == 1][:5])
    #TODO: drop those documents without any keyphrase?!
    filtered_dcm = DocTermMatrix(dict(doc_term_matrix=doc_term_matrix, all_terms=all_terms))
    if dcm_quant_measure == "binary":
        filtered_dcm = DocTermMatrix(dict(doc_term_matrix=[[[ind, 1 if count >= 1 else 0] for ind, count in doc] for doc in filtered_dcm.dtm], all_terms=all_terms))
    elif dcm_quant_measure == "tf-idf":
        filtered_dcm = DocTermMatrix(dict(doc_term_matrix=tf_idf(filtered_dcm, verbose=verbose, descriptions=descriptions), all_terms=all_terms))
    elif dcm_quant_measure == "count":
        pass
    elif dcm_quant_measure == "ppmi":
        DocTermMatrix(dict(doc_term_matrix=ppmi(filtered_dcm, verbose=verbose, descriptions=descriptions), all_terms=all_terms))
    else:
        assert False
    return filtered_dcm


def create_doc_cand_matrix(postprocessed_candidates, pp_descriptions, verbose=False):
    vocab, descriptions = pp_descriptions.values()
    postprocessed_candidates, = postprocessed_candidates.values()
    assert len(postprocessed_candidates) == len(descriptions)
    assert all(cand in desc for ndesc, desc in enumerate(descriptions) for cand in postprocessed_candidates[ndesc])
    all_phrases = list(set(flatten(postprocessed_candidates)))
    # descriptions = [tokenize_text(i)[1] for i in mds_obj.descriptions]
    # if I used gensim for this, it would be `dictionary,doc_term_matrix = corpora.Dictionary(descriptions), [dictionary.doc2bow(doc) for doc in descriptions]`
    dictionary = corpora.Dictionary([all_phrases])
    dtm = [sorted([(nphrase, desc.count_phrase(phrase)) for nphrase, phrase in enumerate(all_phrases) if phrase in desc], key=lambda x:x[0]) for ndesc, desc in enumerate(tqdm(descriptions))]
    doc_term_matrix = DocTermMatrix(all_phrases=all_phrases, dtm=dtm, descriptions=descriptions, verbose=verbose)
    if verbose:
        print("The 25 terms that are most often detected as candidate terms (incl. their #detections):",
              ", ".join(f"{k} ({v})" for k, v in sorted(dict(Counter(flatten(postprocessed_candidates))).items(), key=lambda x: x[1], reverse=True)[:25]))
    return doc_term_matrix