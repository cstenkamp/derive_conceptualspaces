from src.static.settings import COURSE_TYPES

def extract_coursetype(desc, coursetypes=None):
    coursetypes = coursetypes or COURSE_TYPES
    for type in coursetypes:
        if any(i in desc.lower() for i in [f"this {type}"]):
            return type
    counts = {i: desc.lower().count(i) for i in coursetypes}
    if any(i > 0 for i in counts.values()):
        return max(counts.items(), key=lambda x:x[1])[0]
    return None

def extract_tfidf():
    #TODO: implement this, using src.main.util.text_tools.tf_idf
    raise NotImplementedError()