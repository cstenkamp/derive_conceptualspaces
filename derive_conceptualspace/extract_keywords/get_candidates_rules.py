from derive_conceptualspace.settings import get_setting

def extract_coursetype(desc, coursetypes=None):
    coursetypes = coursetypes or get_setting("COURSE_TYPES")
    for type in coursetypes:
        if any(i in desc.unprocessed_text.lower() for i in [f"this {type}"]):
            return type
    counts = {i: desc.bow().get(i, 0) for i in coursetypes}
    if any(i > 0 for i in counts.values()):
        return max(counts.items(), key=lambda x:x[1])[0]
    counts = {i: desc.unprocessed_text.lower().count(i) for i in coursetypes}
    if any(i > 0 for i in counts.values()):
        return max(counts.items(), key=lambda x:x[1])[0]
    return None

