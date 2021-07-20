coursetypes = ["colloquium", "seminar", "internship", "practice", "lecture"]

def extract_coursetype(desc):
    for type in coursetypes:
        if any(i in desc.lower() for i in [f"this {type}"]):
            return type
    counts = {i: desc.lower().count(i) for i in coursetypes}
    if any(i > 0 for i in counts.values()):
        return max(counts.items(), key=lambda x:x[1])[0]
    return None

def extract_tfidf():
    raise NotImplementedError()