import numpy as np

from main.load_semanticspaces import load_ppmi_weighted_feature_vectors, load_mds_representation, get_names
from static.settings import DATA_BASE, DATA_SET, MDS_DIMENSIONS
from main.measures import simple_similiarity, between_a, between_b_inv

def main():
    betweenness_measure = between_a #between_b_inv
    tmp = load_mds_representation(DATA_BASE, "places", MDS_DIMENSIONS)
    tmp2 = get_names(DATA_BASE, "places")
    name_mds = dict(zip(tmp2, tmp))
    paper_candidates = [
        ("fast food restaurant", "french restaurant", "american restaurant"),
        ("restaurant space", "tea room", "bistro"),
        ("marketplace", "slaughterhouse", "butcher shop"),
        ("coffee shop", "restaurant", "cafe"),
        ("bakery", "fast food restaurant", "deli"),
    ]
    for first, second, third in paper_candidates:
        print(f"Question: Is `{third}` between `{first}` and `{second}`?")
        a, b = name_mds[first], name_mds[second]
        scores = {}
        for name, candidate in name_mds.items():
            if any(candidate != a) and any(candidate != b):
                scores[name] = betweenness_measure(a, b, candidate)
        valid_scores = [val for val in scores.values() if val < np.inf]
        print("Mean Betweenness:", sum(valid_scores)/len(valid_scores), "| their Candidate betweeness:", scores[third])
        top5_score = max(sorted(valid_scores)[:5])
        top5_names = [name for name, score in scores.items() if score <= top5_score]
        print("Top Between Candidates:", ", ".join(top5_names))
        print(f"Their Candidate is betweenness place:", f"{sum(1 for i in valid_scores if i < scores[third])}/{len(scores)}")
        print()

if __name__ == '__main__':
    main()
