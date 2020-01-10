"""
Returns the Jaccard similarity.
"""
def jaccardSimilarity(s1, s2):
    return len(s1.intersection(s2)) / (len(s1.union(s2)) * 1.0)
