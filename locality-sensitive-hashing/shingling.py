"""
Construct set of hashed shingles av size k from the document.
"""
def constructShingles(document, k):
    shingles = set()
    for i in range(0, len(document) - k + 1):
        shingles.add(hash(document[i : i + k]) % 2**32)
    return shingles

