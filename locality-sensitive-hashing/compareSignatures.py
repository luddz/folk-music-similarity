"""
Returns similarity between documents with index s1 and s2 by comparing their signature.
"""
def compareSignatures(sm, s1, s2):
    matchingValues = 0
    for i in range(len(sm)):
        if sm[i][s1] == sm[i][s2]:
            matchingValues += 1

    return matchingValues / (len(sm) * 1.0)
