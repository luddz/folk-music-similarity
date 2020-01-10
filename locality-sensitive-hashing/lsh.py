"""
Gererates a list of LSH buckets.
"""
def lsh(sm, b, rows):
    buckets = []
    for i in range(b):
        bucket = {}
        for j in range(len(sm[0])):
            # Get the column sample of size rows
            row = ""
            for s in range(rows):
                row += str(sm[rows*i + s][j])
            # Put it in the hashmap
            hashed = hash(row)
            if hashed in bucket:
                bucket[hashed].append(j)
            else:
                bucket[hashed] = [j]
        buckets.append(bucket)
    return buckets


"""
Finds all candidate pairs. Two documents are considered cancidate pairs iff that exists at least
one bucket with the candidate columns from the both documents.
"""
def findCandidatePairs(buckets):
    candidatePairs = []
    for bucket in buckets:
        for _, value in bucket.items():
            if len(value) > 1:
                candidatePairs.append(value)

    return candidatePairs

    