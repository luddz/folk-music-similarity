from shingling import constructShingles
from compareSets import jaccardSimilarity
from compareSignatures import compareSignatures
from minHashing import minHash
from lsh import lsh, findCandidatePairs
import re
import os
import sys
import time
import json


shingleLength = 4
numHashFunctions = 100
threshold = 0.5

#Change these to where you have the data stored
dataFolder = 'data/'
filename = 'file'

"""
Calculates and returns all document pairs that have similarity greater then the given threshold.
The result comes in the following format: (index of first doc, index of second doc, similarity)
"""
def findSimilarDocuments(sm, thres, unProcessedDocuments):
    similarDocuments = []
    for i in range(len(sm[0])):
        for j in range(i + 1, len(sm[0])):
            similarity = compareSignatures(sm, i, j)
            if similarity >= thres:
                similarDocuments.append((unProcessedDocuments[i][0],unProcessedDocuments[j][0],similarity))
    return similarDocuments

"""
As findSimilarDocuments but only compares the first element in the filw with all the other files.
"""
def findSimilarDocumentsCompareToOne(sm, thres, unProcessedDocuments):
    similarDocuments = []
    for i in range(len(sm[0])):
        similarity = compareSignatures(sm, i, 0)
        if similarity >= thres:
            similarDocuments.append((unProcessedDocuments[i][0],unProcessedDocuments[0][0],similarity))
    return similarDocuments    

"""
Find optimal b and row parameters depending on the number of hash functions and the threshold
"""
def findLSHParameters(n, threshold):
        currentBestPair = None
        currentBestDiff = 1
        for i in range(1, n//2):
                if n % i == 0:
                   currentValue = (1/(n/i))**(1/i)
                   if abs(currentValue - threshold) < currentBestDiff:
                        currentBestDiff = abs(currentValue - threshold)
                        currentBestPair = (i,n//i)
        return currentBestPair

def readDataJson():
    unProcessedDocuments = []
    with open(dataFolder + filename) as json_file:
        data = json.load(json_file)
        for p in data:
            document = [p['id'],p['transcription']]
            unProcessedDocuments.append(document)
    return unProcessedDocuments  
                     
def takeThird(elem):
    return elem[2]                           

if __name__ == '__main__':
    start = time.time()
    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    (row, b) = findLSHParameters(numHashFunctions, threshold)
    
    # Get the data
    documents = []
    unProcessedDocuments = readDataJson()
    for i in range(0, len(unProcessedDocuments)):
        documents.append(constructShingles(unProcessedDocuments[i][1], shingleLength))
    
    # Generate signature matrix
    print(len(documents))
    sm = minHash(documents, numHashFunctions)
    
    # Get candidate pairs
    candidatePairs = findCandidatePairs(lsh(sm, b, row))
    
    # Get similar pairs by comparing all signatures
    similarDocuments = findSimilarDocumentsCompareToOne(sm, threshold, unProcessedDocuments)

    printElements = similarDocuments
    printElements.sort(key=takeThird, reverse=True) 

    for doc in printElements:
        print(doc)

    print(len(similarDocuments))
    allSimilarDocuments = []
    for doc in similarDocuments:
        docDic = dict()
        docDic = {"id1": doc[0], "id2": doc[1], "similarity": doc[2]}
        allSimilarDocuments.append(docDic)
    with open("data_file.json", "w") as write_file:
        json.dump(allSimilarDocuments, write_file)
    end = time.time()
    print(end-start)
