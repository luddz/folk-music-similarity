from readData import *
from song import *
from fastRNN import *
import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import random
from datetime import datetime

def genMeanVar(dataset, n=10):
    means = []
    variance = []
    for data in dataset[:n]:
        mean, var = meanVar(data)
        means.append(mean)
        variance.append(var)

    return means, variance

def meanVar(data):
    song = "M:"+data["meter"]+" K:"+data["key"]+" "+data["transcription"]
    probMatrix = throughRNN(song)
    return np.mean(probMatrix, axis=0).flatten(), np.var(probMatrix, axis=0).flatten()

def euclideanJsonSet(dataset1, dataset2, filePath):
    meanData1, variancesData1 = genMeanVar(dataset1, len(dataset1))
    meanData2, variancesData2 = genMeanVar(dataset2, len(dataset2))

    result = []
    for i in range(len(meanData1)):
        for j in range(len(meanData2)):
            tmp = dict()
            meanDist = cosine(meanData1[i], meanData2[j])
            varDist = cosine(variancesData1[i], variancesData2[j])
            tmp["meanDist"] = str(meanDist)
            tmp["varDist"] = str(varDist)
            tmp["id1"] = dataset1[i]["id"]
            tmp["id2"] = dataset2[j]["id"]
            result.append(tmp)

    writeResults(result, filePath)
    return result

def sortOnMean(result):
    res = sorted(result, key=lambda k: k['meanDist'])
    return sorted(res, key=lambda k: k['id1'])

def sortOnVariane(result):
    res = sorted(result, key=lambda k: k['varDist'])
    return sorted(res, key=lambda k: k['id1'])


def euclideanJsonOnID(id1=0, id2=0):
    dataset1 = readJsonFile("../dataset/training-data.json")
    dataset2 = readJsonFile("../dataset/output-data.json")
    meanId1, varId1 = meanVar(dataset1[id1])
    meanId2, varId2 = meanVar(dataset2[id2])
    meanDist = np.linalg.norm(meanId1 - meanId2)
    varDist = np.linalg.norm(varId1 - varId2)
    print "ID1:",id1," ID2:",id2
    print "Mean:", meanDist
    print "Variance:", varDist
    return

def genMIDIConvert(id=0):
    # data = readJsonFile("../dataset/training-data.json")
    data = readJsonFile("../dataset/output-data.json")
    song = data[id]
    print "X:1"
    print "T: id", song["id"]
    print "M:", song["meter"]
    print "K:", song["key"]
    print str(song["transcription"])

def genNormData(dataset, name, training):
    short = []
    for data in dataset:
        if len(data["transcription"].replace(" ", "")) < 200:
            if training:
                data["id"] = 30000 + data["id"]
            short.append(data)
            continue

    result = random.sample(short, 100)

    writeResults(result,name+"-100-songs")

    return

def writeResults(results, fileName):
    time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    with open(fileName+"-"+time+".json", "w") as f:
        json.dump(results, f, indent=4)

    return

def resultsReport():
    # training100 = readJsonFile("../finalDataset/training-100-songs-2019-11-28_11:50:12.json")
    # output100 = readJsonFile("../finalDataset/output-100-songs-2019-11-28_11:50:12.json")
    combined = readJsonFile("../finalDataset/combined-200.json")
    testData = readJsonFile("../finalDataset/testset_cleaned.json")
    testName = "test-vs-combined"
    filePath = "finalResults/"+testName

    res = euclideanJsonSet(testData, combined, filePath)
    meanTop10 = sortOnMean(res)
    varTop10 = sortOnVariane(res)
    writeResults(meanTop10, "finalResults/Mean-"+testName+"-sorted")
    writeResults(varTop10, "finalResults/Variance-"+testName+"-sorted")
    return

def gen100SongDataset():
    trainingDataset = readJsonFile("../dataset/training-data.json")
    outputDataset = readJsonFile("../dataset/output-data.json")
    genNormData(trainingDataset,"training",True)
    genNormData(outputDataset,"output",False)
    return

if __name__ == '__main__':
    # gen100SongDataset()
    resultsReport()
    # genData, orgData = getSongs()
    # euclideanJsonOnID(9, 5)
    # genMIDIConvert(5)
    # trainingDataset = readJsonFile("../dataset/training-data.json")
    # outputDataset = readJsonFile("../dataset/output-data.json")
    # euclideanJsonSet(trainingDataset, trainingDataset)
    # euclideanTest(genData)
