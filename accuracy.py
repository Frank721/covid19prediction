import math
import os

import numpy
import numpy as np


def obtPopuPreT(city, isSource, names):
    modzctaLocs = []
    trends = []
    trendSet = []
    if isSource:
        path = './{}/trends/caserate-by-modzcta_increase.csv'.format(city)
    else:
        path = './{}/trends/caserate-by-modzcta.csv'.format(city)
    popuMatrix = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n' or i == 0:
                items = line.replace('\n', '').split(',')
                for item in items:
                    modzctaLocs.append(item)
                    trends.append([])
            else:
                items = line.replace('\n', '').split(',')
                for idx, item in enumerate(items):
                    trends[idx].append((item))
        idx = 0
        while idx < len(modzctaLocs):
            if (city == 'nyc' and (
                    modzctaLocs[idx] in ['week_ending', 'CASERATE_CITY', 'CASERATE_BX', 'CASERATE_BK', 'CASERATE_MN',
                                         'CASERATE_QN',
                                         'CASERATE_SI'] or modzctaLocs[idx].split('_')[1] not in names)) or (
                    city == 'sf' and modzctaLocs[idx] not in names):
                modzctaLocs.pop(idx)
                trends.pop(idx)
                continue
            modzctaLocs[idx] = modzctaLocs[idx].replace('CASERATE_', '')
            trendSet.append(trends[idx])
            popuMatrix.append([modzctaLocs[idx], 0])
            idx = idx + 1

    path = './{}/region.txt'.format(city)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            for idx, tar in enumerate(popuMatrix):
                if tar[0] == items[0]:
                    popuMatrix[idx] = [items[0], float(items[4])]
                    break

    return modzctaLocs, popuMatrix


def obtAgePreT(city, isSource, names):
    modzctaLocs = []
    trends = []
    trendSet = []
    if isSource:
        path = './{}/trends/caserate-by-modzcta_increase.csv'.format(city)
    else:
        path = './{}/trends/caserate-by-modzcta.csv'.format(city)
    ageMatrix = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n' or i == 0:
                items = line.replace('\n', '').split(',')
                for item in items:
                    modzctaLocs.append(item)
                    trends.append([])
            else:
                items = line.replace('\n', '').split(',')
                for idx, item in enumerate(items):
                    trends[idx].append((item))
        idx = 0
        while idx < len(modzctaLocs):
            if (city == 'nyc' and (
                    modzctaLocs[idx] in ['week_ending', 'CASERATE_CITY', 'CASERATE_BX', 'CASERATE_BK', 'CASERATE_MN',
                                         'CASERATE_QN',
                                         'CASERATE_SI'] or modzctaLocs[idx].split('_')[1] not in names)) or (
                    city == 'sf' and modzctaLocs[idx] not in names):
                modzctaLocs.pop(idx)
                trends.pop(idx)
                continue
            modzctaLocs[idx] = modzctaLocs[idx].replace('CASERATE_', '')
            trendSet.append(trends[idx])
            ageMatrix.append([modzctaLocs[idx], -10000])
            idx = idx + 1

    path = './{}/avgAge.txt'.format(city)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split(' ')
            for idx, tar in enumerate(ageMatrix):
                if tar[0] == items[0]:
                    ageMatrix[idx] = [items[0], float(items[1])]
                    break

    return modzctaLocs, ageMatrix


def obtIncPreT(city, isSource, names):
    modzctaLocs = []
    trends = []
    trendSet = []
    if isSource:
        path = './{}/trends/caserate-by-modzcta_increase.csv'.format(city)
    else:
        path = './{}/trends/caserate-by-modzcta.csv'.format(city)
    ageMatrix = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n' or i == 0:
                items = line.replace('\n', '').split(',')
                for item in items:
                    modzctaLocs.append(item)
                    trends.append([])
            else:
                items = line.replace('\n', '').split(',')
                for idx, item in enumerate(items):
                    trends[idx].append((item))
        idx = 0
        while idx < len(modzctaLocs):
            if (city == 'nyc' and (
                    modzctaLocs[idx] in ['week_ending', 'CASERATE_CITY', 'CASERATE_BX', 'CASERATE_BK', 'CASERATE_MN',
                                         'CASERATE_QN',
                                         'CASERATE_SI'] or modzctaLocs[idx].split('_')[1] not in names)) or (
                    city == 'sf' and modzctaLocs[idx] not in names):
                modzctaLocs.pop(idx)
                trends.pop(idx)
                continue
            modzctaLocs[idx] = modzctaLocs[idx].replace('CASERATE_', '')
            trendSet.append(trends[idx])
            ageMatrix.append([modzctaLocs[idx], -10000])
            idx = idx + 1

    path = './{}/income.txt'.format(city)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split(' ')
            for idx, tar in enumerate(ageMatrix):
                if tar[0] == items[0]:
                    ageMatrix[idx] = [items[0], float(items[1])]
                    break

    return modzctaLocs, ageMatrix


def obtainSimByPopu(modzctaLocs, popu, tarPopuList):
    obtainResults = np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(popu[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput


def obtainSimByAge(modzctaLocs, age, tarPopuList):
    obtainResults = np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(age[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput


def obtainSimByInc(modzctaLocs, age, tarPopuList):
    obtainResults = np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(age[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput


def obtRealPre(city, isSource, names):
    modzctaLocs = []
    trends = []
    trendSet = []
    if isSource:
        path = './{}/trends/caserate-by-modzcta_increase.csv'.format(city)
    else:
        path = './{}/trends/caserate-by-modzcta.csv'.format(city)
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            if line == '\n' or i == 0:
                items = line.replace('\n', '').split(',')
                for item in items:
                    modzctaLocs.append(item)
                    trends.append([])
            else:
                items = line.replace('\n', '').split(',')
                for idx, item in enumerate(items):
                    trends[idx].append((item))
        idx = 0
        while idx < len(modzctaLocs):
            if (city == 'nyc' and (
                    modzctaLocs[idx] in ['week_ending', 'CASERATE_CITY', 'CASERATE_BX', 'CASERATE_BK', 'CASERATE_MN',
                                         'CASERATE_QN',
                                         'CASERATE_SI'] or modzctaLocs[idx].split('_')[1] not in names)) or (
                    city == 'sf' and modzctaLocs[idx] not in names):
                modzctaLocs.pop(idx)
                trends.pop(idx)
                continue
            modzctaLocs[idx] = modzctaLocs[idx].replace('CASERATE_', '')
            # size = modzcatSizes[modzctaLocs[idx]]
            # for innerIdx,item in enumerate(trends[idx]):
            #    trends[idx][innerIdx] = 1000000*float(item)/float(size)
            trendSet.append(trends[idx])
            idx = idx + 1
    trendList = {}
    idx = 0
    while idx < len(modzctaLocs):
        trendList[modzctaLocs[idx]] = trendSet[idx]
        idx = idx + 1

    return trendList


def obtRealSimilar(cityTrends, tarCityTrends, key):
    sourceList = cityTrends[key]
    similarityList = {}
    for key in tarCityTrends.keys():
        tarList = tarCityTrends[key]
        maxSimilarity = -10000
        for idx in range(0, len(tarList) - len(sourceList)):
            tmpTargetList = tarList[idx:idx + len(sourceList)]
            similarity = 0
            for innerIdx in range(len(sourceList)):
                similarity += 1 - (
                        abs(float(sourceList[innerIdx]) - float(tmpTargetList[innerIdx])) /
                        (abs(float(sourceList[innerIdx])) + abs(float(tmpTargetList[innerIdx])) + 1e-12)
                )
            if similarity > maxSimilarity:
                maxSimilarity = similarity
        similarityList[key] = maxSimilarity
    sortedList = sorted(similarityList.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    for idx in range(0, len(sortedList)):
        sortedList[idx] = sortedList[idx][0]
    return sortedList


def obtainSimBySVSM(city, tarCity, key):
    path = './{}/names.txt'.format(city)
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            names.append(line)
    idx = names.index(key)
    if city == tarCity:
        path = './{}/svsm_{}.txt'.format(city, city)
    else:
        path = './{}/svsm_{}{}.txt'.format(city, city, tarCity)
    trendList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            trendList.append(items)

    return trendList[idx]


def obtainSimByTriplet(city, tarCity, key):
    path = './{}/names.txt'.format(city)
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            names.append(line)
    idx = names.index(key)
    if city == tarCity:
        path = './{}/triplet_{}.txt'.format(city, city)
    else:
        path = './{}/triplet_{}{}.txt'.format(city, city, tarCity)
    trendList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            trendList.append(items)

    return trendList[idx]


def obtainSimByOur(city, tarCity, key, modelDir):
    path = './{}/names.txt'.format(city)
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            names.append(line)
    idx = names.index(key)
    path = './{}/casestudy_{}{}{}.txt'.format(city, city, tarCity, modelDir)
    trendList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            trendList.append(items)

    return trendList[idx]


def getPosition(tarIncList, key):
    sortedList = sorted(tarIncList, key=lambda kv: (kv[1], kv[0]), reverse=True)
    for idx in range(0, len(sortedList)):
        sortedList[idx] = sortedList[idx][0]

    return sortedList.index(key)


def dtw_distance(ts_a, ts_b, d=lambda x, y: abs(x - y), mww=10000):
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = numpy.array(ts_a), numpy.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = numpy.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(float(ts_a[0]), float(ts_b[0]))
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(float(ts_a[i]), float(ts_b[0]))

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(float(ts_a[0]), float(ts_b[j]))

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - mww), min(N, i + mww)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(float(ts_a[i]), float(ts_b[j]))

    # Return DTW distance given window
    return cost[-1, -1], cost


def vector_distance(ts_a, ts_b):
    sim = 0
    totalValue = 0
    vectorValue = 0
    for i in range(len(ts_b)):
        totalValue = totalValue + abs(float(ts_a[i]))
        vectorValue = vectorValue + abs(float(ts_b[i]) - float(ts_a[i]))
        if abs(float(ts_b[i]) + float(ts_a[i])) == 0.0:
            sim += 1
        else:
            sim += 1 - abs(float(ts_b[i]) - float(ts_a[i])) / abs(float(ts_b[i]) + float(ts_a[i]))
    if sim == 0.0:
        return 1
    return sim / len(ts_b)
    # return max(1-vectorValue/totalValue,0)


def obtSimilarity(cityTrends, names, tarCityTrends, targetNames):
    trendSimilarity = numpy.zeros([len(cityTrends), len(tarCityTrends)])

    for i in range(0, len(trendSimilarity)):
        for j in range(0, len(trendSimilarity[i])):
            leng = len(cityTrends.get(names[i]))
            a = cityTrends.get(names[i])
            b = tarCityTrends.get(targetNames[j])[0:leng]
            similarity = vector_distance(cityTrends.get(names[i]), tarCityTrends.get(targetNames[j])[0:leng])
            trendSimilarity[i][j] = similarity

    return trendSimilarity


def evalSFNYCTopK(city, tarCity, modelDir):
    sumAccPop = 0
    sumAccAge = 0
    sumAccSVSM = 0
    sumAccTriplet = 0
    sumAccOur = 0
    sumAccInc = 0
    casePath = './{}/casestudy_{}{}{}.txt'.format(city, city, tarCity, modelDir)
    if not os.path.exists(casePath):
        return
    names = []
    targetNames = []
    path = './{}/names.txt'.format(city)
    idx = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            idx = idx + 1
            item = line.replace('\n', '')
            names.append(item)
    path = './{}/names.txt'.format(tarCity)
    idx = 0
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            idx = idx + 1
            item = line.replace('\n', '')
            targetNames.append(item)
    cityTrends = obtRealPre(city, True, names)
    tarCityTrends = obtRealPre(tarCity, False, targetNames)
    trendSimilarity = obtSimilarity(cityTrends, names, tarCityTrends, targetNames)
    sourceModzctaLocs, sourcePopoList = obtPopuPreT(city, True, names)

    _, sourceAgeList = obtAgePreT(city, True, names)
    _, sourceIncList = obtIncPreT(city, True, names)

    tarModzctaLocs, tarPopoList = obtPopuPreT(tarCity, False, targetNames)
    _, tarAgeList = obtAgePreT(tarCity, False, targetNames)
    _, tarIncList = obtIncPreT(tarCity, False, targetNames)
    with open(casePath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            key = names[idx]
            innerIdx = names.index(key)
            outputOur = obtainSimByOur(city, tarCity, key, modelDir)

            outputSVSM = obtainSimBySVSM(city, tarCity, key)
            outputTriplet = obtainSimByTriplet(city, tarCity, key)
            outputPop = obtainSimByPopu(tarModzctaLocs, sourcePopoList[innerIdx], tarPopoList)
            outputAge = obtainSimByAge(tarModzctaLocs, sourceAgeList[innerIdx], tarAgeList)
            outputInc = obtainSimByInc(tarModzctaLocs, sourceIncList[innerIdx], tarIncList)

            sumAccPop = sumAccPop + trendSimilarity[innerIdx, targetNames.index(outputPop[0])]
            sumAccAge = sumAccAge + trendSimilarity[innerIdx, targetNames.index(outputAge[0])]
            sumAccInc = sumAccInc + trendSimilarity[innerIdx, targetNames.index(outputInc[0])]
            sumAccSVSM = sumAccSVSM + trendSimilarity[innerIdx, targetNames.index(outputSVSM[0])]
            sumAccTriplet = sumAccTriplet + trendSimilarity[innerIdx, targetNames.index(outputTriplet[0])]

            sumAccOur = sumAccOur + trendSimilarity[innerIdx, targetNames.index(outputOur[0])]

    print('Acc by Popu:{}'.format(sumAccPop / (len(cityTrends))))
    print('Acc by Age:{}'.format(sumAccAge / (len(cityTrends))))
    print('Acc by Inc:{}'.format(sumAccInc / (len(cityTrends))))
    print('Acc by SVSM:{}'.format(sumAccSVSM / (len(cityTrends))))
    print('Acc by Triplet:{}'.format(sumAccTriplet / (len(cityTrends))))
    print('Acc by Our:{}'.format(sumAccOur / (len(cityTrends))))


def main():
    city = 'nyc'
    tarCity = 'sf'
    modelDir = '_inner'
    evalSFNYCTopK(city, tarCity, modelDir)


if __name__ == "__main__":
    main()
