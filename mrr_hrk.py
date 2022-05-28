import os
import numpy as np


def obtPopuPreT(city,isSource,names):
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

def obtAgePreT(city,isSource,names):
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

def obtIncPreT(city,isSource,names):
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
    obtainResults=np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(popu[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput

def obtainSimByAge(modzctaLocs, age, tarPopuList):
    obtainResults=np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(age[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput


def obtainSimByInc(modzctaLocs, age, tarPopuList):
    obtainResults=np.zeros(len(tarPopuList))
    for idx in range(0, len(tarPopuList)):
        obtainResults[idx] = abs(age[1] - tarPopuList[idx][1])

    idxResults = sorted(enumerate(obtainResults), key=lambda x: x[1])

    fOutput = []
    for idx in range(0, len(idxResults)):
        fOutput.append(modzctaLocs[idxResults[idx][0]])
    return fOutput


def obtRealPre(city,isSource,names):
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
        idx = idx +1

    return trendList

def obtRealSimilar(cityTrends, tarCityTrends, key):
    sourceList = cityTrends[key]
    similarityList = {}
    for key in tarCityTrends.keys():
        tarList = tarCityTrends[key]
        maxSimilarity = -10000
        for idx in range(0, len(tarList)-len(sourceList)):
            tmpTargetList = tarList[idx:idx+len(sourceList)]
            similarity = 0
            for innerIdx in range(len(sourceList)):
                similarity += 1 - (
                        abs(float(sourceList[innerIdx]) - float(tmpTargetList[innerIdx])) /
                        (abs(float(sourceList[innerIdx])) + abs(float(tmpTargetList[innerIdx])) + 1e-12)
                )
            if similarity>maxSimilarity:
                maxSimilarity = similarity
        similarityList[key]=maxSimilarity
    sortedList = sorted(similarityList.items(), key=lambda kv: (kv[1], kv[0]), reverse= True)
    for idx in range(0, len(sortedList)):
        sortedList[idx]=sortedList[idx][0]
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
        path = './{}/svsm_{}{}.txt'.format(city,city,tarCity)
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
        path = './{}/triplet_{}{}.txt'.format(city,city,tarCity)
    trendList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            trendList.append(items)

    return trendList[idx]

def obtainSimByOur(city, tarCity, key,modelDir):
    path = './{}/names.txt'.format(city)
    names = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            names.append(line)
    idx = names.index(key)
    path = './{}/casestudy_{}{}{}.txt'.format(city, city,tarCity,modelDir)
    trendList = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            items = line.replace('\n', '').split('\t')
            trendList.append(items)

    return trendList[idx]

def evalSFNYCMRR(city, tarCity, modelDir):
    sumAccPop = 0
    sumAccAge = 0
    sumAccInc = 0
    sumAccSVSM = 0
    sumAccTriplet = 0
    sumAccOur = 0
    casePath = './{}/casestudy_{}{}{}.txt'.format(city,city,tarCity,modelDir)
    if not os.path.exists(casePath):
        return
    names=[]
    targetNames=[]
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

    cityTrends = obtRealPre(city,True,names)
    tarCityTrends = obtRealPre(tarCity,False,targetNames)
    sourceModzctaLocs, sourcePopoList = obtPopuPreT(city,True,names)

    _, sourceAgeList = obtAgePreT(city,True,names)
    _, sourceIncList = obtIncPreT(city,True,names)

    tarModzctaLocs, tarPopoList = obtPopuPreT(tarCity,False,targetNames)
    _, tarAgeList = obtAgePreT(tarCity,False,targetNames)
    _, tarIncList = obtIncPreT(tarCity,False,targetNames)
    with open(casePath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            key = names[idx]
            innerIdx = sourceModzctaLocs.index(key)
            outputReal = obtRealSimilar(cityTrends, tarCityTrends, key)
            outputOur = obtainSimByOur(city, tarCity, key,modelDir)

            tmpOur = 1+outputReal.index(outputOur[0])
            sumAccOur = sumAccOur + 1/tmpOur

            outputPop = obtainSimByPopu(tarModzctaLocs, sourcePopoList[innerIdx],tarPopoList)
            outputAge = obtainSimByAge(tarModzctaLocs, sourceAgeList[innerIdx],tarAgeList)
            outputInc = obtainSimByInc(tarModzctaLocs, sourceIncList[innerIdx], tarIncList)
            outputSVSM = obtainSimBySVSM(city,tarCity,key)
            outputTriplet = obtainSimByTriplet(city,tarCity,key)

            tmpPop = 1+outputReal.index(outputPop[0])
            sumAccPop = sumAccPop + 1 / tmpPop
            tmpAge = 1+outputReal.index(outputAge[0])
            sumAccAge = sumAccAge + 1 / tmpAge
            tmpInc = 1+outputReal.index(outputInc[0])
            sumAccInc = sumAccInc + 1 / tmpInc
            tmpSVSM = 1+outputReal.index(outputSVSM[0])
            sumAccSVSM = sumAccSVSM + 1 / tmpSVSM
            tmpTriplet = 1+outputReal.index(outputTriplet[0])
            sumAccTriplet = sumAccTriplet + 1 / tmpTriplet

    if city == 'nyc':
        num = 0
    else:
        num = 1
    print('MRR Acc by Popu:{}'.format(sumAccPop/(len(cityTrends)-num)))
    print('MRR Acc by Age:{}'.format(sumAccAge/(len(cityTrends)-num)))
    print('MRR Acc by Inc:{}'.format(sumAccInc/(len(cityTrends)-num)))
    print('MRR Acc by SVSM:{}'.format(sumAccSVSM/(len(cityTrends)-num)))
    print('MRR Acc by Triplet:{}'.format(sumAccTriplet/(len(cityTrends)-num)))
    print('MRR Acc by Our:{}'.format(sumAccOur/(len(cityTrends)-num)))

def getPosition(tarIncList,key):
    sortedList = sorted(tarIncList, key=lambda kv: (kv[1], kv[0]), reverse= True)
    for idx in range(0, len(sortedList)):
        sortedList[idx] = sortedList[idx][0]

    return sortedList.index(key)

def evalSFNYCTopK(city, tarCity,K,modelDir):
    sumAccPop = 0
    sumAccAge = 0
    sumAccSVSM = 0
    sumAccTriplet = 0
    sumAccOur = 0
    sumAccInc = 0
    casePath = './{}/casestudy_{}{}{}.txt'.format(city,city,tarCity,modelDir)
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
    cityTrends = obtRealPre(city,True, names)
    tarCityTrends = obtRealPre(tarCity,False,targetNames)
    sourceModzctaLocs, sourcePopoList = obtPopuPreT(city,True, names)

    _, sourceAgeList = obtAgePreT(city,True, names)
    _, sourceIncList = obtIncPreT(city,True, names)

    tarModzctaLocs, tarPopoList = obtPopuPreT(tarCity,False,targetNames)
    _, tarAgeList = obtAgePreT(tarCity,False,targetNames)
    _, tarIncList = obtIncPreT(tarCity,False,targetNames)
    with open(casePath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f.readlines()):
            key = names[idx]
            innerIdx = sourceModzctaLocs.index(key)
            outputReal = obtRealSimilar(cityTrends, tarCityTrends, key)
            outputOur = obtainSimByOur(city, tarCity, key,modelDir)

            outputSVSM = obtainSimBySVSM(city, tarCity,key)
            outputTriplet = obtainSimByTriplet(city, tarCity,key)
            outputPop = obtainSimByPopu(tarModzctaLocs, sourcePopoList[innerIdx], tarPopoList)
            outputAge = obtainSimByAge(tarModzctaLocs, sourceAgeList[innerIdx], tarAgeList)
            outputInc = obtainSimByInc(tarModzctaLocs, sourceIncList[innerIdx], tarIncList)

            tmpPop = outputReal.index(outputPop[0])
            if tmpPop < K:
                incIdx = getPosition(sourcePopoList, key)
                sumAccPop = sumAccPop + 1
            tmpAge = outputReal.index(outputAge[0])
            if tmpAge < K:
                sumAccAge = sumAccAge + 1
                incIdx = getPosition(sourceIncList,key)
            tmpInc = outputReal.index(outputInc[0])
            if tmpInc < K :
                incIdx = getPosition(sourceIncList,key)
                sumAccInc = sumAccInc + 1
            tmpSVSM = outputReal.index(outputSVSM[0])
            if tmpSVSM < K:
                sumAccSVSM = sumAccSVSM + 1


            tmpTriplet = outputReal.index(outputTriplet[0])
            if tmpTriplet < K:
                sumAccTriplet = sumAccTriplet + 1
            tmpOur = outputReal.index(outputOur[0])
            if tmpOur < K:
                sumAccOur = sumAccOur + 1

    if city == 'nyc':
        num = 0
    else:
        num = 0
    print('Top{} Acc by Popu:{}'.format(K-1, sumAccPop / (len(cityTrends)-num)))
    print('Top{} Acc by Age:{}'.format(K-1, sumAccAge / (len(cityTrends)-num)))
    print('Top{} Acc by Inc:{}'.format(K-1, sumAccInc / (len(cityTrends)-num)))
    print('Top{} Acc by SVSM:{}'.format(K-1, sumAccSVSM / (len(cityTrends)-num)))
    print('Top{} Acc by Triplet:{}'.format(K-1, sumAccTriplet / (len(cityTrends)-num)))
    print('Top{} Acc by Our:{}'.format(K-1, sumAccOur / (len(cityTrends)-num)))

def main():
    city = 'nyc'
    tarCity = 'sf'
    modelDir = '_inner'
    evalSFNYCMRR(city, tarCity,modelDir)
    evalSFNYCTopK(city, tarCity,4,modelDir)
    evalSFNYCTopK(city, tarCity,6,modelDir)
    evalSFNYCTopK(city, tarCity,11,modelDir)


if __name__ == "__main__":
    main()
