# Data science tools
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from plots import plotFilteredImage
from math import sqrt
import gc
import torch
from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray 

def applyMedianFilter(image):
    filteredImage = ndimage.median_filter(image, size=3)
    return filteredImage

def applyMedianFilterDataset(dataset, nameFile):
    dataItens= dataset.shape
    filteredDataset = []
    for i in range(dataItens[0]):
        filteredImage = applyMedianFilter(dataset[i])
        filteredDataset.append(filteredImage)
        if i < 25:
           plotFilteredImage(dataset[i], filteredImage, nameFile+str(i))
    return np.array(filteredDataset)

def createAllDataList(saudaveis, doentes):
    allData=[]
    for patientId in saudaveis:
        # Para cada paciente
        values = saudaveis[patientId]
        for image in values:
            allData.append(image)
    
    for patientId in doentes:
        # Para cada paciente
        values = doentes[patientId]
        for image in values:
            allData.append(image)
    print('allData.shape', np.array(allData).shape)
    return allData

def getAllValuesDictionary(dictionaryData):
    allData=[]
    for patientId in dictionaryData:
        # Para cada paciente
        values = dictionaryData[patientId]
        for image in values:
            allData.append(image)
    return allData

def getMinMaxFromSingleDic(dictionaryData):
    dicValues = np.array(getAllValuesDictionary(dictionaryData))
    maxValue = np.max(dicValues)
    minValue = np.min(dicValues)
    
    del dicValues
    
    return [minValue, maxValue]

def getMaxMinValueFromDataDic(saudaveisDictionaryData, doentesDictionaryData):
    print('getMaxMinValueFromDataDic')

    minMaxSaudaveis = getMinMaxFromSingleDic(saudaveisDictionaryData)
    minMaxDoentes = getMinMaxFromSingleDic(doentesDictionaryData)
    minMaxValues = np.array(minMaxSaudaveis+minMaxDoentes)
    
    maxValue = np.max(minMaxValues)
    minValue = np.min(minMaxValues)

    return maxValue, minValue

def calcCount(dictionaryData):
    dicValues = np.array(getAllValuesDictionary(dictionaryData))
    shapeDim = (dicValues.flatten()).shape
    count = shapeDim[0]
    return count

def calcSumValues(dictionaryData):
    dicValues = np.array(getAllValuesDictionary(dictionaryData))
    flattedArray = (dicValues.flatten())
    del dicValues

    sumValue = np.sum(flattedArray)
    count = calcCount(dictionaryData)
    
    del flattedArray

    return [sumValue, count]

def cleanMemory():
    gc.collect()
    torch.cuda.empty_cache()

def getMean(saudaveisDictionaryData, doentesDictionaryData):
    print('getting mean')
    sumCountSaudaveis = calcSumValues(saudaveisDictionaryData)
    sumCountDoentes = calcSumValues(doentesDictionaryData)

    allSum = sumCountSaudaveis[0] + sumCountDoentes[0]
    allCount = sumCountSaudaveis[1] + sumCountDoentes[1]
    mean = allSum/allCount

    cleanMemory()

    return mean

def calcStdNumeradorSingleDic(dictionaryData, mean):
    dicValues = np.array(getAllValuesDictionary(dictionaryData))
    subtraction = dicValues - mean
    squaredSubtration = np.power(subtraction, 2)
    sumSquared = np.sum(squaredSubtration)

    del dicValues
    del subtraction
    del squaredSubtration
    # print('sumSquared', sumSquared)
    return sumSquared

def getStd(saudaveisDictionaryData, doentesDictionaryData, mean):
    print('getting std')
    
    sumSquaredSaudaveis = calcStdNumeradorSingleDic(saudaveisDictionaryData, mean)
    sumSquaredDoentes = calcStdNumeradorSingleDic(doentesDictionaryData, mean)
    numerador = sumSquaredDoentes + sumSquaredSaudaveis
    countSaudaveis = calcCount(saudaveisDictionaryData)
    countDoentes = calcCount(doentesDictionaryData)
    count = countSaudaveis+countDoentes
    
    std = sqrt(numerador/count)
    
    cleanMemory()
    
    return std


def getMeanStdEntireBase(saudaveisDictionaryData, doentesDictionaryData, soma=0):
    mean = getMean(saudaveisDictionaryData, doentesDictionaryData)
    std = getStd(saudaveisDictionaryData, doentesDictionaryData, mean)
    print('mean, std', mean, std)
    
    # allDataList = createAllDataList(saudaveisDictionaryData, doentesDictionaryData )
    # std = np.std(allDataList)
    # mean = np.mean(allDataList)
    # print('np mean', mean, 'np std', std)

    return mean, mean

def getMaxMinValue(dataset):
    flattenDataset = dataset.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    topValues = flattenDataset[indices]

    indices = np.argpartition(flattenDataset, 100)[:100]
    minValuesValues = flattenDataset[indices]
    minMean = np.mean(minValuesValues)
    print('minMean', minMean)
        
    topMean = np.mean(topValues)
    print('topMean', topMean)
    
    media = np.mean(flattenDataset)  
    print('Media base', media)
    
    desvioPadrao = np.std(flattenDataset)  
    print('Desvio padrao Base', desvioPadrao)
    
    variancia = np.var(flattenDataset)  
    print('Variancia', variancia)
    
    # indices = np.argpartition(flattenDataset, -100)[-100:] 
    # top100 = flattenDataset[indices]
    # top100mean = np.mean(top100)
    return topMean, minMean#, top100mean

def preprocessImage(saudaveisDataset, doentesDataset):
    filteredSaudaveisDataset = applyMedianFilterDataset(saudaveisDataset, 'saudaveis')
    print('filteredSaudaveisDataset', filteredSaudaveisDataset.shape)
    filteredDoentesDataset = applyMedianFilterDataset(doentesDataset,'doentes')
    print('filteredDoentesDataset', filteredDoentesDataset.shape)
    filteredDataset = np.concatenate((filteredSaudaveisDataset, filteredDoentesDataset), axis=0)
    top10mean, min10mean = getMaxMinValue(filteredDataset)
    deltaT = top10mean-min10mean
    print('top10mean', top10mean)
    print('deltaT', deltaT)
    return filteredSaudaveisDataset, filteredDoentesDataset, top10mean

def applyMedianFilterDictionaryDataset(dictionaryDataset, nameFile, allData):
    # print('teste')
    filteredDataset = {}

    for patientId in dictionaryDataset:
        # Para cada paciente
        values = dictionaryDataset[patientId]
        for image in values:
            # Para cada imagem
            filteredImage = applyMedianFilter(image)
            allData.append(filteredImage)
            if patientId in filteredDataset.keys(): 
                filteredDataset[patientId].append(filteredImage)
            else:
                filteredDataset[patientId] = []
                filteredDataset[patientId].append(filteredImage)
    return filteredDataset, allData

def preprocessDictionaryDataset(saudaveisDictionaryData, doentesDictionaryData):
    allData = []
    filteredSaudaveisDicData, allData = applyMedianFilterDictionaryDataset(saudaveisDictionaryData, 'saudaveis', allData)
    filteredDoentesDicData, allData = applyMedianFilterDictionaryDataset(doentesDictionaryData, 'doentes', allData)
    top10mean, min10mean = getMaxMinValue(np.array(allData))

    deltaT = top10mean - min10mean
    print('deltaT = ', deltaT)
    return filteredSaudaveisDicData, filteredDoentesDicData, deltaT, min10mean

    