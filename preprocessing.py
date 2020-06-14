# Data science tools
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from plots import plotFilteredImage

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

def getMaxMinValue(dataset):
    flattenDataset = dataset.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    #print('indices', indices)
    topValues = flattenDataset[indices]
    #print('topValues', topValues)

    indices = np.argpartition(flattenDataset, 100)[:100]
    #print('indices', indices)
    minValuesValues = flattenDataset[indices]
    #print('minValuesValues', minValuesValues)
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
    return filteredSaudaveisDicData, filteredDoentesDicData,top10mean #deltaT
