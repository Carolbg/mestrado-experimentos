# Data science tools
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from plots import plotFilteredImage

def applyMedianFilter(image):
    filteredImage = ndimage.median_filter(image, size=5)
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

def getMaxValue(dataset):
    flattenDataset = dataset.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    topValues = flattenDataset[indices]
    topMean = np.mean(topValues)
    print('topMean', topMean)
    
    # indices = np.argpartition(flattenDataset, -100)[-100:] 
    # top100 = flattenDataset[indices]
    # top100mean = np.mean(top100)
    return topMean#, top100mean

def preprocessImage(saudaveisDataset, doentesDataset):
    filteredSaudaveisDataset = applyMedianFilterDataset(saudaveisDataset, 'saudaveis')
    print('filteredSaudaveisDataset', filteredSaudaveisDataset.shape)
    filteredDoentesDataset = applyMedianFilterDataset(doentesDataset,'doentes')
    print('filteredDoentesDataset', filteredDoentesDataset.shape)
    filteredDataset = np.concatenate((filteredSaudaveisDataset, filteredDoentesDataset), axis=0)
    top10mean = getMaxValue(filteredDataset)
    return filteredSaudaveisDataset, filteredDoentesDataset, top10mean
