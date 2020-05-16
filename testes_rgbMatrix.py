# Data science tools
import numpy as np
import glob
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
from plots import plotTransformedImages
from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray, CustomTesteDatasetFromNumpyArray

from preprocessing import applyMedianFilterDataset

def main():
    print('Load dataset')
    data, dataDivided, dataTarget, dataRGB, dataRGBDivided = prepareDataFromTXT()
    shuffleSeed, batch_size, max_epochs_stop, n_epochs = getCommonArgs()
    train_idx = splitDataset(data, shuffleSeed)
    
    applyMedianFilterDataset(data)

    return data

    #prepareNumpyDataset(data, dataTarget, train_idx, batch_size, 'data_sem_dividir_')
    #prepareNumpyDividedDataset(dataDivided, dataTarget, train_idx, batch_size, 'data_com_divisao_')

    #prepareNumpyRGBDataset(dataRGB, dataTarget, train_idx, batch_size, 'data_alreadyRGB_')
    #prepareNumpyRGBDividedDataset(dataRGBDivided, dataTarget, train_idx, batch_size, 'data_alreadyRGB_divided')

def getFilesName():
    print('getFilesName')
    #txt_saudaveis_files = glob.glob("../Imagens_TXT_Estaticas/0Saudavel/*.txt")
    #txt_doentes_files = glob.glob("../Imagens_TXT_Estaticas/1Doente/*.txt")
    txt_saudaveis_files = glob.glob("../poucas_Imagens/0Saudavel/*.txt")
    txt_doentes_files = glob.glob("../poucas_Imagens/1Doente/*.txt")

    return txt_saudaveis_files, txt_doentes_files

def readFiles(txt_files):
    print('readFiles')
    data = []
    dataDivided = []
    dataRGB = []
    dataRGBDivided = []
    for i in range(len(txt_files)):
        inputData = np.loadtxt(txt_files[i], dtype='f', delimiter=' ')
        #print('inputData', inputData.shape)
        data.append(inputData)
        dataDivided.append(inputData/255)
        rgbinputData = np.stack((inputData,)*3, axis=-1)
        #print('rgbinputData1', rgbinputData.shape)
        rgbinputData = np.transpose(rgbinputData, (2, 0, 1))
        #print('rgbinputData 2', rgbinputData.shape)
        #input()
        dataRGB.append(rgbinputData)
        dataRGBDivided.append(rgbinputData/255)
    return data, dataDivided, dataRGB, dataRGBDivided

def prepareDataFromTXT():
    print('prepareDataFromTXT')
    txt_saudaveis_files, txt_doentes_files= getFilesName()
    saudaveisData, saudaveisDataDivided, saudaveisDataRGB, saudaveisDataRGBDivided = readFiles(txt_saudaveis_files)
    #doentesData, doentesDataDivided, doentesDataRGB, doentesDataRGBDivided = readFiles(txt_doentes_files)
    saudaveisTarget = np.full(len(saudaveisData), 0)
    #doentesTarget = np.full(len(doentesData), 1)
    data = np.concatenate((saudaveisData,saudaveisData), axis=0)
    dataDivided = np.concatenate((saudaveisDataDivided,saudaveisDataDivided), axis=0)
    dataRGB = np.concatenate((saudaveisDataRGB,saudaveisDataRGB), axis=0)
    dataRGBDivided = np.concatenate((saudaveisDataRGBDivided,saudaveisDataRGBDivided), axis=0)
    dataTarget = np.concatenate((saudaveisTarget, saudaveisTarget), axis=0)
    return data, dataDivided, dataTarget, dataRGB, dataRGBDivided

def splitDataset(dataset, shuffleSeed):
    print('splitDataset\n')
    totalDataset = len(dataset)

    trainTotal = math.ceil(totalDataset*0.70)
    testTotal = math.ceil((totalDataset - trainTotal)/2)
    validationTotal = totalDataset - trainTotal - testTotal

    indices = list(range(totalDataset))

    np.random.seed(shuffleSeed)
    np.random.shuffle(indices)

    train_idx = indices[:totalDataset]
    print('Quantidade de dados para treinamento',len(train_idx))
    

    return train_idx

def prepareNumpyDividedDataset(data, dataTarget, train_idx, batch_size, saveName):
    print('prepareNumpyDividedDataset')

    defaultTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Imagenet standards  # Imagenet standards
    ])

    defaultTransform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Imagenet standards  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    #for batch_idx, (data, target) in enumerate(trainLoader):
    #    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_sem_divisao_transformada')
        i = i+1

def prepareNumpyDataset(data, dataTarget, train_idx, batch_size, saveName):
    print('prepareNumpyDataset')

    defaultTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Imagenet standards  # Imagenet standards
    ])

    defaultTransform1 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Imagenet standards  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    testeDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform1, 1)
    testeLoader = DataLoader(testeDataset, batch_size=batch_size, shuffle=True)
    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    #for batch_idx, (data, target) in enumerate(trainLoader):
    #    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_sem_divisao_transformada')
        i = i+1
    
    i = 0
    for images, labels in iter(testeLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_com_divisao_transformada')
        i = i+1
    #print('Treinamento resultLabels', resultLabelsTraining)

def prepareNumpyRGBDataset(data, dataTarget, train_idx, batch_size, saveName):
    print('prepareNumpyRGBDataset')
    print('teste = ',data.shape)

    defaultTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()  # Imagenet standards  # Imagenet standards
    ])

    defaultTransform1 = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.Resize(size=256),
        transforms.ToTensor()  # Imagenet standards  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    train1Dataset = CustomTesteDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform1)
    train1Loader = DataLoader(train1Dataset, batch_size=batch_size, shuffle=True)

    testeDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform, 1)
    testeLoader = DataLoader(testeDataset, batch_size=batch_size, shuffle=True)
    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    #for batch_idx, (data, target) in enumerate(trainLoader):
    #    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_com_PIL_transformada_sem_divisao')
        i = i+1
    
    i = 0
    for images, labels in iter(train1Loader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_transformada_vazia')
        i = i+1

    for images, labels in iter(testeLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_com_PIL_transformada_com_divisao')
        i = i+1
    #print('Treinamento resultLabels', resultLabelsTraining)


def prepareNumpyRGBDividedDataset(data, dataTarget, train_idx, batch_size, saveName):
    print('prepareNumpyRGBDividedDataset')
    print('teste = ',data.shape)

    defaultTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()  # Imagenet standards  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    #for batch_idx, (data, target) in enumerate(trainLoader):
    #    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, saveName+'_rgb_sem_divisao_transformada')
        i = i+1


def getCommonArgs():
    shuffleSeed = 3
    batch_size = 10
    max_epochs_stop = 30
    n_epochs = 30
    print('n_epochs', n_epochs)
    return shuffleSeed, batch_size, max_epochs_stop, n_epochs