# Data science tools
import numpy as np
import glob
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
from plots import plotTransformedImages, plotTestTransformedImages, plotAllSubsetImages
from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray 
from utilsParams import getCommonArgs
from skimage import transform
import cv2
from prepareDataDictionary import prepareNumpyDatasetBalancedData, splitData

def mainPrepareDictionaryDataFromNumpy(dataAugmentation):
    print('Lidando com numpy data')
    shuffleSeed, batch_size, max_epochs_stop, n_epochs, device = getCommonArgs()
    saudaveisDictionaryData, doentesDictionaryData = mainReadNumpyData()
    
    trainData, trainTarget, testData, testTarget, validationData, validationTarget = splitData(shuffleSeed, saudaveisDictionaryData, doentesDictionaryData)

    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDatasetBalancedData(trainData, trainTarget, testData, testTarget, validationData, validationTarget, batch_size, dataAugmentation)
    return trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device

def mainReadNumpyData():
    print('\nPrepareDataFromNumpy arrays')
    numpy_saudaveis_files, numpy_doentes_files = getFilesName()
    saudaveisDictionaryData = readFilesByPatient(numpy_saudaveis_files, 'saudaveis')
    doentesDictionaryData = readFilesByPatient(numpy_doentes_files, 'doentes')
    return saudaveisDictionaryData, doentesDictionaryData

def getFilesName():
    print('getFilesName')
    # If reading from the script
    # numpy_saudaveis_files = glob.glob("../../Imagens_numpy_array_allData/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../Imagens_numpy_array_allData/1Doentes/*.npy")
    
    # numpy_saudaveis_files = glob.glob("../../Imagens_numpy_array_allData_semAumentoDados/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../Imagens_numpy_array_allData_semAumentoDados/1Doentes/*.npy")
    
    # numpy_saudaveis_files = glob.glob("../../Imagens_numpy_array_allData_asMinMax/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../Imagens_numpy_array_allData_asMinMax/1Doentes/*.npy")
    # folder = "Imagens_numpy_array_allData_semAumentoDados"
    folder="Imagens_numpy_array_allData_entireDatabase_MinMax"
    # folder = "Imagens_numpy_array_allData_entireDatabase_MinMax_double"
    print(folder)
    numpy_saudaveis_files = glob.glob("../../../"+ folder+"/0Saudaveis/*.npy")
    numpy_doentes_files = glob.glob("../../../"+ folder+"/1Doentes/*.npy")

    
    #If not reading from the script
    # numpy_saudaveis_files = glob.glob("../Imagens_numpy_array/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../Imagens_numpy_array/1Doentes/*.npy")
    
    return numpy_saudaveis_files, numpy_doentes_files

def readFilesByPatient(numpy_files_name, patientClass):
    print('readFilesByPatient',patientClass)
    dataAsDictionary = {}
    for i in range(len(numpy_files_name)):
        name = numpy_files_name[i].split('/')
        fileName = name[len(name)-1]
        patientId = fileName.split('.')[0]
        inputData = np.load(numpy_files_name[i])
        #print('inputData.shape', inputData.shape)
        # print('1 - min', np.min(inputData))
        # print('1 - max', np.max(inputData))
        inputData = np.transpose(inputData, (2, 0, 1))
        # print('2 - min', np.min(inputData))
        # print('2 - max', np.max(inputData))
        #print('inputData.shape', inputData.shape)
        #isso porque o tensor tem formato C, H, W
       
        
        if patientId in dataAsDictionary.keys(): 
            dataAsDictionary[patientId].append(inputData)
        else:
            dataAsDictionary[patientId] = []
            dataAsDictionary[patientId].append(inputData)
    return dataAsDictionary
