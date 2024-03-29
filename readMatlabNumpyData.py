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
from prepareDataDictionary import prepareNumpyDatasetBalancedData, splitData, prepareImage
from preprocessing import getMeanStdEntireBase, getMaxMinValueFromDataDic
import gc
import torch
from cnn_visualization import *

def mainPrepareDictionaryDataFromNumpy(dataAugmentation, nEpochs):
    print('Lidando com numpy data')
    shuffleSeed, batch_size, max_epochs_stop, n_epochs, device = getCommonArgs(nEpochs)
    saudaveisDictionaryData, doentesDictionaryData = mainReadNumpyData()
    print('len(saudaveisDictionaryData)', len(saudaveisDictionaryData))
    print('len(doentesDictionaryData)', len(doentesDictionaryData))
    gc.collect()
    torch.cuda.empty_cache()
    
    mean, std = getMeanStdEntireBase(saudaveisDictionaryData, doentesDictionaryData)
    # mean = 0.4516 #73378329850344
    # std = 0.4363 #97180299452515

    trainData, trainTarget, testData, testTarget, validationData, validationTarget = splitData(shuffleSeed, saudaveisDictionaryData, doentesDictionaryData)
    gc.collect()
    torch.cuda.empty_cache()

    setInfoData(trainData, trainTarget, testData, testTarget, validationData, validationTarget)

    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDatasetBalancedData(trainData, trainTarget, testData, testTarget, validationData, validationTarget, batch_size, dataAugmentation, mean, std)
    
    gc.collect()
    torch.cuda.empty_cache()
    
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
    # numpy_saudaveis_files = glob.glob("../../../Imagens_numpy_array_allData/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../../Imagens_numpy_array_allData/1Doentes/*.npy")
    
    # numpy_saudaveis_files = glob.glob("../../../Imagens_numpy_array_allData_semAumentoDados/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../../Imagens_numpy_array_allData_semAumentoDados/1Doentes/*.npy")
    
    # numpy_saudaveis_files = glob.glob("../../../Imagens_numpy_array_allData_asMinMax/0Saudaveis/*.npy")
    # numpy_doentes_files = glob.glob("../../../Imagens_numpy_array_allData_asMinMax/1Doentes/*.npy")
    # folder = "Imagens_numpy_array_allData_semAumentoDados"
    
    # folder="Imagens_numpy_array_allData_entireDatabase_MinMax"
    # folder = "Imagens_numpy_array_allData_entireDatabase_MinMax_extrapolandoLimites"
    # folder = "Imagens_numpy_array_allData_entireDatabase_MinMax_double"
    # folder = "Imagens_numpy_array_allData_rgb"
    # folder = "Imagens_numpy_array_allData_rgb_double"
    
    # folder='Imagens_numpy_array_asCabıoglu_rgb'
    # print(folder)
    # numpy_saudaveis_files = sorted(glob.glob("../../../"+ folder+"/0Saudavel/*.npy"))
    # numpy_doentes_files = sorted(glob.glob("../../../"+ folder+"/1Doente/*.npy"))

    #GDRIVE RUNNING
    # Imagens_numpy_array_asCabıoglu_rgb_aumentoDados
    # Imagens_TXT_Estaticas_Balanceadas_allData_asCabıoglu_DA
    
    #dados para cabioglu aumentando so doentes - grupo 2
    folder='/content/gdrive/My Drive/MestradoCodes/Imagens_numpy_array_asCabıoglu_rgb_aumentoDados'
    
    # folder='/content/gdrive/My Drive/MestradoCodes/ImagensROI_Adib'
    
    #dados para cabioglu aumentando so doentes - grupo 3
    # folder='/content/gdrive/My Drive/MestradoCodes/Imagens_numpy_array_asCabıoglu_rgb_aumentoDadosDoentes'
    
    # folder='/content/gdrive/My Drive/MestradoCodes/Imagens_TXT_Estaticas_Balanceadas_frontalImages_Ratio_Matlab'
    print(folder)
    numpy_saudaveis_files = sorted(glob.glob(folder+"/0Saudavel/*.npy"))
    numpy_doentes_files = sorted(glob.glob(folder+"/1Doente/*.npy"))

    # numpy_saudaveis_files = sorted(glob.glob(folder+"/0SaudavelNPY/*.npy"))
    # numpy_doentes_files = sorted(glob.glob(folder+"/1DoenteNPY/*.npy"))

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
        # print('1 original input shape -> inputData.shape', inputData.shape)
        # print('1 - min', np.min(inputData))
        # print('1 - max', np.max(inputData))
        inputData = np.transpose(inputData, (2, 0, 1))
        # print('2 - min', np.min(inputData))
        # print('2 - max', np.max(inputData))
        # print('2 after transpose inputData.shape', inputData.shape)
        #isso porque o tensor tem formato C, H, W
       
        
        if patientId in dataAsDictionary.keys(): 
            dataAsDictionary[patientId].append(inputData)
        else:
            dataAsDictionary[patientId] = []
            dataAsDictionary[patientId].append(inputData)
    return dataAsDictionary

