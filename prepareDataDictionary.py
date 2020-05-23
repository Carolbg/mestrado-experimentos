# Data science tools
import numpy as np
import glob
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
from plots import plotTransformedImages
from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray 
from preprocessing import preprocessDictionaryDataset
from utilsParams import getCommonArgs

def mainPrepareDictionaryData():
    shuffleSeed, batch_size, max_epochs_stop, n_epochs = getCommonArgs()
    saudaveisDictionaryData, doentesDictionaryData = mainReadData()
    filteredSaudaveisDicData, filteredDoentesDicData, topMean = preprocessDictionaryDataset(saudaveisDictionaryData, doentesDictionaryData)
    trainData, trainTarget, testData, testTarget, validationData, validationTarget = splitData(shuffleSeed, filteredSaudaveisDicData, filteredDoentesDicData)

    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDatasetBalancedData(trainData, trainTarget, testData, testTarget, validationData, validationTarget, batch_size, topMean)
    return trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs

def mainReadData():
    print('\nprepareDataFromTXT')
    txt_saudaveis_files, txt_doentes_files= getFilesName()
    saudaveisDictionaryData = readFilesByPatient(txt_saudaveis_files)
    doentesDictionaryData = readFilesByPatient(txt_doentes_files)
    return saudaveisDictionaryData, doentesDictionaryData

def getFilesName():
    print('getFilesName')
    txt_saudaveis_files = glob.glob("../Imagens_TXT_Estaticas_Balanceadas/0Saudavel/*.txt")
    txt_doentes_files = glob.glob("../Imagens_TXT_Estaticas_Balanceadas/1Doente/*.txt")
    
    return txt_saudaveis_files, txt_doentes_files

def readFilesByPatient(txt_files):
    print('readFiles')
    data = []
    dataAsDictionary = {}
    for i in range(len(txt_files)):
        name = txt_files[i].split('/')
        fileName = name[len(name)-1]
        patientId = fileName.split('.')[0]
        inputData = np.loadtxt(txt_files[i], dtype='f', delimiter=' ')
        if patientId in dataAsDictionary.keys(): 
            dataAsDictionary[patientId].append(inputData)
        else:
            dataAsDictionary[patientId] = []
            dataAsDictionary[patientId].append(inputData)
    return dataAsDictionary

def splitData(shuffleSeed, saudaveisData, doentesData):
    print('\nSplit Healthy Dataset')
    saudaveisIndTra, saudaveisIndTeste, saudaveisIndValid = splitPatientsFromDictionary(shuffleSeed, saudaveisData)
    saudaveisTrainDataset, saudaveisTestDataset, saudaveisValidationDataset = prepareDatasetFromDictionary(saudaveisData, saudaveisIndTra, saudaveisIndTeste, saudaveisIndValid)
    print('\nSplit Cancer Dataset')
    doentesIndTra, doentesIndTeste, doentesIndValid = splitPatientsFromDictionary(shuffleSeed, doentesData)
    doentesTrainDataset, doentesTestDataset, doentesValidationDataset = prepareDatasetFromDictionary(doentesData, doentesIndTra, doentesIndTeste, doentesIndValid)
    trainData, trainTarget = createSplitDataset(shuffleSeed, saudaveisTrainDataset, doentesTrainDataset)
    print('\nTotal de dados para treinamento', len(trainData))
    testData, testTarget = createSplitDataset(shuffleSeed, saudaveisTestDataset, doentesTestDataset)
    print('\nTotal de dados para treinamento', len(testData))
    validationData, validationTarget = createSplitDataset(shuffleSeed, saudaveisValidationDataset, doentesValidationDataset)
    print('\nTotal de dados para treinamento', len(validationData))
    return trainData, trainTarget, testData, testTarget, validationData, validationTarget

def createSplitDataset(shuffleSeed, saudaveisDataset, doentesDataset):
    saudaveisTarget = np.full(len(saudaveisDataset), 0)
    doentesTarget = np.full(len(doentesDataset), 1)

    allData = np.concatenate((saudaveisDataset, doentesDataset), axis=0)
    allTarget = np.concatenate((saudaveisTarget, doentesTarget), axis=0)

    indicesValidation = list(range(len(allTarget)))
    np.random.seed(shuffleSeed)
    np.random.shuffle(indicesValidation)

    allData = allData[indicesValidation]
    allTarget = allTarget[indicesValidation]

    return allData, allTarget

def prepareDatasetFromDictionary(dictionaryData, indicesTreinamento, indicesTeste, indicesValidacao):
    dictKeys = dictionaryData.keys()
    keysArray = np.array(list(dictKeys))
    trainPatients = keysArray[indicesTreinamento]
    testPatients = keysArray[indicesTeste]
    validationPatients = keysArray[indicesValidacao] 
    print('trainPatients', trainPatients)
    print('testPatients', testPatients)
    print('validationPatients', validationPatients) 
    
    trainDataset = []
    for patient in trainPatients:
        images = dictionaryData[patient]
        trainDataset.extend(images)
    print('imagens do trainDataset', len(trainDataset))

    testDataset = []
    for patient in testPatients:
        images = dictionaryData[patient]
        testDataset.extend(images)
    print('imagens do testDataset', len(testDataset))

    validationDataset = []
    for patient in validationPatients:
        images = dictionaryData[patient]
        validationDataset.extend(images)
    print('imagens do validationDataset', len(validationDataset))

    train, test, validation = np.array(trainDataset), np.array(testDataset), np.array(validationDataset)
    
    print('train', train.shape)
    print('test', test.shape)
    print('validation', validation.shape)
    return train, test, validation

def splitPatientsFromDictionary(shuffleSeed, dictionaryData):
    print('Total dados', len(dictionaryData))
    totalPatientsDataset = len(dictionaryData.keys())
    trainPatientsDataset = math.floor(totalPatientsDataset*0.70)
    testPatientsDataset = math.floor((totalPatientsDataset - trainPatientsDataset)/2)
    #validationPatientsTotal = totalPatientsDataset - trainPatientsDataset - testPatientsDataset

    indicesDataset = list(range(totalPatientsDataset))
    np.random.seed(shuffleSeed)
    np.random.shuffle(indicesDataset)

    indicesTreinamento = indicesDataset[:trainPatientsDataset] 
    indicesTeste = indicesDataset[trainPatientsDataset:trainPatientsDataset+testPatientsDataset]
    indicesValidacao = indicesDataset[trainPatientsDataset+testPatientsDataset:]
    
    print('Quantidade de dados para treinamento',len(indicesTreinamento))
    print('Quantidade de dados para teste',len(indicesTeste))
    print('Quantidade de dados para validacao',len(indicesValidacao))

    return indicesTreinamento, indicesTeste, indicesValidacao


def prepareNumpyDatasetBalancedData(dataTrain, dataTargetTrain, dataTest, dataTargetTest, dataValidation, dataTargetValidation, batch_size, topMean):
    print('prepareNumpyDatasetBalancedData')
    dataTrain = dataTrain/topMean
    dataTest = dataTest/topMean
    dataValidation = dataValidation/topMean

    trainTransform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomRotation(degrees=30, fill=(0,)),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards  # Imagenet standards
    ])
    
    testValidationTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        #transforms.Lambda(lambda x: torch.cat([x/topMean, x/topMean, x/topMean], 0)),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(dataTrain,dataTargetTrain, testValidationTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    testDataset = CustomDatasetFromNumpyArray(dataTest,dataTargetTest, testValidationTransform)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

    validationDataset = CustomDatasetFromNumpyArray(dataValidation, dataTargetValidation, testValidationTransform)
    validationLoader = DataLoader(validationDataset, batch_size=batch_size, shuffle=True)

    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        #plotTransformedImages(images, i, 'testes_balanced_train_')
        i = i+1

    i=0
    resultLabelsTesting = torch.zeros(2, dtype=torch.long)
    for images, labels in iter(testLoader):
        l = labels.numpy()
        resultLabelsTesting[0] = resultLabelsTesting[0] + np.count_nonzero(l == 0)
        resultLabelsTesting[1] = resultLabelsTesting[1] + np.count_nonzero(l == 1)
        #plotTransformedImages(images, i, 'testes_balanced_test_')
        i = i+1

    resultLabelsValidation = torch.zeros(2, dtype=torch.long)
    i = 0
    for images, labels in iter(validationLoader):
        l = labels.numpy()
        resultLabelsValidation[0] = resultLabelsValidation[0] + np.count_nonzero(l == 0)
        resultLabelsValidation[1] = resultLabelsValidation[1] + np.count_nonzero(l == 1)
        #plotTransformedImages(images, i, 'testes_balanced_validation_')
        i = i+1

    cat_df = pd.DataFrame({
        'category': ['Saudável', 'Doente'],
        'Treinamento': resultLabelsTraining,
        'Validação': resultLabelsValidation, 
        'Teste': resultLabelsTesting
    })
    print(cat_df)

    n_classes = len(cat_df)
    return trainLoader, testLoader, validationLoader, n_classes, cat_df
