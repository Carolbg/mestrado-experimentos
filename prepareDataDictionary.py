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
from preprocessing import preprocessDictionaryDataset, getMeanStdEntireBase, getMaxMinValueFromDataDic
from utilsParams import getCommonArgs
from skimage import transform
import cv2

def mainPrepareDictionaryData(dataAugmentation):
    print('Lidando com txt data')
    
    shuffleSeed, batch_size, max_epochs_stop, n_epochs, device = getCommonArgs()
    saudaveisRawDictionaryData, doentesRawDictionaryData = mainReadData()
    
    # filteredSaudaveisDicData, filteredDoentesDicData, deltaT, min10mean = preprocessDictionaryDataset(saudaveisDictionaryData, doentesDictionaryData)
    maxValue, minValue = getMaxMinValueFromDataDic(saudaveisRawDictionaryData, doentesRawDictionaryData)
    saudaveisDictionaryData, doentesDictionaryData = prepareImage(saudaveisRawDictionaryData, doentesRawDictionaryData, maxValue, minValue)

    mean, std = getMeanStdEntireBase(saudaveisDictionaryData, doentesDictionaryData)

    trainData, trainTarget, testData, testTarget, validationData, validationTarget = splitData(shuffleSeed, saudaveisDictionaryData, doentesDictionaryData)
    
    # trainData, testData, validationData = minMaxNormalization(trainData, testData, validationData, deltaT, min10mean)
    
    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDatasetBalancedData(trainData, trainTarget, testData, testTarget, validationData, validationTarget, batch_size, dataAugmentation, mean, std )
    return trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device

def mainReadData():
    print('\nprepareDataFromTXT')
    txt_saudaveis_files, txt_doentes_files= getFilesName()
    saudaveisRawDictionaryData = readFilesByPatient(txt_saudaveis_files, 'saudaveis')
    doentesRawDictionaryData = readFilesByPatient(txt_doentes_files, 'doentes')
    return saudaveisRawDictionaryData, doentesRawDictionaryData

def getFilesName():
    print('getFilesName')

    # folder = 'Imagens_TXT_Estaticas_Balanceadas_asCabıoglu'
    # print('folder = Imagens_TXT_Estaticas_Balanceadas_asCabıoglu')
    # txt_saudaveis_files = sorted(glob.glob("../../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/0Saudavel/*.txt"))
    # txt_doentes_files = sorted(glob.glob("../../../Imagens_TXT_Estaticas_Balanceadas_asCabıoglu/1Doente/*.txt"))
    
    #GDRIVE RUNNING
    folder='/content/gdrive/My Drive/MestradoCodes/Imagens_TXT_Estaticas_Balanceadas_asCabıoglu'
    print(folder)
    txt_saudaveis_files = sorted(glob.glob(folder+"/0Saudavel/*.txt"))
    txt_doentes_files = sorted(glob.glob(folder+"/1Doente/*.txt"))

    return txt_saudaveis_files, txt_doentes_files

def readFilesByPatient(txt_files, patientClass):
    print('readFiles')
    dataAsDictionary = {}
    for i in range(len(txt_files)):
        #print('txt_files[i]', txt_files[i])
        name = txt_files[i].split('/')
        #print('name', name)
        fileName = name[len(name)-1]
        #print('fileName', fileName)
        patientId = fileName.split('.')[0]
        inputData = np.loadtxt(txt_files[i], dtype='f', delimiter=' ')
        #print('antes inputData', inputData.shape)

        if patientId in dataAsDictionary.keys(): 
            dataAsDictionary[patientId].append(inputData)
        else:
            dataAsDictionary[patientId] = []
            dataAsDictionary[patientId].append(inputData)
    return dataAsDictionary

def prepareImage(saudaveisRawDictionaryData, doentesRawDictionaryData, maxValue, minValue):
    saudaveisDictionaryData = saudaveisRawDictionaryData.copy()
    for patientId in saudaveisRawDictionaryData:
        # Para cada paciente
        values = saudaveisRawDictionaryData[patientId]
        imagesParsed = []
        for image in values:
            inputData = preProcessingWithRatio(image, maxValue, minValue)
            inputData = convertToCorrectDimmensions(inputData)
            imagesParsed.append(inputData)
            # print('inputData', inputData.shape)

        saudaveisDictionaryData[patientId] = imagesParsed
        # print('saudaveisDictionaryData[patientId]', len(saudaveisDictionaryData[patientId]))
        # print('saudaveisRawDictionaryData[patientId]', len(saudaveisRawDictionaryData[patientId]))
    # print('size saudaveisRawDictionaryData', len(list(saudaveisRawDictionaryData.keys())), len(list(saudaveisRawDictionaryData.values())))
    # print('size saudaveisDictionaryData', len(list(saudaveisDictionaryData.keys())), len(list(saudaveisDictionaryData.values())))

    doentesDictionaryData = doentesRawDictionaryData.copy()
    for patientId in doentesRawDictionaryData:
        # Para cada paciente
        values = doentesRawDictionaryData[patientId]
        imagesParsed = []
        for image in values:
            inputData = preProcessingWithRatio(image, maxValue, minValue)
            inputData = convertToCorrectDimmensions(inputData)
            # print('inputData', inputData.shape)
            imagesParsed.append(inputData)

        doentesDictionaryData[patientId] = imagesParsed
        # print('doentesDictionaryData[patientId]', len(doentesDictionaryData[patientId]))
        # print('doentesRawDictionaryData[patientId]', len(doentesRawDictionaryData[patientId]))
    # print('size doentesRawDictionaryData', len(list(doentesRawDictionaryData.keys())), len(list(doentesRawDictionaryData.values())))
    # print('size doentesDictionaryData', len(list(doentesDictionaryData.keys())), len(list(doentesDictionaryData.values())))

    return saudaveisDictionaryData, doentesDictionaryData

def convertToCorrectDimmensions(inputData):
    #Stack the data to simulate 3d image
    inputData = np.stack((inputData,)*3, axis=2)
    # print('1 inputData', inputData.shape)
    # print('min',np.min(inputData))
    # print('max',np.max(inputData))
    
    inputData = np.transpose(inputData, (2, 0, 1))
    # print('2 inputData', inputData.shape)
    #isso porque o tensor tem formato C, H, W
    #conforme toTensor: 
    # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] 
    # to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] 
    #print('depois inputData', inputData.shape)
    return inputData

def preProcessingWithRatio(numpyImage, maxValue, minValue):
    #Converte para inteiro entre 0-255
    numpyImage = ((numpyImage-minValue) * 255 / (maxValue - minValue)).clip(0,255)
    # print('preprocessing - antes', numpyImage.shape)
    # print('min',np.min(numpyImage))
    # print('max',np.max(numpyImage))

    numpyImage = np.round(numpyImage)
    numpyImage = numpyImage.astype(np.uint8)
    # print('dp', numpyImage.shape)
    # print('min',np.min(numpyImage))
    # print('max',np.max(numpyImage))
    
    #converte para 0-1
    numpyImage = numpyImage/255
    
    return numpyImage

def splitData(shuffleSeed, saudaveisData, doentesData):

    totalPatientsDoentesDataset = len(doentesData.keys())
    numberTrainDoentes = math.floor(totalPatientsDoentesDataset*0.70)
    numberValidacaoDoentes = math.floor((totalPatientsDoentesDataset - numberTrainDoentes)/2)

    print('\nSplit Healthy Dataset')
    saudaveisIndTra, saudaveisIndTeste, saudaveisIndValid = splitPatientsFromDictionary(shuffleSeed, saudaveisData, numberTrainDoentes, numberValidacaoDoentes)
    
    # To force specific patients
    # saudaveisIndTra = ['T0189','T0196','T0193','T0220','T0199','T0217','T0188','T0224','T0216','T0211','T0259','T0194','T0200','T0239','T0236','T0272','T0201','T0226','T0195','T0221','T0238','T0237','T0234','T0275','T0222','T0261']
    # saudaveisIndTeste = ['T0218','T0233','T0208','T0190','T0225','T0177']
    # saudaveisIndValid = ['T0243','T0276','T0191','T0219','T0244','T0212']
    saudaveisTrainDataset, saudaveisTestDataset, saudaveisValidationDataset = prepareDatasetFromDictionary(saudaveisData, saudaveisIndTra, saudaveisIndTeste, saudaveisIndValid, 'saudaveis')
    
    print('\nSplit Cancer Dataset')
    doentesIndTra, doentesIndTeste, doentesIndValid = splitPatientsFromDictionary(shuffleSeed, doentesData)
    # To force specific patients
    # doentesIndTra= ['T0267','T0255','T0138','T0286','T0198','T0246','T0192','T0258','T0202','T0209','T0241','T0179','T0287','T0213','T0203','T0210','T0240','T0270','T0180','T0264','T0269','T0282','T0281','T0277','T0273','T0256']
    # doentesIndTeste=['T0257','T0278','T0285','T0268','T0283','T0271']
    # doentesIndValid=['T0266','T0245','T0263','T0260','T0181','T0204']
    doentesTrainDataset, doentesTestDataset, doentesValidationDataset = prepareDatasetFromDictionary(doentesData, doentesIndTra, doentesIndTeste, doentesIndValid, 'doentes')
    
    trainData, trainTarget = createSplitDataset(shuffleSeed, saudaveisTrainDataset, doentesTrainDataset)
    print('\nTotal de dados para treinamento', len(trainData))
    testData, testTarget = createSplitDataset(shuffleSeed, saudaveisTestDataset, doentesTestDataset)
    print('\nTotal de dados para teste', len(testData))
    validationData, validationTarget = createSplitDataset(shuffleSeed, saudaveisValidationDataset, doentesValidationDataset)
    print('\nTotal de dados para validacao', len(validationData))
    #Trocando teste e validation
    #print('Dados de teste e validacao foram invertidos')
    #return trainData, trainTarget, validationData, validationTarget, testData, testTarget
    return trainData, trainTarget, testData, testTarget, validationData, validationTarget

def createSplitDataset(shuffleSeed, saudaveisDataset, doentesDataset):
    # print('len(saudaveisDataset)', len(saudaveisDataset))
    # print('len(doentesDataset)', len(doentesDataset))
    saudaveisTarget = np.full(len(saudaveisDataset), 0)
    doentesTarget = np.full(len(doentesDataset), 1)
    # print('saudaveisTarget', saudaveisTarget)
    # print('doentesTarget', doentesTarget)

    allData = np.concatenate((saudaveisDataset, doentesDataset), axis=0)
    #print('allData', allData)
    # print('len(allData)', len(allData))
    allTarget = np.concatenate((saudaveisTarget, doentesTarget), axis=0)
    #print('allTarget', allTarget)
    # print('len(allTarget)', len(allTarget))

    indicesValidation = list(range(len(allTarget)))
    #print('indicesValidation 1', indicesValidation)
    np.random.seed(shuffleSeed)
    np.random.shuffle(indicesValidation)
    #print('indicesValidation 2', indicesValidation)

    allData = allData[indicesValidation]
    #print('allData', allData)
    #print('len(allData)', len(allData))
    allTarget = allTarget[indicesValidation]
    #print('allTarget', allTarget)
    #print('len(allTarget)', len(allTarget))

    return allData, allTarget

def prepareDatasetFromDictionary(dictionaryData, indicesTreinamento, indicesTeste, indicesValidacao, name):
    dictKeys = dictionaryData.keys()
    keysArray = np.array(list(dictKeys))
    # print('keysArray', keysArray)
    # print('len', len(keysArray))
    # print('type', type(keysArray))

    # To force specific patients
    # trainPatients = indicesTreinamento
    # testPatients = indicesTeste
    # validationPatients = indicesValidacao

    trainPatients = keysArray[indicesTreinamento]
    testPatients = keysArray[indicesTeste]
    validationPatients = keysArray[indicesValidacao] 
    
    print('trainPatients', trainPatients)
    print('validationPatients', validationPatients) 
    print('testPatients', testPatients)
    # print('originalmente testPatients', testPatients)
    # print('originalmente validationPatients', validationPatients)
    
    trainDataset = []
    # testDataset = []
    # i = 0
    # testPatientsAsTrain = []
    for patient in trainPatients:
        # print('patient', patient)
        images = dictionaryData[patient]
        # print('len(images', len(images))
        trainDataset.extend(images)
        # if i < len(testPatients):
        #     testPatientsAsTrain.append(patient)
        #     # print('patient', patient, 'esta no teste')
        #     i = i+1
        #     testDataset.extend(images)
    # print('testPatientsAsTrain', testPatientsAsTrain)
    print('imagens do trainDataset', len(trainDataset))
    
    # mixedTestPatients = []
    #mixedValidationPatients = []
    testDataset = []
    validationDataset = []

    # i = 0
    # for patient in testPatients:
    #     if i%2 == 0:
    #         #print('paciente ', patient, ' é teste')
    #         images = dictionaryData[patient]
    #         mixedTestPatients.append(patient)
    #         testDataset.extend(images)
    #     i = i+1

    # i = 0
    # for patient in testPatients:
    #     if i%2 == 0:
    #         print('paciente ', patient, ' é teste')
    #         images = dictionaryData[patient]
    #         mixedTestPatients.append(patient)
    #         testDataset.extend(images)
    #     else:
    #         print('paciente ', patient, ' é validacao')
    #         images = dictionaryData[patient]
    #         mixedValidationPatients.append(patient)
    #         validationDataset.extend(images)
    #     i = i+1

    for patient in testPatients:
        images = dictionaryData[patient]
        testDataset.extend(images)

    for patient in validationPatients:
        images = dictionaryData[patient]
        validationDataset.extend(images)

        # if i%2 == 0:
        #     #print('paciente ',patient, ' é teste')
        #     images = dictionaryData[patient]
        #     mixedTestPatients.append(patient)
        #     testDataset.extend(images)
        # i = i+1
        
        # if i%2 == 0:
        #     print('paciente ',patient, ' é teste')
        #     images = dictionaryData[patient]
        #     mixedTestPatients.append(patient)
        #     testDataset.extend(images)
        # else:
        #     print('paciente ',patient, ' é validacao')
        #     images = dictionaryData[patient]
        #     mixedValidationPatients.append(patient)
        #     validationDataset.extend(images)
        # i = i+1

    # print('imagens do mixedTestPatients', mixedTestPatients)
    print('imagens do testDataset', len(testDataset))
    print('imagens do validationDataset', len(validationDataset))

    train, test, validation = np.array(trainDataset), np.array(testDataset), np.array(validationDataset)
    # resizedTrained, resizedTest, resizedValidation = train, test, validation 
    resizedTrained, resizedTest, resizedValidation = resizeImages(train, test, validation)

    # plotAllSubsetImages(resizedTrained, name+'train')
    # plotAllSubsetImages(resizedTest, name+'test')
    # plotAllSubsetImages(resizedValidation, name+'validation')

    print('train', resizedTrained.shape)
    print('test', resizedTest.shape)
    print('validation', resizedValidation.shape)
    return resizedTrained, resizedTest, resizedValidation

def resizeImages(train, test, validation):
    resizedTrained = np.zeros((train.shape[0], 3, 224, 224))
    resizedTest = np.zeros((test.shape[0], 3, 224, 224))
    resizedValidation = np.zeros((validation.shape[0], 3, 224, 224))

    for i in range(train.shape[0]):
        resizedTrained[i, :] = transform.resize(train[i], (3, 224, 224))
    
    for i in range(test.shape[0]):
        resizedTest[i, :] = transform.resize(test[i], (3, 224, 224))
    
    for i in range(validation.shape[0]):
        resizedValidation[i, :] = transform.resize(validation[i], (3, 224, 224))
    
    return resizedTrained, resizedTest, resizedValidation

def splitPatientsFromDictionary(shuffleSeed, dictionaryData, numberTrain=None, numberValidacao=None):
    print('Total dados', len(dictionaryData))
    totalPatientsDataset = len(dictionaryData.keys())
    if numberTrain == None:
        trainPatientsDataset = math.floor(totalPatientsDataset*0.70)
        validationPatientsDataset = math.floor((totalPatientsDataset - trainPatientsDataset)/2)
    else: 
        trainPatientsDataset = numberTrain
        validationPatientsDataset = numberValidacao
    #validationPatientsTotal = totalPatientsDataset - trainPatientsDataset - testPatientsDataset

    indicesDataset = list(range(totalPatientsDataset))
    np.random.seed(shuffleSeed)
    np.random.shuffle(indicesDataset)

    indicesTreinamento = indicesDataset[:trainPatientsDataset] 
    indicesValidacao = indicesDataset[trainPatientsDataset:trainPatientsDataset+validationPatientsDataset]
    indicesTeste= indicesDataset[trainPatientsDataset+validationPatientsDataset:]
    
    print('Quantidade de dados para treinamento',len(indicesTreinamento))
    print('Quantidade de dados para teste',len(indicesTeste))
    print('Quantidade de dados para validacao',len(indicesValidacao))

    return indicesTreinamento, indicesTeste, indicesValidacao

def minMaxNormalization(dataTrain, dataTest, dataValidation, deltaT, min10mean):
    print('Aplicando min max')
    #dataTrain = (dataTrain)/deltaT
    dataTrain = (dataTrain-min10mean)/deltaT
    print('dataTrain', type(dataTrain))
    #plotAllSubsetImages(dataTrain, 'train')

    flattenDataset = dataTrain.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    topValues = flattenDataset[indices]
    indices = np.argpartition(flattenDataset, 100)[:100]
    minValuesValues = flattenDataset[indices]
    minMean = np.mean(minValuesValues)
    # print('Train minMean', minMean)
    topMean = np.mean(topValues)
    # print('topMean', topMean)
    media = np.mean(flattenDataset)  
    # print('Media train', media)
    desvioPadrao = np.std(flattenDataset)  
    # print('Desvio padrao train', desvioPadrao)
    variancia = np.var(flattenDataset)  
    # print('Variancia train', variancia)
    # print('min train', dataTrain.min())

    #dataTest = (dataTest)/deltaT
    dataTest = (dataTest-min10mean)/deltaT
    #plotAllSubsetImages(dataTest, 'test')

    flattenDataset = dataTest.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    topValues = flattenDataset[indices]
    indices = np.argpartition(flattenDataset, 100)[:100]
    minValuesValues = flattenDataset[indices]
    minMean = np.mean(minValuesValues)
    # print('Test minMean', minMean)
    topMean = np.mean(topValues)
    # print('topMean', topMean)
    media = np.mean(flattenDataset)  
    # print('Media Test', media)
    desvioPadrao = np.std(flattenDataset)  
    # print('Desvio padrao Test', desvioPadrao)
    variancia = np.var(flattenDataset)  
    # print('Variancia Test', variancia)
    # print('min traiTestn', dataTest.min())

    #dataValidation = (dataValidation)/deltaT
    dataValidation = (dataValidation-min10mean)/deltaT
    #plotAllSubsetImages(dataValidation, 'validation')

    flattenDataset = dataValidation.flatten()
    indices = np.argpartition(flattenDataset, -100)[-100:] 
    topValues = flattenDataset[indices]
    indices = np.argpartition(flattenDataset, 100)[:100]
    minValuesValues = flattenDataset[indices]
    minMean = np.mean(minValuesValues)
    # print('Validation minMean', minMean)
    topMean = np.mean(topValues)
    # print('topMean', topMean)
    media = np.mean(flattenDataset)  
    # print('Media Validation', media)
    desvioPadrao = np.std(flattenDataset)  
    # print('Desvio padrao Validation', desvioPadrao)
    variancia = np.var(flattenDataset)  
    # print('Variancia Validation', variancia)
    # print('min validation', dataValidation.min())
    return dataTrain, dataTest, dataValidation

def prepareNumpyDatasetBalancedData(dataTrain, dataTargetTrain, dataTest, dataTargetTest, dataValidation, dataTargetValidation, batch_size, dataAugmentation, mean, std ):
    print('prepareNumpyDatasetBalancedData')

    #The data augmentation step conveys four types of image data generation: 
    # (i) horizontal and vertical flip; 
    # (ii) rotation between 0-45 degrees; 
    # (iii) 20% zoom and;
    # (iv) normalized noises, e.g. Gaussian. 
    f = np.array([0.22, 0.22, 0.22])
    #print('f', f)
    fillData = tuple(np.round(f * 255).astype(np.int64))
    #print('fillData', fillData)
    trainTransform = transforms.Compose([
        #transforms.RandomRotation(degrees=30, fill=(0,)),
        #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        #transforms.Resize((224, 224)),
        transforms.ToPILImage(),
        transforms.RandomRotation(degrees=30, fill=(fillData)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([mean,mean, mean], [std, std, std])  
    ])
    
    testValidationTransform = transforms.Compose([
    #     transforms.Resize((224, 224)),
        # transforms.ToTensor(),
    #     transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    #     #transforms.Lambda(lambda x: torch.cat([x/topMean, x/topMean, x/topMean], 0)),
        transforms.Normalize([mean, mean, mean], [std, std, std])  
    ])
    
    if dataAugmentation:
        print('Com aumento de dados', trainTransform)
        trainDataset = CustomDatasetFromNumpyArray(dataTrain, dataTargetTrain, trainTransform)
    else:
        print('Sem aumento de dados')
        print('com testValidationTransform', testValidationTransform)
        trainDataset = CustomDatasetFromNumpyArray(dataTrain, dataTargetTrain, testValidationTransform)
    
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    testDataset = CustomDatasetFromNumpyArray(dataTest, dataTargetTest, testValidationTransform)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

    validationDataset = CustomDatasetFromNumpyArray(dataValidation, dataTargetValidation, testValidationTransform)
    validationLoader = DataLoader(validationDataset, batch_size=batch_size, shuffle=True)

    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        # plotTransformedImages(images, i, 'transformed_train', mean, std)
        # plotAllSubsetImages(images, 'transformed_train'+str(i), mean, std)
        i = i+1

    i=0
    resultLabelsTesting = torch.zeros(2, dtype=torch.long)
    for images, labels in iter(testLoader):
        l = labels.numpy()
        resultLabelsTesting[0] = resultLabelsTesting[0] + np.count_nonzero(l == 0)
        resultLabelsTesting[1] = resultLabelsTesting[1] + np.count_nonzero(l == 1)
        # plotTransformedImages(images, i, 'transformed_test', mean, std)
        # plotAllSubsetImages(images, 'transformed_test'+str(i), mean, std)
        i = i+1

    resultLabelsValidation = torch.zeros(2, dtype=torch.long)
    i = 0
    for images, labels in iter(validationLoader):
        l = labels.numpy()
        resultLabelsValidation[0] = resultLabelsValidation[0] + np.count_nonzero(l == 0)
        resultLabelsValidation[1] = resultLabelsValidation[1] + np.count_nonzero(l == 1)
        # plotTransformedImages(images, i, 'transformed_validation',  mean, std)
        # plotAllSubsetImages(images, 'transformed_validation'+str(i), mean, std)
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
