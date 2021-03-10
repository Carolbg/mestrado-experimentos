from readMatlabNumpyData import mainPrepareDictionaryDataFromNumpy
from prepareDataDictionary import mainPrepareDictionaryData
from utilsParams import prepareTrainingLoss

def prepareCNN(isNumpy, nEpochs):
    #funcoes com coisas que nao vao mudar em decorrencia do AG
    
    #faco a leitura das imagens e divisao em treinamento/teste/validacao
    #DATASET STEPS:
    print('isNumpy', isNumpy, 'nEpochs', nEpochs)
    print('Load dataset')
    
    dataAugmentation=False
    if isNumpy:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device = mainPrepareDictionaryDataFromNumpy(dataAugmentation, nEpochs)
    else:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device = mainPrepareDictionaryData(dataAugmentation, nEpochs)

    #defino o criterio, CrossEntropyLoss para todo mundo tambem
    criterion = prepareTrainingLoss()

    return trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion,  max_epochs_stop, n_epochs
    