from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareDensenetModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray
from prepareDataDictionary import mainPrepareDictionaryData
from utils import saveCsvConfusionMatrix
from readMatlabNumpyData import mainPrepareDictionaryDataFromNumpy
import gc
import torch

def mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy=True, nEpochs=30, maxEpochs=None):
    print('\n\nTESTES COM DENSENET\n\n')
    resultsPlotName = resultsPlotName + '_densenet'
    print('isNumpy', isNumpy)
    
    #DATASET STEPS:
    print('Load dataset')
    if isNumpy:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, defaultMaxEpochs, n_epochs, device = mainPrepareDictionaryDataFromNumpy(dataAugmentation, nEpochs)
    else:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, defaultMaxEpochs, n_epochs, device = mainPrepareDictionaryData(dataAugmentation, nEpochs)

    max_epochs_stop = maxEpochs if maxEpochs != None else defaultMaxEpochs
    n_epochs = nEpochs

    print('\n\nReal n_epochs', n_epochs)
    print('\n\nReal max_epochs_stop', max_epochs_stop)

    gc.collect()
    torch.cuda.empty_cache()

    #PREPARE MODEL STEPS:
    print('\nPrepare model')
    model = prepareDensenetModelWithTXT(experimentType, device)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model, typeLR)
    gc.collect()
    torch.cuda.empty_cache()
    
    print('Train model')
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, max_epochs_stop, n_epochs, device)
    
    gc.collect()
    torch.cuda.empty_cache()

    print('\nConfusion matrix Train\n', cmTrain)
    print('\nConfusion matrix Validation\n', cmValidation)
    
    #PLOT TRAINING RESULTS
    print('\nPlot training results')
    plotData(history, resultsPlotName)

    #TEST MODEL
    print('Test model')
    historyTest, cmTest = evaluate(model, testLoader, criterion, n_classes, resultsPlotName, device)
    print('\nConfusion matrix Test\n', cmTest)
    saveCsvConfusionMatrix(cmTest, resultsPlotName)
    gc.collect()
    torch.cuda.empty_cache()
    
    return model, history, historyTest, cmTrain, cmValidation,cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df
    
