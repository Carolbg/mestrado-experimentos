from handleTXTDataset import createDataLoaders #prepareDataFromTXT, splitDataset, prepareNumpyDataset, getCommonArgs
from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareDensenetModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray
from prepareDataDictionary import mainPrepareDictionaryData
from utils import saveCsvConfusionMatrix
from readMatlabNumpyData import mainPrepareDictionaryDataFromNumpy

def mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy=True):
    print('\n\nTESTES COM DENSENET\n\n')
    
    resultsPlotName = resultsPlotName + 'densenet_'

    #DATASET STEPS:
    print('Load dataset')
    #trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = createDataLoaders()
    if isNumpy:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryDataFromNumpy(dataAugmentation)
    else:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryData(dataAugmentation)
    
    #PREPARE MODEL STEPS:
    print('\nPrepare model')
    model = prepareDensenetModelWithTXT(experimentType)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model, typeLR)

    print('Train model')
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, max_epochs_stop=max_epochs_stop, 
        n_epochs=n_epochs)

    print('\nConfusion matrix Train\n', cmTrain)
    print('\nConfusion matrix Validation\n', cmValidation)
    
    #PLOT TRAINING RESULTS
    print('\nPlot training results')
    plotData(history, resultsPlotName)

    #TEST MODEL
    print('Test model')
    historyTest, cmTest = evaluate(model, testLoader, criterion, n_classes, resultsPlotName)
    print('\nConfusion matrix Test\n', cmTest)
    saveCsvConfusionMatrix(cmTest, resultsPlotName)

    return model, history, historyTest, cmTrain, cmValidation,cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df
    
