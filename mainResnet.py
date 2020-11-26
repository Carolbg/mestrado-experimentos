from handleTXTDataset import createDataLoaders#prepareDataFromTXT, splitDataset, prepareNumpyDataset, getCommonArgs
from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareResnetModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray
from prepareDataDictionary import mainPrepareDictionaryData
from utils import saveCsvConfusionMatrix
from readMatlabNumpyData import mainPrepareDictionaryDataFromNumpy

def mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR, isNumpy=True):
    print('\n\nTESTES COM RESNET\n\n')

    resultsPlotName = resultsPlotName + '_resnet'
    #DATASET STEPS:
    print('isNumpy', isNumpy)
    print('Load dataset')
    #trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = createDataLoaders()
    if isNumpy:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device = mainPrepareDictionaryDataFromNumpy(dataAugmentation)
    else:
        trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device = mainPrepareDictionaryData(dataAugmentation)
    
    #PREPARE MODEL STEPS:
    print('\nPrepare model')
    model = prepareResnetModelWithTXT(experimentType, device)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model, typeLR)

    print('Train model')
    save_file_name = 'resnet-txt-teste.pt'
    checkpoint_path = 'resnet-txt-teste.pth'
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, max_epochs_stop, n_epochs, device)

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
    
    return model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df 
    
