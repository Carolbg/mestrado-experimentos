from handleTXTDataset import createDataLoaders#prepareDataFromTXT, splitDataset, prepareNumpyDataset, getCommonArgs
from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareResnetModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray
from prepareDataDictionary import mainPrepareDictionaryData

def mainResnet(resultsPlotName):
    print('\n\nTESTES COM RESNET\n\n')
    #DATASET STEPS:
    print('Load dataset')
    #trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = createDataLoaders()
    trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryData()
    
    #PREPARE MODEL STEPS:
    print('\nPrepare model')
    model = prepareResnetModelWithTXT(n_classes)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model)

    print('Train model')
    save_file_name = 'resnet-txt-teste.pt'
    checkpoint_path = 'resnet-txt-teste.pth'
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, save_file_name, max_epochs_stop=max_epochs_stop, 
        n_epochs=n_epochs, print_every=1)

    print('\nConfusion matrix Train\n', cmTrain)
    print('\nConfusion matrix Validation\n', cmValidation)

    #PLOT TRAINING RESULTS
    print('\nPlot training results')
    plotData(history, resultsPlotName)

    #TEST MODEL
    print('Test model')
    historyTest, cmTest = evaluate(model, testLoader, criterion, n_classes, resultsPlotName)
    print('\nConfusion matrix Test\n', cmTest)
    tn, fp, fn, tp = cmTest.ravel()
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)
    
    return model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df 
    
