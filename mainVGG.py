from handleTXTDataset import createDataLoaders
from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareVGG16ModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray
from prepareDataDictionary import mainPrepareDictionaryData
from utils import saveCsvConfusionMatrix

def mainVGG(resultsPlotName, experimentType, dataAugmentation):
    print('\n\nTESTES COM VGG\n\n')

    resultsPlotName = 'vgg_'+resultsPlotName
    #DATASET STEPS:
    print('Load dataset')
    #trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = createDataLoaders()
    trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryData(dataAugmentation)
    
    #PREPARE MODEL STEPS:
    print('\nPrepare model')
    model = prepareVGG16ModelWithTXT(experimentType)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model, typeLR)

    print('Train model')
    save_file_name = 'vgg16-txt-teste.pt'

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
    #print('Results Head', results)
    #print('test_error_count = ', test_error_count)
    #print('Results Head', results.head())
    
    # results2 = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])
    # plotTestingAcc(results2, 'vgg')

    saveCsvConfusionMatrix(cmTest, resultsPlotName)

    return model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df 
    
