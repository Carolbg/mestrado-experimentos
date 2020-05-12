from handleTXTDataset import createDataLoaders#prepareDataFromTXT, splitDataset, prepareNumpyDataset, getCommonArgs
from prepareModel import prepareTrainingLoss, prepareTrainingOptimizer, prepareResnetModelWithTXT
from training import train
from testing import evaluate
from plots import plotData, plotTestingAcc
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray

def mainResnet():
    print('\n\nTESTES COM RESNET\n\n')
    #DATASET STEPS:
    print('Load dataset')
    # data, dataTarget = prepareDataFromTXT()
    # shuffleSeed, batch_size, max_epochs_stop, n_epochs = getCommonArgs()
    # train_idx, test_idx, valid_idx = splitDataset(data, shuffleSeed)
    # trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDataset(data, dataTarget, train_idx, test_idx, valid_idx, batch_size)
    trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = createDataLoaders()

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
    plotData(history, 'resnet')

    #TEST MODEL
    print('Test model')
    results, test_error_count, historyTest, cmTest = evaluate(model, testLoader, criterion, n_classes)
    print('\nConfusion matrix Test\n', cmTest)
    tn, fp, fn, tp = cmTest.ravel()
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)
    
    results2 = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])
    plotTestingAcc(results2, 'resnet')
    #plotAll(results2, history)

    return model, history, historyTest, results, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df 
    
