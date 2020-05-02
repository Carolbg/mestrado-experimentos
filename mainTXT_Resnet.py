from handleTXTDataset import prepareDataFromTXT, splitDataset, prepareNumpyDataset, getCommonArgs
from prepareModel import prepareModel, prepareTrainingLoss, prepareTrainingOptimizer, prepareResnetModelWithTXT
from training import train
from testing import evaluate
from plots import plotLosses, plotAcc, plotTestingAcc, plotAll
#from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray

def mainResnet():
    print('\n\nTESTES COM RESNET\n\n')
    #DATASET STEPS:
    print('Load dataset')
    data, dataTarget = prepareDataFromTXT()
    shuffleSeed, batch_size, max_epochs_stop, n_epochs = getCommonArgs()
    train_idx, test_idx, valid_idx = splitDataset(data, shuffleSeed)
    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareNumpyDataset(data, dataTarget, train_idx, test_idx, valid_idx, batch_size)

    #PREPARE MODEL STEPS:
    print('Prepare model')
    model = prepareResnetModelWithTXT(data, n_classes)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model)

    print('Train model')
    save_file_name = 'resnet-txt-teste.pt'
    checkpoint_path = 'resnet-txt-teste.pth'
    model, history, train_loss, valid_loss, train_acc, valid_acc, valid_best_acc = train(model, criterion,
        optimizer, trainLoader, validationLoader, save_file_name, max_epochs_stop=max_epochs_stop, 
        n_epochs=n_epochs, print_every=1)

      #PLOT TRAINING RESULTS
    print('Plot training results')
    plotLosses(history, 'resnet')
    plotAcc(history, 'resnet')

    #TEST MODEL
    print('Test model')
    results, test_error_count = evaluate(model, testLoader, criterion, n_classes)
    #print('Results Head', results)
    #print('test_error_count = ', test_error_count)
    #print('Results Head', results.head())
    
    results2 = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])
    plotTestingAcc(results2, 'resnet')
    #plotAll(results2, history)

    return model, history, train_loss, valid_loss, train_acc, valid_acc, valid_best_acc, results, trainLoader, testLoader, validationLoader, n_classes, cat_df, results2
    
