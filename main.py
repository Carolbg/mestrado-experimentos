from handleDataset import readDataset, splitDataset, prepareDataset
from prepareModel import prepareModel, prepareTrainingLoss, prepareTrainingOptimizer
from training import train
from testing import evaluate
from plots import plotLosses, plotAcc, plotTestingAcc

def main():
    
    #DATASET STEPS:
    print('Load and prepare dataset')
    dataset = readDataset('../Imagens_Teste_Estaticas')

    shuffleSeed = 3
    train_idx, test_idx, valid_idx = splitDataset(dataset, shuffleSeed)

    batch_size = 41
    trainLoader, testLoader, validationLoader, n_classes, cat_df = prepareDataset(dataset, train_idx, 
        test_idx, valid_idx, batch_size)

    #PREPARE MODEL STEPS:
    print('Prepare model')
    model = prepareModel(dataset, n_classes)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model)

    #TRAIN MODEL:
    print('Train model')
    save_file_name = 'vgg16-teste1.pt'
    checkpoint_path = 'vgg16-teste1.pth'
    model, history, train_loss, valid_loss, train_acc, valid_acc, valid_best_acc = train(model, criterion,
        optimizer, trainLoader, validationLoader, save_file_name, max_epochs_stop=2, 
        n_epochs=2, print_every=1)


    #PLOT TRAINING RESULTS
    print('Plot training results')
    plotLosses(history)
    plotAcc(history)

    #TEST MODEL
    print('Test model')
    results, test_error_count = evaluate(model, testLoader, criterion, n_classes)
    #print('Results Head', results)
    print('test_error_count = ', test_error_count)
    print('Results Head', results.head())
    


    results2 = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])
    plotTestingAcc(results2)
    
    print('results2 = ', results2)
    return model, history, train_loss, valid_loss, train_acc, valid_acc, valid_best_acc, results, trainLoader, testLoader, validationLoader, n_classes, cat_df, results2
    
    
