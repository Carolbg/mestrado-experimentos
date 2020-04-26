from handleDataset import readDataset, splitDataset, prepareDataset
from prepareModel import prepareModel, prepareTrainingLoss, prepareTrainingOptimizer
from training import train
from testing import evaluate
from plots import plotLosses, plotAcc

def main():

    #DATASET STEPS:
    dataset = readDataset('../Imagens_Teste_Estaticas')

    shuffleSeed = 10
    train_idx, test_idx, valid_idx = splitDataset(dataset, shuffleSeed)

    batch_size = 41
    trainLoader, testLoader, validationLoader, n_classes = prepareDataset(dataset, train_idx, 
        test_idx, valid_idx, batch_size)

    #PREPARE MODEL STEPS:
    model = prepareModel(dataset, n_classes)
    criterion = prepareTrainingLoss()
    optimizer = prepareTrainingOptimizer(model)

    #TRAIN MODEL:
    save_file_name = 'vgg16-teste1.pt'
    checkpoint_path = 'vgg16-teste1.pth'
    #model, history, train_loss, valid_loss, train_acc, valid_acc = train(model, criterion,
    #    optimizer, trainLoader, validationLoader, save_file_name, max_epochs_stop=1, 
    #    n_epochs=1, print_every=1)

    #plotLosses(history)
    #plotAcc(history)
    #evaluate(model, testLoader, criterion)


main()
    
    
