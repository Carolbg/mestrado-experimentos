from utils_cnnInit import *

def readData(isNumpy, nEpochs):
    global trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion,  max_epochs_stop, n_epochs
    trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion,  max_epochs_stop, n_epochs = prepareCNN(isNumpy, nEpochs)

def getData():
    print('aqui n_epochs', n_epochs)
    return trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion,  max_epochs_stop, n_epochs