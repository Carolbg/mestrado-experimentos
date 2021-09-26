import torch.nn as nn
import torch
from torch import optim, cuda
from torch.optim import lr_scheduler

def getCommonArgs(nEpochs=30):
    shuffleSeed = 1
    print('shuffleSeed', shuffleSeed)
    batch_size = 10
    max_epochs_stop = 10
    n_epochs = nEpochs
    print('n_epochs', n_epochs, 'max_epochs_stop', max_epochs_stop)
    device = getDevice()

    return shuffleSeed, batch_size, max_epochs_stop, n_epochs, device

def getDevice():
    # Whether to train on a gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print('device = ', device)

    return device

def init_weights(m):
    print('type(m)', type(m))
    if type(m) == nn.Linear:
        # nn.init.xavier_normal_(w)
        torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        torch.nn.init.zeros_(m.bias)

def getFullyConnectedStructure(n_inputs, n_classes, experimentType):
    #nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes))
    if experimentType == 1:
        lastLayer = nn.Linear(n_inputs, n_classes)
    elif experimentType == 2:
        lastLayer = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Linear(256, n_classes))
    elif experimentType == 3:
        lastLayer = nn.Sequential(nn.Linear(n_inputs, 512), nn.ReLU(), nn.Linear(512, n_classes))
    elif experimentType == 4:
        lastLayer = nn.Sequential(nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Linear(1024, n_classes))
    elif experimentType == 5:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(1024, n_classes)
        )
    elif experimentType == 6:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 4096), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, n_classes)
        )
    elif experimentType == 7:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 4096), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, n_classes)
        )
    elif experimentType == 8:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 4096), nn.ReLU(), nn.Dropout(0.5), 
            nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, n_classes)
        )

    # Best config of PSO Densenet Frontal
    elif experimentType == 10:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(), nn.Dropout(0.07977606125630654), 
            nn.Linear(64, 32), nn.ReLU(), 
            nn.Linear(32, 16), nn.ReLU(), 
            nn.Linear(16, n_classes)
        )
    # Best config of AG Densenet Frontal
    elif experimentType == 11:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Dropout(0.06673564902817675), 
            nn.Linear(1024, 8), nn.ReLU(),
            nn.Linear(8, n_classes)
        )
    
    # Best config of PSO Densenet Cabioglu
    elif experimentType == 12:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 64), nn.ReLU(), nn.Dropout(0.2950256631398443), 
            nn.Linear(64, 32), nn.ReLU(), 
            nn.Linear(32, n_classes)
        )
    # Best config of AG Densenet Cabioglu
    elif experimentType == 13:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.34015512129585607),
            nn.Linear(256, n_classes)
        )
    # Best AG Resnet 0.90
    elif experimentType == 14:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
            nn.Dropout(0.4267150503559332),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    # Best AG VGG 0.92
    elif experimentType == 15:
        lastLayer = nn.Sequential(
            nn.Linear(n_inputs, 32),
            nn.ReLU(),
            nn.Dropout(0.3065990603705573),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.3153523843591079),
            nn.Linear(8, 2)
        )
    print('lastLayer', lastLayer)
    lastLayer.apply(init_weights)
    return lastLayer

def prepareTrainingLoss():
    criterion = nn.CrossEntropyLoss()
    return criterion

def prepareTrainingOptimizer(model, typeLR):

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if typeLR == 1:
        lr = 0.0001
        print('Learning Rate', lr)
        optimizer = optim.Adam(model.parameters(), lr)
    elif typeLR == 2:
        optimizer = optim.Adam(model.parameters())
    elif typeLR == 4: # Best AG Resnet 0.90
        lr = 0.00001 
        print('Learning Rate', lr)
        optimizer = optim.Adam(model.parameters(), lr)
    elif typeLR == 5:
        lr = 0.00000001 # Best AG VGG 0.92
        print('Learning Rate', lr)
        optimizer = optim.Adam(model.parameters(), lr)


    print('optimizer', optimizer)
    return optimizer

def decayLR():
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return exp_lr_scheduler