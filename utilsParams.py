import torch.nn as nn
import torch
from torch import optim, cuda
from torch.optim import lr_scheduler

def getCommonArgs():
    shuffleSeed = 1
    print('shuffleSeed', shuffleSeed)
    batch_size = 10
    max_epochs_stop = 30
    n_epochs = 30
    print('n_epochs', n_epochs, 'max_epochs_stop', max_epochs_stop)
    device = getDevice()

    return shuffleSeed, batch_size, max_epochs_stop, n_epochs, device

def getDevice():
    # Whether to train on a gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print('device = ', device)

    return device

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

    print('lastLayer', lastLayer)
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
    else:
        optimizer = optim.Adam(model.parameters())

    print('optimizer', optimizer)
    return optimizer

def decayLR():
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return exp_lr_scheduler