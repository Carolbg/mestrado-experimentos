import torch.nn as nn
from torch import optim

def getCommonArgs():
    shuffleSeed = 3
    print('shuffleSeed', shuffleSeed)
    batch_size = 10
    max_epochs_stop = 30
    n_epochs = 30
    print('n_epochs', n_epochs)
    return shuffleSeed, batch_size, max_epochs_stop, n_epochs

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
            nn.Flatten(),
            nn.Linear(n_inputs, 1024), nn.ReLU(),
            nn.Linear(1024, n_classes)
        )
    elif experimentType == 9:
        lastLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(n_inputs, 1024), nn.ReLU(),
            nn.Linear(1024, n_classes)
        )
    elif experimentType == 10:
        lastLayer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, 1024), nn.ReLU(),
            nn.Dropout(0.5), 
            nn.Linear(1024, n_classes)
        )
    elif experimentType == 11:
        lastLayer = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Linear(n_inputs, 1024), nn.ReLU(),
            nn.Dropout(0.5), 
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