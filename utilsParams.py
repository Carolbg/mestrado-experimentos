import torch.nn as nn
from torch import optim

def getCommonArgs():
    shuffleSeed = 3
    batch_size = 10
    max_epochs_stop = 30
    n_epochs = 30
    print('n_epochs', n_epochs)
    return shuffleSeed, batch_size, max_epochs_stop, n_epochs

def getFullyConnectedStructure(n_inputs, n_classes):
    #nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    #lastLayer = nn.Sequential(nn.Linear(n_inputs, 256), nn.ReLU(), nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))
    #lastLayer = nn.Sequential(nn.Linear(n_inputs, 1024), nn.ReLU(), nn.Linear(1024, n_classes), nn.LogSoftmax(dim=1))
    lastLayer = nn.Sequential(nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
    #lastLayer = nn.Sequential(nn.Linear(n_inputs, 512), nn.ReLU(), nn.Linear(512, n_classes), nn.LogSoftmax(dim=1))
    print('lastLayer', lastLayer)
    return lastLayer
    #return nn.Sequential(nn.Linear(n_inputs, n_classes), nn.ReLU())

def prepareTrainingLoss():
    criterion = nn.NLLLoss()
    return criterion

def prepareTrainingOptimizer(model):

    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters())

    #for p in optimizer.param_groups[0]['params']:
    #    if p.requires_grad:
    #        print(p.shape)
    return optimizer