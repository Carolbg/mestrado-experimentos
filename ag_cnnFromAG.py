import torch.nn as nn
from torchvision import models
from torch import optim

def convertAgToCNN(individuo, device):
    model = generateResnetModelFromAG(device, individuo)
    
    #LR
    lrIndex = individuo[0][0]
    optimizer = prepareOptimizer(model, lrIndex)

    #epocas
    epocas = (individuo[1][0])*10
    
    return model, optimizer, epocas

def generateResnetModelFromAG(device, individuo):
    model = models.resnet50(pretrained=True)
    # print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features

    # Add on classifier
    model.fc = getFullyConnectedStructureFromAG(n_inputs, individuo)
    print('custom fc', model.fc)

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}

    return model.to(device)

def getFullyConnectedStructureFromAG(nInputs, individuo):
    individuoSize = len(individuo)
    # print('individuoSize', individuoSize)
    layersAsArray = []
    for i in range(2, individuoSize, 2):
        # print('i', i)
        geneLayer = individuo[i]
        # print('geneLayer', geneLayer)
        #tem a camada dessa layer?
        if geneLayer[0] == 1:
            geneDropout = individuo[i+1]
            layersAsArray.extend(defineSingleLayer(nInputs, geneLayer, geneDropout))
            nInputs = pow(2, geneLayer[1])

    # addFinalLayer
    layersAsArray.extend(defineFinalLayer(nInputs))

    # print('layersAsArray', layersAsArray)
    layers = nn.Sequential(*layersAsArray)
    return layers

def defineFinalLayer(n_inputs):
    nClasses = 2
    layer = nn.Sequential(nn.Linear(n_inputs, nClasses))
    return layer

def defineSingleLayer(n_inputs, geneLayer, geneDropout):
    #sem dropout
    # print('geneDropout', geneDropout)
    numeroNeurons = pow(2, geneLayer[1])

    if geneDropout[0] == 0:
        layer = nn.Sequential(
            nn.Linear(n_inputs, numeroNeurons), 
            nn.ReLU()
        )
    else:
        layer = nn.Sequential(
            nn.Linear(n_inputs, numeroNeurons), 
            nn.ReLU(), 
            nn.Dropout(geneDropout[1])
        )
    # print('layer', layer)
    return layer

def prepareOptimizer(model, expoente):
    lr = pow(10, -expoente)
    # print('Learning Rate', lr)
    optimizer = optim.Adam(model.parameters(), lr)

    # print('optimizer', optimizer)
    return optimizer

