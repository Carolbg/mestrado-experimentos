import torch.nn as nn
from torchvision import models
from torch import optim
import copy

def convertParticleToCNN(particle, device, cnnType):
    #cnnType = 1 => resnet, cnnType = 2 => VGG, cnnType = 3 => Densenet

    layer = {
        'layerType': 'FC',
        'layerNumber': 1
    }
    particleCopy = copy.deepcopy(particle)
    if len(particle) == 2 and particle[1]['layerType'] == 'Dropout':
        particleCopy[1] = layer
        # print('-----> NO IF convertParticleToCNN')
        # print('particleCopy', particleCopy)
    else:
        particleCopy.append(layer)
    

    if cnnType == 1:
        model = generateResnetModelFromAG(device, particleCopy)
    elif cnnType == 2:
        model = generateVGGModelFromAG(device, particleCopy)
    else:
        model = generateDensenetModelFromAG(device, particleCopy)
    
    # LR
    lrIndex = particleCopy[0]['layerNumber']
    # print('particle[0]', particle[0], 'lrIndex', lrIndex)
    optimizer = prepareOptimizer(model, lrIndex)

    return model, optimizer

def generateResnetModelFromAG(device, particle):
    model = models.resnet50(pretrained=True)
    # print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    # print('n_inputs', n_inputs)

    # Add on classifier
    model.fc = getFullyConnectedStructureFromParticle(n_inputs, particle)
    print('modelFC', model.fc)

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}

    return model.to(device)

def generateVGGModelFromAG(device, particle):
    model = models.vgg16(pretrained=True)
    # print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[0].in_features

    # Add on classifier
    model.classifier = getFullyConnectedStructureFromParticle(n_inputs, particle)
    print('custom fc', model.classifier)

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}

    return model.to(device)

def generateDensenetModelFromAG(device, particle):
    model = models.densenet201(pretrained=True)
    # print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier.in_features

    # Add on classifier
    model.classifier = getFullyConnectedStructureFromParticle(n_inputs, particle)
    print('custom fc', model.classifier)

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}

    return model.to(device)

def getFullyConnectedStructureFromParticle(nInputs, particle):
    particleSize = len(particle)
    # print('particleSize', particleSize)
    layersAsArray = []
    
    i = 1
    while i < particleSize:
        # for i in range(1, particleSize):
        # print('i', i)
        geneLayer = particle[i]['layerNumber']
        # print('particle[i]',  particle[i], 'geneLayer', geneLayer)

        geneDropout = None
        #tem a camada dessa layer?
        if i+1<particleSize and particle[i+1]['layerType'] == 'Dropout':
            i = i+1
            geneDropout = particle[i]['layerNumber']
            # print('particle[i]',  particle[i], 'geneDropout', geneDropout)
            i = i+1
        else:
            i = i+1
        layersAsArray.extend(defineSingleLayer(nInputs, geneLayer, geneDropout))
        nInputs = pow(2, geneLayer)
        # print('nInputs', nInputs)

    # addFinalLayer
    # layersAsArray.extend(defineFinalLayer(nInputs))

    # print('layersAsArray', layersAsArray)
    layers = nn.Sequential(*layersAsArray)
    return layers

def defineSingleLayer(n_inputs, geneLayer, geneDropout):
    # print('n_inputs, geneLayer, geneDropout', n_inputs, geneLayer, geneDropout)
    numeroNeurons = pow(2, geneLayer)
    # print('numeroNeurons', numeroNeurons)

    if geneDropout == None:
        layer = nn.Sequential(
            nn.Linear(n_inputs, numeroNeurons), 
            nn.ReLU()
        )
    else:
        layer = nn.Sequential(
            nn.Linear(n_inputs, numeroNeurons), 
            nn.ReLU(), 
            nn.Dropout(geneDropout)
        )
    # print('layer', layer)
    return layer

def prepareOptimizer(model, expoente):
    lr = pow(10, -expoente)
    # print('Learning Rate', lr)
    optimizer = optim.Adam(model.parameters(), lr)

    # print('optimizer', optimizer)
    return optimizer
