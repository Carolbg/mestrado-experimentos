from torchvision import transforms, datasets, models
import torch.nn as nn
from utilsParams import *

def prepareVGG16ModelWithTXT(experimentType, device, keepOriginalStructure=False):
    
    model = models.vgg16(pretrained=True)
    print('model', model.classifier)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    if keepOriginalStructure:
        print('keepingOriginalStructure')
        n_inputs = model.classifier[6].in_features
        model.classifier[6] = getFullyConnectedStructure(n_inputs, 2, experimentType)
    else:
        n_inputs = model.classifier[0].in_features
        model.classifier = getFullyConnectedStructure(n_inputs, 2, experimentType)

    print('custom fc', model.classifier)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params, 'total_trainable_params', total_trainable_params)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    return model.to(device)

def prepareResnetModelWithTXT(experimentType, device):
    #TODO in future: treinamento completo a partir do zero aqui
    model = models.resnet50(pretrained=True)
    print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    #print('model = ', model)

    # Add on classifier
    model.fc = getFullyConnectedStructure(n_inputs, 2, experimentType)
    print('custom fc', model.fc)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params, 'total_trainable_params', total_trainable_params)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    return model.to(device)

def prepareDensenetModelWithTXT(experimentType, device):
    model = models.densenet201(pretrained=True)
    print('model', model.classifier)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    n_inputs = model.classifier.in_features

    #print('model = ', model)
    # out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
    # Add on classifier
    model.classifier = getFullyConnectedStructure(n_inputs, 2, experimentType)
    print('custom fc', model.classifier)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print('total_params', total_params, 'total_trainable_params', total_trainable_params)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    
    return model.to(device)
