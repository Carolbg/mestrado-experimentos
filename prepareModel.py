from torchvision import transforms, datasets, models
import torch.nn as nn
from utilsParams import *

def prepareVGG16ModelWithTXT(n_classes):
    
    model = models.vgg16(pretrained=True)
    print('model', model.classifier)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    #n_inputs = model.classifier[6].in_features
    n_inputs = model.classifier[0].in_features

    # Add on classifier
    #model.classifier[6] = getFullyConnectedStructure(n_inputs, n_classes)
    model.classifier = getFullyConnectedStructure(n_inputs, n_classes)
    print('custom fc', model.classifier)
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)

    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    return model


def prepareResnetModelWithTXT(n_classes):
    model = models.resnet50(pretrained=True)
    print('model', model.fc)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features

    # Add on classifier
    model.fc = getFullyConnectedStructure(n_inputs, n_classes)
    print('custom fc', model.fc)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    return model


def prepareDensenetModelWithTXT(n_classes):
    model = models.densenet201(pretrained=True)
    print('model', model.classifier)
    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False
    
    n_inputs = model.classifier.in_features

    # Add on classifier
    model.classifier = getFullyConnectedStructure(n_inputs, n_classes)
    print('custom fc', model.classifier)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    
    model.idx_to_class = {0: 'Saudavel', 1: 'Doente'}
    
    return model

