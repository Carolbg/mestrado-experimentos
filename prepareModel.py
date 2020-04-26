from torchvision import transforms, datasets, models
import torch.nn as nn
from torch import optim

def prepareModel(dataset, n_classes):

    model = models.vgg16(pretrained=True)
    #print('model', model)

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.classifier[6].in_features

    # Add on classifier
    model.classifier[6] = nn.Sequential(
        nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, n_classes), nn.LogSoftmax(dim=1))

    total_params = sum(p.numel() for p in model.parameters())
    #print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')

    #summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')
    #print(model.classifier[6])

    model.class_to_idx = dataset.class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }
    #print('model.idx_to_class', model.idx_to_class)
    return model

def prepareTrainingLoss():
    criterion = nn.NLLLoss()
    return criterion

def prepareTrainingOptimizer(model):
    optimizer = optim.Adam(model.parameters())

    #for p in optimizer.param_groups[0]['params']:
    #    if p.requires_grad:
    #        print(p.shape)
    return optimizer