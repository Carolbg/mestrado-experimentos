# PyTorch
from torchvision import transforms, datasets, models
import torch
from torch import optim
from torch.utils.data import DataLoader, sampler, SubsetRandomSampler, Subset, Dataset
import torch.nn as nn

# Data science tools
import numpy as np
import pandas as pd
import math

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from customDataset import CustomDataset

def readDataset(imageFolder):
    dataset = datasets.ImageFolder(imageFolder)
    #print(dataset)

    return dataset

def splitDataset(dataset, shuffleSeed):
    totalDataset = len(dataset)

    trainTotal = math.ceil(totalDataset*0.70)
    testTotal = math.ceil((totalDataset - trainTotal)/2)
    validationTotal = totalDataset - trainTotal - testTotal

    indices = list(range(totalDataset))

    np.random.seed(shuffleSeed)
    np.random.shuffle(indices)

    train_idx, test_idx, valid_idx = indices[:trainTotal], indices[trainTotal:trainTotal+testTotal], indices[trainTotal+testTotal:]
    print('Quantidade de dados para treinamento',len(train_idx))
    print('Quantidade de dados para teste',len(test_idx))
    print('Quantidade de dados para validacao',len(valid_idx))

    return train_idx, test_idx, valid_idx

def prepareDataset(dataset, train_idx, test_idx, valid_idx, batch_size):
    train_sampler = Subset(dataset, train_idx)
    validation_sampler = Subset(dataset, valid_idx)
    test_sampler = Subset(dataset, test_idx)

    trainTransform = transforms.Compose([
            transforms.RandomResizedCrop(size=256),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=224),  # Image net standards
            transforms.ToTensor()
    ])
        
    testValidationTransform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

            
    trainMapDataset = CustomDataset(train_sampler, trainTransform)
    trainLoader = DataLoader(trainMapDataset, batch_size=batch_size, shuffle=True)

    testMapDataset = CustomDataset(test_sampler, testValidationTransform)
    testLoader = DataLoader(testMapDataset, batch_size=batch_size, shuffle=True)

    validationMapDataset = CustomDataset(validation_sampler, testValidationTransform)
    validationLoader = DataLoader(validationMapDataset, batch_size=batch_size, shuffle=True)


    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
    #print('Treinamento resultLabels', resultLabelsTraining)

    resultLabelsTesting = torch.zeros(2, dtype=torch.long)
    for images, labels in iter(testLoader):
        l = labels.numpy()
        resultLabelsTesting[0] = resultLabelsTesting[0] + np.count_nonzero(l == 0)
        resultLabelsTesting[1] = resultLabelsTesting[1] + np.count_nonzero(l == 1)
    #print('Treinamento + Teste resultLabels', resultLabelsTesting)

    resultLabelsValidation = torch.zeros(2, dtype=torch.long)
    for images, labels in iter(validationLoader):
        l = labels.numpy()
        resultLabelsValidation[0] = resultLabelsValidation[0] + np.count_nonzero(l == 0)
        resultLabelsValidation[1] = resultLabelsValidation[1] + np.count_nonzero(l == 1)
    #print('Treinamento + Teste + Validation resultLabels', resultLabelsValidation)

    # Dataframe of categories
    cat_df = pd.DataFrame({
                            'category': ['Cancer', 'Normal'],
                            'Treinamento': resultLabelsTraining,
                            'Validação': resultLabelsValidation, 
                            'Teste': resultLabelsTesting
                        })
    print(cat_df)

    # Plot the images of the training dataset
    #for images, labels in iter(train_loader):
    #    i = i+1
    #    transpose = np.transpose(images[0].numpy(), (1, 2, 0))
    #    imgplot = plt.imshow(transpose)
    #    plt.show()

    #cat_df.set_index('category')['n_train'].plot.bar(
    #    color='r', figsize=(20, 6))
    #plt.xticks(rotation=80)
    #plt.ylabel('Count')
    #plt.title('Training Images by Category')


    #trainiter = iter(trainLoader)
    #features, labels = next(trainiter)
    #print('features.shape', features.shape, 'labels.shape', labels.shape)

    n_classes = len(cat_df)
    #print(f'There are {n_classes} different classes.')
    return trainLoader, testLoader, validationLoader, n_classes