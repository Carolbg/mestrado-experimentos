# Data science tools
import numpy as np
import glob
import math
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import pandas as pd
from plots import plotTransformedImages
from customDatasetFromNumpyArray import CustomDatasetFromNumpyArray 

def getFilesName():
    print('getFilesName')
    #txt_saudaveis_files = glob.glob("../Imagens_TXT_Estaticas/0Saudavel/*.txt")
    #txt_doentes_files = glob.glob("../Imagens_TXT_Estaticas/1Doente/*.txt")
    txt_saudaveis_files = glob.glob("../poucas_Imagens/0Saudavel/*.txt")
    txt_doentes_files = glob.glob("../poucas_Imagens/1Doente/*.txt")

    return txt_saudaveis_files, txt_doentes_files

def readFiles(txt_files):
    print('readFiles')
    data = []
    for i in range(len(txt_files)):
        inputData = np.loadtxt(txt_files[i], dtype='f', delimiter=' ')
        data.append(inputData)
    return data

def prepareDataFromTXT():
    print('prepareDataFromTXT')
    txt_saudaveis_files, txt_doentes_files= getFilesName()
    saudaveisData = readFiles(txt_saudaveis_files)
    doentesData = readFiles(txt_doentes_files)
    saudaveisTarget = np.full(len(saudaveisData), 0)
    doentesTarget = np.full(len(doentesData), 1)
    data = np.concatenate((saudaveisData,doentesData), axis=0)
    dataTarget = np.concatenate((saudaveisTarget, doentesTarget), axis=0)
    return data, dataTarget

def splitDataset(dataset, shuffleSeed):
    print('splitDataset\n')
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


def prepareNumpyDataset(data, dataTarget, train_idx, test_idx, valid_idx, batch_size):
    print('prepareNumpyDataset')

    defaultTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0))  # Imagenet standards  # Imagenet standards
    ])

    customTransform = transforms.Compose([
        transforms.ToPILImage(),
        #transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30, fill=(0,)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards  # Imagenet standards
    ])
    
    testValidationTransform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
    ])
    
    trainDataset = CustomDatasetFromNumpyArray(data[train_idx],dataTarget[train_idx], defaultTransform)
    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)

    testDataset = CustomDatasetFromNumpyArray(data[test_idx],dataTarget[test_idx], testValidationTransform)
    testLoader = DataLoader(testDataset, batch_size=batch_size, shuffle=True)

    validationDataset = CustomDatasetFromNumpyArray(data[valid_idx],dataTarget[valid_idx], testValidationTransform)
    validationLoader = DataLoader(validationDataset, batch_size=batch_size, shuffle=True)

    resultLabelsTraining = torch.zeros(2, dtype=torch.long)

    #for batch_idx, (data, target) in enumerate(trainLoader):
    #    print('Batch idx {}, data shape {}, target shape {}'.format(batch_idx, data.shape, target.shape))

    i=0
    for images, labels in iter(trainLoader):
        l = labels.numpy()
        resultLabelsTraining[0] = resultLabelsTraining[0] + np.count_nonzero(l == 0)
        resultLabelsTraining[1] = resultLabelsTraining[1] + np.count_nonzero(l == 1)
        if i<40:
            plotTransformedImages(images, i, 'traininig_only_handleRBG')
        i = i+1
    #print('Treinamento resultLabels', resultLabelsTraining)

    resultLabelsTesting = torch.zeros(2, dtype=torch.long)
    for images, labels in iter(testLoader):
        l = labels.numpy()
        resultLabelsTesting[0] = resultLabelsTesting[0] + np.count_nonzero(l == 0)
        resultLabelsTesting[1] = resultLabelsTesting[1] + np.count_nonzero(l == 1)

    #print('Treinamento + Teste resultLabels', resultLabelsTesting)

    resultLabelsValidation = torch.zeros(2, dtype=torch.long)
    i = 0
    for images, labels in iter(validationLoader):
        l = labels.numpy()
        resultLabelsValidation[0] = resultLabelsValidation[0] + np.count_nonzero(l == 0)
        resultLabelsValidation[1] = resultLabelsValidation[1] + np.count_nonzero(l == 1)
        #plotTransformedImages(images, i, 'validation')
        i = i+1
    #print('Treinamento + Teste + Validation resultLabels', resultLabelsValidation)

    # Dataframe of categories
    cat_df = pd.DataFrame({
        'category': ['Saudável', 'Doente'],
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
    return trainLoader, testLoader, validationLoader, n_classes, cat_df

def getCommonArgs():
    shuffleSeed = 3
    batch_size = 10
    max_epochs_stop = 30
    n_epochs = 30
    print('n_epochs', n_epochs)
    return shuffleSeed, batch_size, max_epochs_stop, n_epochs