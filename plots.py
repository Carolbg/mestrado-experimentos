import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import itertools
import numpy as np
import pandas as pd
from numpy import savetxt

def plotComparative(history, item1, item2, saveName, xlabel, ylabel, title, labels):
    fig = plt.figure(figsize=(8, 2))
    for c in [item1, item2]:
        plt.plot(
            history[c], label=labels[c])
    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(saveName + '.png')

def plotLosses(history, model):
    xlabel = 'Época'
    ylabel = 'CrossEntropyLoss'
    title = 'Losses treinamento e validação'

    saveName = 'plotLosses_' + model
    labels = {
        'train_loss': 'Loss treinamento', 
        'valid_loss': 'Loss validacao'
    }
    plotComparative(history, 'train_loss', 'valid_loss', saveName, xlabel, ylabel, title, labels)


def plotAcc(history, model):
    xlabel = 'Época'
    ylabel = 'Acurácia'
    title = 'Acurácia treinamento e validação'

    saveName = 'plotAcc_'+model
    # print('saveName', saveName)

    labels = {
        'train_acc': 'Acurácia treinamento', 
        'validation_acc': 'Acurácia validação'
    }
    plotComparative(history, 'train_acc', 'validation_acc', saveName, xlabel, ylabel, title, labels)

def plotSensitividade(history, model):
    xlabel = 'Época'
    ylabel = 'Sensitividade'
    title = 'Sensitividade treinamento e validação'

    saveName = 'plotSensitividade_'+model
    labels = {
        'train_sensitividade': 'Sensitividade treinamento',
        'validation_sensitividade': 'Sensitividade validação'
    }
    plotComparative(history, 'train_sensitividade', 'validation_sensitividade', saveName, xlabel, ylabel, title,labels)

def plotEspecificidade(history, model):
    xlabel = 'Época'
    ylabel = 'Especificidade'
    title = 'Especificidade treinamento e validação'

    saveName = 'plotEspecificidade_' + model
    labels = {
        'train_especificidade': 'Especificidade treinamento',
        'validation_especificidade': 'Especificidade validação'
    }
    plotComparative(history, 'train_especificidade', 'validation_especificidade', saveName, xlabel, ylabel, title, labels)

def plotF1Score(history, model):
    xlabel = 'Época'
    ylabel = 'F1Score'
    title = 'F1Score treinamento e validação'
    saveName = 'plotF1Score_' + model
    labels = {
        'train_f1Score': 'F1-Score treinamento',
        'validation_f1Score': 'F1-Score validação'
    }
    plotComparative(history, 'train_f1Score', 'validation_f1Score', saveName, xlabel, ylabel, title, labels)   


def plotSingleSet(data, saveName, xlabel, ylabel, title, labels):
    fig = plt.figure(figsize=(8, 2))
    plt.plot(data, label=labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    fig.savefig(saveName + '.png')

def plotF1ScoreSingleSet(history, model):
    xlabel = 'Epoch'
    ylabel = 'F1Score'
    title = 'F1Score validation'
    saveName = 'plotSingleF1Score' + model
    plotSingleSet(history['validation_f1Score'], saveName, xlabel, ylabel, title, 'F1-Score validation')   


def plotLossSingleSet(history, model):
    xlabel = 'Number of epochs'
    ylabel = 'loss'
    title = 'Validation loss'
    saveName = 'plotSingleLoss' + model
    plotSingleSet(history['valid_loss'], saveName, xlabel, ylabel, title, '')   


def plotAUC(fpr, tpr, roc_auc, model):
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkslategrey',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1 - Specificity')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    fig.savefig('plotAUC_testingSet_'+ model + '.png')


def plotData(history, model):
    # print('model', model)
    plotAcc(history, model)
    plotSensitividade(history, model)
    plotEspecificidade(history, model)
    plotF1Score(history, model)
    plotLosses(history, model)
    plotF1ScoreSingleSet(history, model)
    plotLossSingleSet(history, model)

def plotTestingAcc(results, model):
    # Plot using seaborn
    fig = sns.lmplot(y='acuracia', x='Treinamento', data=results, height=6)
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)
    fig.savefig('plotTestingAcc_'+model+'.png')

def plotTransformedImageAndHistogram(images, i, typeImg):
    inputs = images[0]
    inputs = inputs.permute(1, 2, 0)
    numpyImage = inputs.numpy()
    fig, (ax1, ax2) =  plt.subplots(1, 2)
    pos = ax1.imshow(numpyImage, cmap='gray')
    fig.colorbar(pos, ax=ax1)
    print('min = ', numpyImage.min())
    print('max = ', numpyImage.max())
    # fig = plt.figure()
    # plt.title('Imagens e histograma')
    # plt.subplot(1, 2, 1)
    # plt.imshow(numpyImage)
    # plt.colorbar(numpyImage)
    
    #plt.subplot(1, 2, 2)
    pos2 = ax2.hist(numpyImage[0], range=[0,2])

    ax2.set_title('Histograma imagem apos o pre-processamento')
    ax2.set_xlabel('Valores dos pixels')
    ax2.set_ylabel('Quantidade')

    fig.savefig(typeImg + '_transformeImg_' + str(i) +'.png')
    #teste = teste.flatten()
    #savetxt(typeImg+'data_' + str(i) +'_.csv', teste, delimiter=',')
    plt.close()

def plotTestTransformedImages(numpyImage, name):

    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.3)
    
    # show original image
    fig.add_subplot(121)
    plt.title('Imagem \npre-processada')
    #plt.set_cmap('gray')
    pos = plt.imshow(numpyImage)
    plt.colorbar(pos)

    fig.add_subplot(122)
    plt.title('Histograma')
    plt.xlabel('Valores dos pixels')
    plt.ylabel('Quantidade')
    #print('numpyImage aquiii', numpyImage.shape)
    plt.hist(numpyImage)
    #plt.xticks(np.arange(0, 2.25, 0.25))

    fig.savefig(name +'.png')


def plotAllSubsetImages(images, typeImg, mean, std):
    print('mean', mean, 'std', std)
    numberImages = images.shape
    print('shape',numberImages[0])
    for i in range(numberImages[0]):
        numpyImage = np.transpose(images[i], (1, 2, 0))
        
        numpyImage = numpyImage.numpy()
        # #Removendo a normalização
        meanArray = np.array([mean, mean, mean])
        stdArray = np.array([std, std, std])
        numpyImage = stdArray * numpyImage + meanArray

        fig = plt.figure(figsize=(10, 4))
        fig.subplots_adjust(wspace=0.3)
        
        # show original image
        fig.add_subplot(121)
        plt.title('Imagens depois min max')
        pos = plt.imshow(numpyImage)
        plt.colorbar(pos)

        fig.add_subplot(122)
        plt.title('Histograma')
        plt.xlabel('Valores dos pixels')
        plt.ylabel('Quantidade')
        plt.hist(numpyImage.flatten())
        plt.xticks(np.arange(0, 2.25, 0.25))
        fig.savefig(typeImg + '_imagens_histograma_' + str(i) +'.png')
        plt.pause(0.001)

def plotTransformedImages(images, i, typeImg, mean, std):

    inputs = images[0]

    print('images[0]', type(images[0]))
    inputs = inputs.permute(1, 2, 0)
    numpyImage = inputs.numpy()
    #Removendo a normalização
    mean = np.array([mean, mean, mean])
    std = np.array([std, std, std])
    numpyImage = std * numpyImage + mean
    # numpyImage = np.clip(numpyImage, 0, 1)

    print('1 - min', np.min(numpyImage))
    print('1 - max', np.max(numpyImage))
    print('numpyImage shape', numpyImage.shape)

    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(wspace=0.5)
    # fig.subplots_adjust(wspace=0.3)
    
    # show original image
    fig.add_subplot(121)
    plt.title('Imagem \npre-processada')
    pos = plt.imshow(numpyImage)
    # plt.set_cmap('gray')
    # pos = plt.imshow(numpyImage, cmap='gray')
    
    plt.colorbar(pos)

    fig.add_subplot(122)
    plt.title('Histograma')
    plt.xlabel('Valores dos pixels')
    plt.ylabel('Quantidade')
    #print('teste', numpyImage.shape)
    teste = numpyImage[:, :, 0]
    #print('teste', teste.shape)
    plt.hist(numpyImage.flatten(), range=[0,1])
    plt.xticks(np.arange(0, 1, 0.1))
    # plt.hist(teste, range=[0,2])
    # plt.xticks(np.arange(0, 2.25, 0.25))


    # fig.add_subplot(133)
    # plt.title('Histograma flatten')
    # plt.xlabel('Valores dos pixels')
    # plt.ylabel('Quantidade')
    # print('numpyImage.flatten()', numpyImage.flatten().shape)
    # plt.hist(numpyImage.flatten(), range=[0,2])
    # plt.xticks(np.arange(0, 2.25, 0.1))

    fig.savefig(typeImg + '_imagens_histograma_' + str(i) +'.png')
    plt.pause(0.001)

def plotHistogram(image, i):
    fig = plt.figure()
    plt.hist(image[0], range=[0,2])
    plt.xlabel('Valores dos pixels')
    plt.ylabel('Quantidade')
    plt.title('Histograma das imagens depois do pre processamento')
    fig.savefig('histogram' + str(i) +'.png')

def prepareAllDF(test, trainValidation):
    print('trainValidation', trainValidation)
    print('test', test)
    all = trainValidation.copy()
    print('all 1', all)
    all = all.assign(test=test.acuracia)
    print('all 2', all)
    all = all.assign(test=test.loss)
    print('all 3', all)

def plotAll(test, trainValidation):

    fig = plt.figure(figsize=(8, 2))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(100 * trainValidation[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    fig.savefig('plotAcc.png')

def plotFilteredImage(image, filteredImage, nameFile):
    fig = plt.figure()
    plt.title('Median Filter')
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(image)
    ax2.imshow(filteredImage)
    #plt.colorbar()
    fig.savefig('median_filter' + nameFile + '.png')
    
    plt.close()
    # fig = plt.figure()
    # plt.gray() 
    # plt.title('Median Filter')
    # ax1 = fig.add_subplot(121)  # left side
    # ax2 = fig.add_subplot(122)  # right side
    # ax1.imshow(image)
    # ax2.imshow(filteredImage)
    # fig.savefig('gray_median_filter' + str(index) + '.png')

def plotImageDataFromPatient(patientData, patientId):
    fig = plt.figure()
    plt.title('Patient Filter')
    i = 0
    values = patientData[patientId]
    for image in values:
        fig = plt.figure()
        plt.imshow(image)
        plt.colorbar()
        i = i+1
        fig.savefig('image_'+ patientId + '_'+ str(i) + '.png')
        
        plt.close()