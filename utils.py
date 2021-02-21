from sklearn.metrics import confusion_matrix, f1_score, precision_score, roc_curve, auc
import csv
import torch
from plots import plotAUC

# def calculateConfusionMatrix(y_true,y_pred):
#     confMatrix = confusion_matrix(y_true, y_pred, labels=[0,1])
#     plotConfusionMatrix(confMatrix)
#     input('aqui')
#     return confMatrix

def calcMetrics(targetData, predictedData):
    tn, fp, fn, tp, cm = getConfusionMatrixInfo(targetData, predictedData)
    #saudavel
    train_especificidade = calcEspecificidade(tn, fp)

    #Doente
    train_sensitividade = calcSensitividade(tp, fn)
    train_acc = calcAcc(tn, fp, fn, tp)
    f1Score = getF1Score(targetData, predictedData)
    precision = getPrecision(targetData, predictedData)
    return train_acc, train_especificidade, train_sensitividade, f1Score, cm, precision

def getConfusionMatrixInfo(y_true, y_pred):
    #print('target', y_true, 'predicted', y_pred)
    # Neste cenario, 1 eh doente e 0 saudavel
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    #plot_confusion_matrix(cm, normalize = False, target_names = ['saudavel', 'Doente'], title = 'Confusion Matrix')
    tn, fp, fn, tp = cm.ravel()
    #print('tn, fp, fn, tp ', tn, fp, fn, tp )
    #print('Confusion Matrix = ', cm)
    return tn, fp, fn, tp , cm

def getPrecision(target, predicted):
    return precision_score(target, predicted, labels=[0,1])

def getF1Score(target, predicted):
    return f1_score(target, predicted, labels=[0,1])

def calcEspecificidade(tn, fp) :
    espe = tn / (tn+fp)
    return espe

def calcSensitividade(tp, fn):
    sensi = tp / (tp+fn)
    return sensi

def calcAcc(tn, fp, fn, tp):
    acc = (tp+tn)/(tn + fp + fn + tp)
    return acc

def calcROC(target, predicted, modelName):
    fpr, tpr, thresholds = roc_curve(target, predicted)
    print('fpr, tpr, thresholds', fpr, tpr, thresholds)
    roc_auc = auc(fpr, tpr)
    print('roc_auc', roc_auc)

    plotAUC(fpr, tpr, roc_auc, modelName)
    
    return roc_auc, fpr, tpr, thresholds

def saveCsvConfusionMatrix(confusionMatrix, resultsPlotName):
    tn, fp, fn, tp = confusionMatrix.ravel()
    print('tn', tn)
    print('fp', fp)
    print('fn', fn)
    print('tp', tp)

    with open('confusion_'+resultsPlotName+'.csv', mode='w') as cm_file:
        cm_writer = csv.writer(cm_file, delimiter=',')
        cm_writer.writerow(['-','0 Saudavel', '1 Doente'])
        cm_writer.writerow(['0', tn, fn])
        cm_writer.writerow(['1',fp, tp])

def convertToNumpy(data):
    hasGpu = torch.cuda.is_available()
    if not hasGpu:
        return data.numpy()
    return data.cpu().numpy()
