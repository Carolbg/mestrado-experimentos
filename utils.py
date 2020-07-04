from sklearn.metrics import confusion_matrix, f1_score
import csv


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
    return train_acc, train_especificidade, train_sensitividade, f1Score, cm

def getConfusionMatrixInfo(y_true, y_pred):
    #print('target', y_true, 'predicted', y_pred)
    # Neste cenario, 1 eh doente e 0 saudavel
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    #plot_confusion_matrix(cm, normalize = False, target_names = ['saudavel', 'Doente'], title = 'Confusion Matrix')
    tn, fp, fn, tp = cm.ravel()
    #print('tn, fp, fn, tp ', tn, fp, fn, tp )
    #print('Confusion Matrix = ', cm)
    return tn, fp, fn, tp , cm

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