from ag_cnnFromAG import *
from training import train
from testing import evaluate

def calcFitness(population, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion):
    tp = len(population)

    fitnessArray = [calcFitnessIndividuo(population[i], i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion) for i in range(tp)]
    return fitnessArray
        
def calcFitnessIndividuo(individuo, i, trainLoader, testLoader, validationLoader, cat_df, batch_size, device, criterion):
    
    model, optimizer, epocas = convertAgToCNN(individuo, device)
    print('epocas', epocas)
    resultsPlotName = 'runAG_individuo_'+str(i)
    
    #treinamento
    model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation = train(model, criterion,
        optimizer, trainLoader, validationLoader, resultsPlotName, epocas, epocas, device)

    #teste
    historyTest, cmTest = evaluate(model, testLoader, criterion, 2, resultsPlotName, device)
    testAcc = historyTest['test_acc'][0]  

    print('fitness', testAcc)
    return testAcc