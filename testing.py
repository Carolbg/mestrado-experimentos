import numpy as np
import pandas as pd
import torch
from utils import calcMetrics

dataClasses = ('Saudavel', 'Doente')

def accuracy(output, target):
    """Compute the topk accuracy(s)"""

    with torch.no_grad():
        batch_size = target.size(0)
        # Find the predicted classes and transpose
        _, pred = torch.max(output, dim=1)
        #_, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
    
        #pred = pred.t()
        
        # Determine predictions equal to the targets
        correct = pred == target.data#pred.eq(target.view(1, -1).expand_as(pred))
        # Find the percentage of correct
        #correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        #res = (correct_k.mul_(100.0 / batch_size).item())
        return pred #res, pred

def evaluate(model, test_loader, criterion, n_classes, resultsPlotName, device):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    #classes = []
    losses = 0.0
    # Hold accuracy results
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0
    # test_error_count = 0.0
    
    allTestingTarget = []
    allTestingPredicted = []
    with torch.no_grad():
        model.eval()
        for data, target in test_loader:
            # Forward pass
            data = data.to(device)
            target = target.to(device)
                
            output = model(data)
            # Calculate validation accuracy
            values, pred = torch.max(output, 1)
            loss = criterion(output, target)
            
            #print('values', values)
            #print('pred', pred)
            #print('target.data', target.data)
            # Multiply average loss times the number of examples in batch
            losses += loss.item() * data.size(0)
            allTestingPredicted = np.concatenate((allTestingPredicted, pred.numpy()), axis=0)
            allTestingTarget = np.concatenate((allTestingTarget, target.numpy()), axis=0)
            
    test_acc, test_especificidade, test_sensitividade, test_f1Score, cmTest = calcMetrics(allTestingTarget, allTestingPredicted)
    history = pd.DataFrame({
        'test_acc': [test_acc], 'test_sensitividade': [test_sensitividade], 
        'test_especificidade': [test_especificidade], 'test_f1Score': [test_f1Score]})
    print('\nTesting result\n', history)

    history.to_csv('results_'+resultsPlotName+'.csv', index = False, header=True)
    
    losses = losses / len(test_loader.dataset)
    print('TestLoader Losses', losses)

    return history, cmTest
