import numpy as np
import pandas as pd
import torch

def accuracy(output, target):
    """Compute the topk accuracy(s)"""

    with torch.no_grad():
        batch_size = target.size(0)
        print('batch_size', batch_size)
        # Find the predicted classes and transpose
        _, pred = output.topk(k=1, dim=1, largest=True, sorted=True)
        #print('pred', pred)
        
        pred = pred.t()
        #print('predT', pred)
        
        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        #print('correct', correct)

        # Find the percentage of correct
        
        correct_k = correct[:1].view(-1).float().sum(0, keepdim=True)
        #print('correct_k', correct_k)
        res = (correct_k.mul_(100.0 / batch_size).item())
        #print('res', res)
        return res


def evaluate(model, test_loader, criterion):
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

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros(len(test_loader.dataset))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:
            
            # Raw model output
            out = model(data)
            print('out = ', out)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk (top 1) accuracy
                #print('pred = ', pred)
                #print('true = ', true)
                
                acc_results[i] = accuracy(pred.unsqueeze(0), true.unsqueeze(0))
                #print('acc_results', acc_results)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                i += 1
    
    #print('acc_results', acc_results)
    # Send results to a dataframe and calculate average across classes
    #results = pd.DataFrame(acc_results)
    #results['class'] = classes
    #results['loss'] = losses
    results = pd.DataFrame({'acuracia': acc_results, 'class':classes, 'loss': losses})
    results = results.groupby(classes).mean()
    #print('results', results)
    return results.reset_index().rename(columns={'index': 'class'})
