from timeit import default_timer as timer
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from utils import calcMetrics
import matplotlib.pyplot as plt

def train(model, criterion, optimizer, trainLoader, validLoader, save_file_name,
        max_epochs_stop=3, n_epochs=20, print_every=2):

    # Early stopping intialization
    epochs_no_improve = 0
    valid_loss_min = np.Inf

    valid_max_acc = 0
    history = []

    running_loss_history = []
    running_corrects_history = []
    val_running_loss_history = []
    val_running_corrects_history = []

    overall_start = timer()
    model.epochs = 0

    # Main loop
    for epoch in range(n_epochs):
        #print('Epoca = ', epoch)
        # keep track of training and validation loss each epoch
        train_loss = 0.0
        valid_loss = 0.0
        validation_acc = 0.0
        
        running_loss = 0.0
        running_corrects = 0.0
        val_running_loss = 0.0
        val_running_corrects = 0.0

        allValidationPredicted = []
        allValidationTarget = []
        allTrainingPredicted = []
        allTrainingTarget = []
        start = timer()

        # Training loop
        for i, data in enumerate(trainLoader, 0):
            model.train()
            inputs, labels = data

            optimizer.zero_grad()

            # Predicted outputs are log probabilities
            output = model(inputs)

            # Loss and backpropagation of gradients
            # The losses are averaged across observations for each minibatch.
            loss = criterion(output, labels)
            
            loss.backward()
            # Update the parameters
            optimizer.step()

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, 1)

            # Track train loss by multiplying average loss by number of examples in batch
            train_loss += loss.item() * inputs.size(0)
            
            #curso
            running_corrects += torch.sum(pred == labels.data)
            running_loss += loss.item()
            

            # Neste cenario, 0 eh doente e 1 saudavel
            allTrainingPredicted = np.concatenate((allTrainingPredicted, pred.numpy()), axis=0)
            allTrainingTarget = np.concatenate((allTrainingTarget, labels.numpy()), axis=0)
            
        # After training loops ends, start validation
        
        model.epochs += 1

        # Don't need to keep track of gradients
        with torch.no_grad():
            # Set to evaluation mode
            #model.eval()

            # Validation loop
            for data, target in validLoader:
                # Forward pass
                model.eval()
                output = model(data)

                # Validation loss
                loss = criterion(output, target)
                # Calculate validation accuracy
                values, pred = torch.max(output, dim=1)
                # Multiply average loss times the number of examples in batch
                valid_loss += loss.item() * data.size(0)
                val_running_loss += loss.item()
                val_running_corrects += torch.sum(pred == target.data)

                # Neste cenario, 0 eh doente e 1 saudavel
                allValidationPredicted = np.concatenate((allValidationPredicted, pred.numpy()), axis=0)
                allValidationTarget = np.concatenate((allValidationTarget, target.numpy()), axis=0)
        
            # Calculate average losses
            train_loss = train_loss / len(trainLoader.dataset)
            valid_loss = valid_loss / len(validLoader.dataset)

            # Calculate average accuracy
            train_acc, train_especificidade, train_sensitividade, train_f1Score, cmTrain = calcMetrics(allTrainingTarget, allTrainingPredicted)
            validation_acc, validation_especificidade, validation_sensitividade, validation_f1Score, cmValidation = calcMetrics(allValidationTarget, allValidationPredicted)

            history.append([
                train_acc, validation_acc,
                train_especificidade, validation_especificidade,
                train_sensitividade, validation_sensitividade,
                train_f1Score, validation_f1Score,
                train_loss, valid_loss ])

            # Print training and validation results
            if (epoch) % print_every == 0:
                print(
                    f'\nEpoch: {epoch} \tTraining Loss: {train_loss:.4f} \t\tValidation Loss: {valid_loss:.4f}'
                )
                print(
                    f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * validation_acc:.2f}%'
                )
                print(
                    f'\t\tTraining F1-Score: {train_f1Score:.2f}\t Validation F1-Score: {validation_f1Score:.2f} \n'
                )

            # Save the model if validation loss decreases
            if valid_loss < valid_loss_min:
                # Save model
                #torch.save(model.state_dict(), save_file_name)
                # Track improvement
                epochs_no_improve = 0
                valid_loss_min = valid_loss
                valid_best_acc = validation_acc
                best_epoch = epoch

            # Otherwise increment count of epochs with no improvement
            else:
                epochs_no_improve += 1
                # Trigger early stopping
                if epochs_no_improve >= max_epochs_stop:
                    print(
                        f'\nEarly Stopping! Total epochs: {epoch}. Best epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * validation_acc:.2f}%'
                    )
                    total_time = timer() - overall_start
                    print(
                        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
                    )

                    # Load the best state dict
                    #model.load_state_dict(torch.load(save_file_name))
                    # Attach the optimizer
                    model.optimizer = optimizer

                    # Format history
                    history = pd.DataFrame(
                        history,
                        columns=[
                            'train_acc', 'validation_acc', 
                            'train_especificidade', 'validation_especificidade',
                            'train_sensitividade', 'validation_sensitividade',
                            'train_f1Score', 'validation_f1Score',
                            'train_loss', 'valid_loss'
                        ])
                    return model, history

        epoch_loss = running_loss/len(trainLoader.dataset)
        epoch_acc = running_corrects.float()/len(trainLoader.dataset)

        running_loss_history.append(epoch_loss)
        running_corrects_history.append(epoch_acc)

        val_epoch_loss = val_running_loss/len(validLoader.dataset)
        val_epoch_acc = val_running_corrects.float()/len(validLoader.dataset)
        val_running_loss_history.append(val_epoch_loss)
        val_running_corrects_history.append(val_epoch_acc)
        

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(
        f'\nBest epoch: {best_epoch} with loss: {valid_loss_min:.2f} and acc: {100 * validation_acc:.2f}%'
    )
    print(
        f'{total_time:.2f} total seconds elapsed. {total_time / (epoch+1):.2f} seconds per epoch.'
    )
    # Format history
    history = pd.DataFrame(history, columns=['train_acc', 'validation_acc', 
                                'train_especificidade', 'validation_especificidade',
                                'train_sensitividade', 'validation_sensitividade',
                                'train_f1Score', 'validation_f1Score',
                                'train_loss', 'valid_loss'])
    
    #print('Trained model', model)
    print('\nHistorico treinamento e teste \n', history)

    fig = plt.figure()
    plt.plot(running_corrects_history, label='Training accuracy')
    plt.plot(val_running_corrects_history, label='Validation accuracy')
    plt.legend()
    #plt.show()
    fig.savefig('Corrects history.png')

    fig = plt.figure()
    plt.plot(running_loss_history, label='traininig loss')
    plt.plot(val_running_loss_history, label='validation loss')
    plt.legend()
    #plt.show()
    fig.savefig('Loss history.png')

    return model, history, train_loss, valid_loss, train_acc, validation_acc, valid_best_acc, cmTrain, cmValidation

