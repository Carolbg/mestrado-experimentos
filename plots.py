import matplotlib.pyplot as plt
import seaborn as sns

def plotLosses(history):
    fig = plt.figure(figsize=(8, 2))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    fig.savefig('plotLosses.png')

def plotAcc(history):
    fig = plt.figure(figsize=(8, 2))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    fig.savefig('plotAcc.png')

def plotTestingAcc(results):
    # Plot using seaborn
    fig = sns.lmplot(y='acuracia', x='Treinamento', data=results, height=6)
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)
    fig.savefig('plotTestingAcc.png')