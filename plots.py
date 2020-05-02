import matplotlib.pyplot as plt
import seaborn as sns

def plotLosses(history, model):
    fig = plt.figure(figsize=(8, 2))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    fig.savefig('plotLosses_'+model+'.png')

def plotAcc(history, model):
    fig = plt.figure(figsize=(8, 2))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    fig.savefig('plotAcc_'+model+'.png')

def plotTestingAcc(results, model):
    # Plot using seaborn
    fig = sns.lmplot(y='acuracia', x='Treinamento', data=results, height=6)
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)
    fig.savefig('plotTestingAcc_'+model+'.png')


def plotTransformedImages(images, i, typeImg):
    inputs = images[0]
    inputs = inputs.permute(1, 2, 0)
    fig = plt.figure()
    plt.imshow(inputs.numpy())
    fig.savefig(typeImg + '_transformeImg_' + str(i) +'.png')
    plt.close()

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
