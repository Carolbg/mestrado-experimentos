import matplotlib.pyplot as plt
from utilsParams import getDevice

def setInfoData(aTrainData, aTrainTarget, aTestData, aTestTarget, aValidationData, aValidationTarget):
    global trainData, trainTarget, testData, testTarget, validationData, validationTarget
    trainData, trainTarget, testData, testTarget, validationData, validationTarget = aTrainData, aTrainTarget, aTestData, aTestTarget, aValidationData, aValidationTarget

def getData():
    return trainData, trainTarget, testData, testTarget, validationData, validationTarget

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def visualizeFirstConvResnet(model, testLoader):
    model.conv1.register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'firstLayerResnet', 'First conv layer Resnet')

def visualizeMiddleResnet(model, testLoader):
    model.layer2[0].register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'middleLayerResnet', 'Middle conv layer Resnet')

def visualizeLastLayerResnet(model, testLoader):
    model.layer4[0].register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'lastLayerResnet', 'Last conv layer Resnet')

def visualizeFirstLayerVGG(model, testLoader):
    model.features[0].register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'firstLayerVGG', 'First conv layer VGG')

def visualizeLastLayerVGG(model, testLoader):
    model.features[28].register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'lastLayerVGG', 'Last conv layer VGG')

def visualizeMiddleLayerVGG(model, testLoader):
    model.features[14].register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'middleLayerVGG', 'Middle conv layer VGG')

def visualizeFirstLayerDensenet(model, testLoader):
    model.features.conv0.register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'firstLayerDensenet', 'First conv layer Densenet')

def visualizeMiddleLayerDensenet(model, testLoader):
    model.features.denseblock1.denselayer1.conv1.register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'middleLayerDensenet', 'Middle conv layer Densenet')

def visualizeLastLayerDensenet(model, testLoader):
    model.features.denseblock4.denselayer32.conv1.register_forward_hook(get_activation('Conv2d'))
    visualizeLayerCNN(model, testLoader, 'lastLayerDensenet', 'Last conv layer Densenet')

def visualizeLayerCNN(model, testLoader, figName, title):
    plt.rcParams.update({'font.size': 32})
    for batch_idx, (image, target) in enumerate(testLoader):
        # print('batch_idx', batch_idx)
        image = image.to('cuda:0')
        output = model(image)

        act = activation['Conv2d'].squeeze() 
        print('act.size', act.size())

        plt.figure()
        fig, axarr = plt.subplots(act.size(0), 4, figsize=(50, 50))
        st = fig.suptitle(title)
        # print('act', act.size())
        for idx in range(act.size(0)):
            # axarr[idx, 0].imshow(act[idx][0], aspect="auto")
            axarr[idx, 0].imshow(act[idx][0].detach().cpu().numpy(),  interpolation='none')
            axarr[idx, 1].imshow(act[idx][1].detach().cpu().numpy(),  interpolation='none')
            axarr[idx, 2].imshow(act[idx][2].detach().cpu().numpy(),  interpolation='none')
            axarr[idx, 3].imshow(act[idx][3].detach().cpu().numpy(),  interpolation='none')
    
        fig.savefig('features_'+ figName + str(batch_idx) + '.png')
                

