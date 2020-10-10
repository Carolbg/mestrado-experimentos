#!/usr/bin/env python3

import sys
sys.path.append('../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *
from readMatlabNumpyData import mainPrepareDictionaryDataFromNumpy

resultsPlotName = 'matlabPreprocessing_1camada_lr'
#resultsPlotName = '1camada_lr'
experimentType = 1
dataAugmentation = False
typeLR = 1

trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs, device = mainPrepareDictionaryData(dataAugmentation)


# try:
#     model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR)
# except Exception as e:
#     print('Error in vgg', e)

# try:
#     model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR)
# except Exception as e:
#     print('Error in resnet', e)

# try: 
#     model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR)
# except Exception as e:
#     print('Error in densenet', e)
