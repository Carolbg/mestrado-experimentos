#!/usr/bin/env python3

import sys
sys.path.append('../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

resultsPlotName = 'dataAugmentation_matlabPreprocessing_3camadas_ex6_comDropout_lr'
experimentType = 6
dataAugmentation = True
typeLR = 1

print('Config: ', resultsPlotName)
print('experimentType', experimentType, ' dataAugmentation ',dataAugmentation, ' typeLR ', typeLR)

#trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryData(dataAugmentation)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in vgg', e)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in resnet', e)

try: 
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except Exception as e:
    print('Error in densenet', e)
