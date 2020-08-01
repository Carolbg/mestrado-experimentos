#!/usr/bin/env python3

import sys
sys.path.append('../')

from mainVGG import *
from mainResnet import *
from mainDensenet import *

resultsPlotName = '3camadas_ex6_comDropout_lr'
experimentType = 6
dataAugmentation = False
typeLR = 1

#trainLoader, testLoader, validationLoader, n_classes, cat_df, batch_size, max_epochs_stop, n_epochs = mainPrepareDictionaryData(dataAugmentation)

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainVGG(resultsPlotName, experimentType, dataAugmentation, typeLR)
except:
    print('Error in vgg')

try:
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainResnet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except:
    print('Error in resnet')

try: 
    model, history, historyTest, cmTrain, cmValidation, cmTest, trainLoader, testLoader, validationLoader, n_classes, cat_df = mainDensenet(resultsPlotName, experimentType, dataAugmentation, typeLR)
except:
    print('Error in densenet')
