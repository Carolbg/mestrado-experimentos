from skimage.filters import try_all_threshold
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import glob

def try_segmentation_skimage(image):
    fig, ax = try_all_threshold(image, figsize=(10, 6), verbose=True)
    fig.savefig('test_segmentation.png')

def cvSegmentation(originalImage):
    img = np.stack((originalImage,)*3, axis=2)
    #img = cv.cvtColor(inputData, cv.COLOR_BGR2GRAY)

    ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
                cv.THRESH_BINARY,11,2)
    th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv.THRESH_BINARY,11,2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    
    fig = plt.figure(figsize=(10, 4))
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    fig.savefig('cvSegmentation.png')

def trySegmenentation():
    saudaveisSegmentacaoData = mainReadData()
    try_segmentation_skimage(saudaveisSegmentacaoData)
    cvSegmentation(saudaveisSegmentacaoData)

def mainReadData():
    print('\nprepareDataFromTXT')
    testes_segmentacao = glob.glob("../testes_segmentacao/*.txt")
    saudaveisSegmentacaoData = readFiles(testes_segmentacao, 'saudaveis_segmentacao')
    return saudaveisSegmentacaoData

def readFiles(txt_files, patientClass):
    print('readFiles')
    for i in range(len(txt_files)):
        name = txt_files[i].split('/')
        fileName = name[len(name)-1]
        inputData = np.loadtxt(txt_files[i], dtype='f', delimiter=' ')
    return inputData