# Data science tools
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def applyMedianFilter(image):
    filteredImage = ndimage.median_filter(image, size=20)
    return filteredImage

def applyMedianFilterDataset(dataset):
    dataItens= dataset.shape
    filteredDataset = []
    for i in range(dataItens[0]):
        filteredImage = applyMedianFilter(dataset[i])
        filteredDataset.append(filteredImage)
        plot(dataset[i], filteredImage, i)
    return filteredDataset

def plot(image, filteredImage, index):
    fig = plt.figure()
    plt.title('Median Filter')
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(image)
    ax2.imshow(filteredImage)
    fig.savefig('median_filter' + str(index) + '.png')

    fig = plt.figure()
    plt.gray() 
    plt.title('Median Filter')
    ax1 = fig.add_subplot(121)  # left side
    ax2 = fig.add_subplot(122)  # right side
    ax1.imshow(image)
    ax2.imshow(filteredImage)
    fig.savefig('gray_median_filter' + str(index) + '.png')