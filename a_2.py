# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 01:19:16 2020

@author: K555D
"""

import cv2
import numpy as np

img = cv2.imread("test.jpg", 1)

# Resizing image in order to be divisible into blocks.
img = cv2.resize(img, (256, 128))

# Define HOGDescriptor parameters:
winSize = (256, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9
derivAperture = 1
winSigma = 8
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64
signedGradient = 0

hog = cv2.HOGDescriptor(winSize,
                        blockSize,
                        blockStride,
                        cellSize,
                        nbins,
                        derivAperture,
                        winSigma,
                        histogramNormType,
                        L2HysThreshold,
                        gammaCorrection,
                        nlevels,
                        signedGradient
                        )
hist = hog.compute(img)

# Save HOG features:
np.save("HOG_features", hist)
    