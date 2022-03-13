import cv2 as cv
import numpy as np


img = cv.imread('pepe.jpg') #BGR
cv.imshow('pepe',img)

#filter2d

cv.waitKey(0)