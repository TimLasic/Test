import cv2 as cv
import numpy as np


img = cv.imread('pepe.jpg') #BGR
cv.imshow('pepe',img)

blank = np.zeros(img.shape[:2],dtype='uint8') # uint8 datatype of an image [:2] prvi 2 vrednosti
cv.imshow('Blank',blank)



circle = cv.circle(blank.copy(),(img.shape[1]//2 ,img.shape[0]//2),100, 255, -1)
cv.imshow('circle',circle)

#rectangle = cv.rectangle(blank.copy(),(30,30), (370,370), 255,-1)
#weird_mask = cv.bitwise_and(rectangle,rectangle)
#cv.imshow('weird_mask',weird_mask)

mask = cv.rectangle(blank.copy(),(img.shape[1]//2,img.shape[0]//2),(img.shape[1]//2 + 100,img.shape[0]//2 + 100), 255, -1)
cv.imshow('mask',mask)

masked_image = cv.bitwise_and(img,img,mask=mask)
cv.imshow('masked_image',masked_image)



cv.waitKey(0)