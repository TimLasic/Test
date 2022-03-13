import cv2 as cv
import numpy as np



img = cv.imread('pepe.jpg') #BGR
cv.imshow('pepe',img)

blank = np.zeros(img.shape[:2],dtype='uint8') # uint8 datatype of an image [:2] prvi 2 vrednosti
cv.imshow('Blank',blank)

#something = cv.cvtColor(img,cv.COLOR_BGR2RGB)
#cv.imshow('something',something)

b,g,r = cv.split(img)#modra,zelena,rdeča
print(img.shape)
print(b.shape)
print(g.shape)
print(r.shape)

blue = cv.merge([b,blank,blank])
cv.imshow('blue',blue)

merged = cv.merge([b,g,r])
cv.imshow('merged',merged)

cv.waitKey(0)