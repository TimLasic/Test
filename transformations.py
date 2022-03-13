import cv2 as cv
import numpy as np

img = cv.imread('pepe.jpg') #BGR
cv.imshow('pepe',img)

#translacije shiftanje slike po x ali y osi
def translate(img, x, y):
    transMat = np.float32([[1,0,x],[0,1,y]]) #matrika
    dimensions = (img.shape[1],img.shape[0]) #width height
    return cv.warpAffine(img,transMat,dimensions)

# -x --> left
# -y -->up
# x -->right
# y -->down

translated = translate(img,100,100)
cv.imshow('translated',translated)


#rotacije 
def rotate(img,angle,rotPoint=None):
    (height,width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width//2,height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width,height)

    return cv.warpAffine(img,rotMat,dimensions)

rotated = rotate(img,45)
cv.imshow('rotated',rotated)

rotaded_rotaded = rotate(rotated,-45)
cv.imshow('rotaded_rotaded',rotaded_rotaded)

#resizing
resized = cv.resize(img,(500,500),interpolation=cv.INTER_CUBIC)
cv.imshow('resized',resized)

#flip
flip = cv.flip(img,0) #0,1,-1 vertical,horizontal,oboje
cv.imshow('flip',flip)

cv.waitKey(0)