
import cv2 as cv



img = cv.imread('pepe.jpg') #BGR
cv.imshow('pepe',img)

#CONVERTING TO GRAYSCALE
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow('pepega',gray)


#blur
blur = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT) #PRAŠTECILA(7,7)
#cv.imshow('BLUR',blur)


#eDGE CASCADE pokaže kje so stene
canny = cv.Canny(blur,125,175) #blur zmanjša število risov
cv.imshow('canny',canny)

#dilating the image
dilated = cv.dilate(canny,(7,7), iterations=3)
cv.imshow('dilated',dilated)

#eroding
eroded = cv.erode(dilated,(3,3), iterations=1)
cv.imshow('eroded',eroded)

#resized
resized = cv.resize(img,(500,500))
cv.imshow('resized',resized)

#croping
cropped = img[50:200, 200:400] #height #width
cv.imshow('cropped',cropped)

cv.waitKey(0)