
import cv2 as cv
import numpy as np
from numba import jit
import random

import time

# image reading showing 
#img = cv.imread('pepe.jpg')

# cv.imshow('PEPE', img)


#RESIZE METHOD
#def rescaleFrame(frame, scale=0.75):
#    # image, video, live camera
#    width = int(frame.shape[1] * scale)
#    height = int(frame.shape[0] *scale)
#    dimensions = (width,height)
#
#    return cv.resize(frame,dimensions, interpolation=cv.INTER_AREA)

def changeRes(width,height):
    # live camer
    capture.set(3,width) # 3 = width
    capture.set(4,height) # 4 = height

#MASK
lower = np.array([42, 47, 42]) # ustvaril sem polje kjer "bi naj" bila spodnja mejfa kože
upper = np.array([216, 188, 180]) # ustvaril sem polje kjer "bi naj" bila zgornja mejfa kože
count = 0

def mask():
    for i in range(0,256,320//64): #320//64
        for j in range(0,194,240//46): #240//46
            blank_copy = blank.copy()
            maskk = cv.rectangle(blank_copy,(0+i,0+j),(64+i,46+j), 255, -1)
            masked_imagee = cv.bitwise_and(rgb,rgb,mask=maskk)
            mask = cv.inRange(masked_imagee,lower,upper)
            #checkPxInMask(masked_imagee)
            cv.imshow('masked_image', masked_imagee)
            cv.imshow('mask', mask)
            #count = count + 1
            cv.waitKey(1)
            if  0xFF==ord('q'):
                break

@jit(nopython=True)
def checkPxInMask(masked_image):
    for i in range(0,64,1):
        for j in range(0,46,1):
            check(masked_image[i,j])
            if  0xFF==ord('q'):
                break

def check(px):
    R, G, B = px
    return ( ((R > 95) and (G > 40) and (B > 20))
        and ((max(px)-min(px))>15) and (abs(R - G) > 15) and
        (R > G) and (R > B))


'''@jit(nopython=True)'''

#RESIZING IMAGES
#resized_image = rescaleFrame(img, scale=.2)
#cv.imshow('Image', resized_image)

#VIDEO CAPTURE 0 ,1 ...
capture = cv.VideoCapture(0)
changeRes(320,240)


#blank in mask
blank = np.zeros((240,320),dtype='uint8') # uint8 datatype of an image [:2] prvi 2 vrednosti

#widt / 46 - step
#height /64 -step

#definition for color
lower = np.array([197, 140, 133])
upper = np.array([236, 188, 180])

while True:
   
   isTrue, frame = capture.read()

   #frame_resized = rescaleFrame(frame)
   if isTrue == True:

        #CONVERTING TO rgb
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # masked image
        mask()

        
        #cv.imshow('Video', rgb)
        #cv.imshow('Video resized', frame_resized) # RESIZED VIDEO
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
   else:
        break

capture.release()
cv.destroyAllWindows() 
#videoCapture(capture)
# END OF VIDEO CAPTURE

''' The skin colour at uniform daylight illumination rule is defined as (R > 95 ) AND (G > 40 ) AND (B > 20 ) AND (max{R, G, B} - min{R, G, B} 15) AND (|R - G| > 15 ) AND (R > G) AND (R > B)
def check(px):
    R, G, B = px
    return ( ((R > 95) and (G > 40) and (B > 20))
        and ((max(px)-min(px))>15) and (abs(R - G) > 15) and
        (R > G) and (R > B))

def iterate_over_list(img):  # your method
    img = img.tolist()
    skinmask =  [[(1 if (check(px) or check2(px)) else 0) for px in row] for row in img]
    return skinmask

'''

'''img = rgb.tolist()
        skinmask =  [[(1 if (check(px)) else 0) for px in row] for row in img]
        mask=np.array(skinmask, dtype = "uint8")
        skin = cv.bitwise_and(frame, frame, mask = mask)''' 
cv.waitKey(0)
