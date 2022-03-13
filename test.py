import cv2 as cv
import numpy as np
from numba import jit
import random


def changeRes(width,height):
    # live camer
    capture.set(3,width) # 3 = width
    capture.set(4,height) # 4 = height


lower = np.array([255,219,172]) # ustvaril sem polje kjer "bi naj" bila spodnja mejfa kože
upper = np.array([141,85,36]) # ustvaril sem polje kjer "bi naj" bila zgornja mejfa kože

capture = cv.VideoCapture(0)
changeRes(320,240)

while True:
   isTrue, frame = capture.read()
   #frame_resized = rescaleFrame(frame)
   
   if isTrue == True:
        #CONVERTING TO rgb
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mask = cv.inRange(rgb,lower,upper) # preveri če obstaja kkašen array med tema dvema vrednostima
        #result = cv.bitwise_and(frame,frame,mask=mask)

        countours, hierarchy = cv.findContours(mask,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE) #external -na kak način se bo shranjevalo v hierarchy simple -shrai samo pomembne točke

        if len(countours) != 0:
            for countor in countours:
                if cv.contourArea(countor) > 500: # preverimo če je več kot 500 pikslov
                    x, y, w, h = cv.boundingRect(countor) # začnemo risat objekt okoli našega countora 
                    cv.rectangle(frame,(x,y),(x + w, y +h),(0,0,255), 3)

        cv.imshow("frame",frame)
        cv.imshow("mask",mask)
        if cv.waitKey(20) & 0xFF==ord('q'):
            break
   else:
        break

capture.release()
cv.destroyAllWindows() 