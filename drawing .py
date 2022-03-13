import cv2 as cv
import numpy as np
import time

blank = np.zeros((500,500,3),dtype='uint8') # uint8 datatype of an image
cv.imshow('Blank',blank)

blank = np.zeros((240,320),dtype='uint8')

print(320//46)



# 1.Paint the image a certain colour
#blank[:] = 0,255,0 # nastaviš vse barve
#blank[200:300,300:400] = 0,255,0 # nastaviš vse barve
#cv.imshow('Green',blank)


# 2. draw a rectangle
#cv.rectangle(blank,(0,0),(250,500),(0,255,0),thickness=2) #0,0 = origin 250 = širina 500 =  višina thickness=cv.FILLED je enako kot -1
#cv.rectangle(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2),(0,255,0),thickness=2) #

#cv.imshow('Rectangle',blank)

# 3 circle
#cv.circle(blank, (250,250),40,(0,0,255),thickness=3) #(250,250) center
#cv.imshow('Circle',blank)

# draw a line
#cv.line(blank,(100,250),(300,400),(255,255,255),thickness=3)
#cv.imshow('Line',blank)

#text
#cv.putText(blank, 'hello world',(0,225), cv.FONT_HERSHEY_TRIPLEX,1.0,(0,255,0),thickness=2)
#cv.imshow('Text',blank)
cv.waitKey(0)