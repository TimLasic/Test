from doctest import DocTestRunner
from re import M
import cv2 as cv
import numpy as np

#def roberts(img,kernel,size):
#    img_copy = np.empty((250, 250), int)
#    zac = 0
#    konec = 3
#    for i in range(0,250,1): 
#        for j in range(0,250,1): 
#            for k in range(j,i+3,1):
#                for l in range(j,i+3,1):

def convolve(img,kernel):
    (iH,iW)=img.shape[:2]
    (kH,kW)=kernel.shape[:2]
    pad = (kW - 1) // 2
    if(kW == 2):
        pad = 1
    img = cv.copyMakeBorder(img, pad, pad, pad, pad,
		cv.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32") #float32
    for y in np.arange(pad,iH+pad): #med intervalom
        for x in np.arange(pad,iW+pad):
            roi = img[y-pad:y +pad +1, x -pad:x+pad+1] #extrectam mini sliko iz originalne slike : pomeni npr. 0-3 ali kaj druga  roi=region of interest
            k = (roi * kernel).sum()
            #print(k)
            output[y-pad, x -pad] = k
    
    #output = (255*(output - np.min(output))/np.ptp(output)).astype(int)
    #output =((output - output.min()) * (1/(output.max() - output.min()) * 255)).astype('uint8')
    output = np.absolute(output)
    output = np.uint8(255*output/np.max(output))
    #output = (output * 255).astype("uint8")
    #output =(((output + output.min()) / output.max() ) * 255).astype('uint8')
    return output

def convolveRob(img,kernel):
    (iH,iW)=img.shape[:2]
    (kH,kW)=kernel.shape[:2]
    pad = 1
    img = cv.copyMakeBorder(img, pad, pad, pad, pad,
		cv.BORDER_REPLICATE)
    output = np.zeros((iH, iW), dtype="float32") #float32
    for y in np.arange(pad,iH+pad): #med intervalom
        for x in np.arange(pad,iW+pad):
            roi = img[y-pad:y +pad , x -pad:x+pad] #extrectam mini sliko iz originalne slike : pomeni npr. 0-3 ali kaj druga  roi=region of interest
            k = (roi * kernel).sum()

            output[y-pad, x -pad] = k
    
    #output = (255*(output - np.min(output))/np.ptp(output)).astype(int)
    #output =((output - output.min()) * (1/(output.max() - output.min()) * 255)).astype('uint8')
    #output =(((output + output.min()) / output.max() ) * 255).astype('uint8')
    output = np.absolute(output)
    output = np.uint8(255*output/np.max(output))
    #output = (output * 255).astype("uint8")
    return output

def combine(gx,gy):
    (iH,iW)=gx.shape[:2]
    output = np.zeros((iH, iW), dtype="float32") #float32
    for i in range(0,iW,1): #height
        for j in range(0,iH,1): #width
            px = gx[j,i]
            py = gy[j,i]
            absol = int(abs(gx[j,i]))+int(abs(gy[j,i]))
            '''if(absol > 255):
                output[j,i]=255
            else:'''
            x = output[j,i]
            output[j,i]=(absol)
    #output =((output - output.min()) * (1/(output.max() - output.min()) * 255)).astype('uint8')
    #output =(((output + output.min()) / output.max() ) * 255).astype('uint8')
    output = np.absolute(output)
    output = np.uint8(255*output/np.max(output))
    #output = (output * 255).astype("uint8")     
    return output




img = cv.imread('tim-bright.png') #BGR

#value =100
#mat = np.ones(img.shape,dtype = 'uint8')*value
#brighter = cv.add(img,mat) #svetlo
#subtract = cv.subtract(img,mat) #temno

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray,(3,3),cv.BORDER_DEFAULT) #PRAŠTECILA(7,7)
#blur = cv.resize(blur,(100,100))

cv.imshow('pepe',blur)

#kernel roberts  \begin{equation*} G = \sqrt{G_x^2 * G_y^2} \end{equation*} ------------------------------------------------------------------------------------

_array1 = np.array((
	[1, 0],
	[0,  -1]
    ), dtype="int")

output1 = convolveRob(blur,_array1)
cv.imshow("original", gray)
cv.imshow(" roberts x convolve", output1)

_array2 = np.array((
	[-1, 0],
	[0,  1]
    ), dtype="int")

output2 = convolveRob(blur,_array2)

cv.imshow(" roberts y convolve", output2)
output = combine(output1,output2)
cv.imshow(" roberts combined convolve", output)

#canny
canny = cv.Canny(blur,125,175) #blur zmanjša število risov
cv.imshow('canny',canny)

#prewit skalrani produkt mase  v novi array |absolut gx lpus absolut gy -----------------------------------------------------------------------------------------------
_array3 = np.array((
	[1,0,-1],
	[1,0,-1],
    [1,0,-1]
    ), dtype="int")

output3 = convolve(blur,_array3)

cv.imshow(" prewit x convolve", output3)

_array4 = np.array((
	[1,1,1],
	[0,0,0],
    [-1,-1,-1]
    ), dtype="int")

output4 = convolve(blur,_array4)

cv.imshow(" prewit y convolve", output4)
output = combine(output3,output4)
cv.imshow(" prewit combined convolve", output)


#sobel---------------------------------------------------------------------------------------------------------------------------------------------


_array5 = np.array((
	[-1,0,1],
	[-2,0,2],
    [-1,0,1]
    ), dtype="int")
output5 = convolve(blur,_array5)

cv.imshow(" sobel x convolve", output5)

_array6 = np.array((
	[1,2,1],
	[0,0,0],
    [-1,-2,-1]
    ), dtype="int")
output6 = convolve(blur,_array6)

cv.imshow(" sobel y convolve", output6)

output = combine(output5,output6)
cv.imshow(" sobel combined convolve", output)
#imgSobel = cv.Sobel(blur, cv.CV_32F, 0,1, ksize=3)
#cv.imshow('real sobel',imgSobel)# TRESHOLD ZA NASLEDNO NALOGO
cv.waitKey(0)

#manhant
#seštevaš absolutne vrednosti razlik med slikami točk pikslov
# algoritme zapisan postopek k povprečje 68 zapisan
'''
odgovori na vprašanja:
1. Kje se pojavijo razlike pri detekciji robov nad temno in svetlo sliko in zakaj?
razlike se predvsem pojavijo pri številu zajetih robov in posledično jasnosti slike.
bolj kot osvetliš sliko(30%) bolj pridejo do izstopa robovi. težje je razločiti robove ko je slika temnejša / PREHODI so težje za zaznat
2.Zakaj je pred uporabo detektorja robov smiselno uporabiti filter za glajenje?
filter odstrani odvečni šum in posledično  prikaže robove bolj razločno/natačno.
3.Kaj nam pove gradient slike? Kako se uporablja pri detektorjih robov?
je sprememba intenzivnosti slike ali sprememba barve na sliki.
enačenje gradientov gc + gy nam bi naj pomagalo še odstranil odvečni šum in prikazati robove bolj natančno.
'''