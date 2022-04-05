   
import numpy as np
import cv2 as cv
from kmeans import KMean


img  = cv.imread('pepe.jpg')
cv.imshow("staro",img)
img_data = (img / 255.).reshape(img.shape[0]*img.shape[1], 3) #našel probal sem -1  posplošit
img_data.shape

kmeans = KMean(k=9, max_iter=100)
closest = kmeans.zagon(img_data)
k_colors = kmeans.centrs[closest] # array ki ga je treba pretvorit
k_img = np.reshape(k_colors, (img.shape)) 
cv.imshow("novo",k_img)
cv.waitKey(0)

#prihaja do anpak npr. k= 16 max_iter = 300
'''invalid value encountered in true_divide
  ret = um.true_divide('''