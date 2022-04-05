
import numpy as np
import cv2 as cv
class KMean:
    def __init__(self,k,max_iter=100):

        self.k = k
        self.centrs = None
        self.max_iter = max_iter
        
    def inicializirajCentre(self, img, k):
       
        centrs = img.copy()
        np.random.shuffle(centrs)# vrne random centre
        return centrs[:k] # vrne k centrov
    
    def najblizji(self, img, centrs):
        """returns an array containing the index to the nearest centroid for each point"""
        #distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = np.absolute((np.abs(img - centrs[:, np.newaxis])).sum(axis=2))  # vrne array distanc 
        #np.newaxis pretvorba dimenzij oz. dodajnje dimenzije 2 in računanje po 2 dimenziji (x,y,vrednost) računamo vrednost
                                                                                            #0 #1 #2
        
        return np.argmin(distances, axis=0) # vrne indekse najkraših razdalj
    def premakni(self, img, closest, centrs):
       
        return np.array([img[closest==k].mean(axis=0) for k in range(centrs.shape[0])]) # vrne novo destinacijo  
        # preveri vse indexe razdalj   pri vseh centrih
    
    def zagon(self, img):
        self.centrs = self.inicializirajCentre(img, self.k)
       
        for i in range(self.max_iter):
            closest = self.pridobi(img)
            self.centrs = self.premakni(img, closest, self.centrs)
        return closest  # vrne indexe najbližjega 
    
    def pridobi(self, img):
        closest = self.najblizji(img, self.centrs) # vrne indexe najbližjega 
        return closest 


