
import numpy as np
import cv2 as cv
class KMean:
    def __init__(self,k,max_iter=300):

        self.k = k
        self.centroids = None
        self.max_iter = max_iter
        
    def initialize_centroids(self, points, k):
        """returns k centroids from the initial points"""
        centroids = points.copy()
        np.random.shuffle(centroids)
        return centroids[:k]
    
    def closest_centroid(self, points, centroids):
        """returns an array containing the index to the nearest centroid for each point"""
        #distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
        distances = np.absolute((abs(points - centroids[:, np.newaxis])).sum(axis=2))

        
        return np.argmin(distances, axis=0)
    def move_centroids(self, points, closest, centroids):
        """returns the new centroids assigned from the points closest to them"""
        return np.array([points[closest==k].mean(axis=0) for k in range(centroids.shape[0])])
    
    def fit(self, points):
        self.centroids = self.initialize_centroids(points, self.k)
        centroids = None
        for i in range(self.max_iter):
            closest = self.predict(points)
            self.centroids = self.move_centroids(points, closest, self.centroids)
        return closest 
    
    def predict(self, points):
        closest = self.closest_centroid(points, self.centroids)
        return closest 

if __name__=='__main__':  
    img  = cv.imread('pepe.jpg')
    cv.imshow("staro",img)
    img_data = (img / 255.).reshape(-1, 3)
    img_data.shape
    kmeans = KMean(k=16, max_iter=300)
    closest = kmeans.fit(img_data)
    k_colors = kmeans.centroids[closest]
    k_img = np.reshape(k_colors, (img.shape)) 
    cv.imshow("novo",k_img)

    # https://towardsdatascience.com/introduction-to-k-means-clustering-implementation-and-image-compression-8c59c439d1b dobil vzgled
