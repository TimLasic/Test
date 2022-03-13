import cv2
import numpy as np

def nothing(x):
    pass

# Load image
image = cv2.imread('tim.png')

# Create a window
cv2.namedWindow('image')

# Create trackbars for color change
# Hue is from 0-179 for Opencv
cv2.createTrackbar('rMin', 'image', 0, 255, nothing)
cv2.createTrackbar('gMin', 'image', 0, 255, nothing)
cv2.createTrackbar('bMin', 'image', 0, 255, nothing)
cv2.createTrackbar('rMax', 'image', 0, 255, nothing)
cv2.createTrackbar('gMax', 'image', 0, 255, nothing)
cv2.createTrackbar('bMax', 'image', 0, 255, nothing)

# Set default value for Max HSV trackbars
cv2.setTrackbarPos('rMin', 'image', 0)
cv2.setTrackbarPos('gMin', 'image', 0)
cv2.setTrackbarPos('bMin', 'image', 0)
cv2.setTrackbarPos('rMax', 'image', 255)
cv2.setTrackbarPos('gMax', 'image', 255)
cv2.setTrackbarPos('bMax', 'image', 255)

# Initialize HSV min/max values
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

while(1):
    # Get current positions of all trackbars
    hMin = cv2.getTrackbarPos('rMin', 'image')
    sMin = cv2.getTrackbarPos('gMin', 'image')
    vMin = cv2.getTrackbarPos('bMin', 'image')
    hMax = cv2.getTrackbarPos('rMax', 'image')
    sMax = cv2.getTrackbarPos('gMax', 'image')
    vMax = cv2.getTrackbarPos('bMax', 'image')

    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(image, image, mask=mask)

    # Print if there is a change in HSV value
    if((phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

    # Display result image
    cv2.imshow('image', result)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()