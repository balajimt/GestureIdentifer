import cv2
#-----------------------------TEXT OPTIONS--------------------------------------
font = cv2.FONT_HERSHEY_DUPLEX
size = 0.5
textX = 10
textY = 355
textH = 18
#---------------------------REGION OF INTEREST----------------------------------
# Coordinates of region of Interest
xROI = 350
yROI = 200

# Height and width f region of Interest
heightROI = 200
widthROI = 200

#---------------------------CAMERA OPTIONS--------------------------------------
# Boolean variables to keep track
saveImageFile = False
guessGesture = False
lastGesture = -1

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = True
counter = 0
# This parameter controls number of image samples to be taken PER gesture

gestureName = ""
path = ""
mod = 0

# Per Gesture the number of images
numOfSamples = 1000
