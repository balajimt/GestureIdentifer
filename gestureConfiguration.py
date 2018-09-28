import cv2
#-----------------------------TEXT OPTIONS--------------------------------------
frameFont = cv2.FONT_HERSHEY_DUPLEX
frameFontSize = 0.5
textX = 10
textY = 355
textH = 18
#---------------------------REGION OF INTEREST----------------------------------
# Coordinates of region of Interest
xROI = 350
yROI = 200

# Height and width of region of Interest
heightROI = 200
widthROI = 200

#---------------------------CAMERA OPTIONS--------------------------------------
# Boolean variables to keep track
saveImageFile = False
guessGesture = False
lastGesture = -1
quietMode = False
binaryMode = True
counter = 0
gestureName = ""
path = ""
mod = 0

# Per Gesture the number of images
numOfSamples = 1000

#---------------------------CNN PARAMETERS----------------------------------
epochs = 14
batchSize = 50
outputClasses = 5

#------------------------IMAGE PARAMETERS-----------------------------------
# Input image dimensions
imgHeight = 200
imgWidth = 200
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
channels = 1

#----------------------WEIGHT FILE PARAMETERS-------------------------------
defaultPath = "./"
trainFolder = './newtest'
WeightFileName = ["newtestfile5000.hdf5","newtestweight50002.hdf5"]
output = ["Hi", "Stop","Spider", "Thumbsup", "Yo"]

#---------------------------------------------------------------------------
