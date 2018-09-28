import gestureConfiguration as gc
import gestureNetwork as cnn
import numpy as np
import cv2
import pyautogui
import os
import time

def skinmaskFilter(frame):
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])
    # rgb(80, 244, 66) Green
    cv2.putText(frame,'Skin Mask',(gc.xROI,gc.yROI-20), gc.frameFont, gc.frameFontSize,(80,244,66),1,1)
    cv2.rectangle(frame, (gc.xROI,gc.yROI),(gc.xROI+gc.widthROI,gc.yROI+gc.heightROI),(80,244,66),1)
    roi = frame[gc.yROI:gc.yROI+gc.heightROI, gc.xROI:gc.xROI+gc.widthROI]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #Apply skin color range
    mask = cv2.inRange(hsv, low_range, upper_range)
    skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    mask = cv2.erode(mask, skinkernel, iterations = 1)
    mask = cv2.dilate(mask, skinkernel, iterations = 1)

    #blur
    mask = cv2.GaussianBlur(mask, (15,15), 1)
    #cv2.imshow("Blur", mask)

    #bitwise and mask original frame
    res = cv2.bitwise_and(roi, roi, mask = mask)
    # color to grayscale
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    if gc.saveImageFile == True:
        extractImage(res)
    elif gc.guessGesture == True:
        returnGesture = cnn.guessGesture(gc.mod, res)
        if gc.lastGesture != returnGesture :
            gc.lastGesture = returnGesture
            print(cnn.output[gc.lastGesture])
            time.sleep(0.01 )
            #guessGesture = False
    return res


#%%
def thresholdFilter(frame):
    # Color orange
    cv2.putText(frame,'Binary Threshold',(gc.xROI,gc.yROI-20), gc.frameFont, gc.frameFontSize,(80,244,66),1,1)
    cv2.rectangle(frame, (gc.xROI,gc.yROI),(gc.xROI+gc.widthROI,gc.yROI+gc.heightROI),(80,244,66),1)
    roi = frame[gc.yROI:gc.yROI+gc.heightROI, gc.xROI:gc.xROI+gc.widthROI]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    #Uses Otsu's threshold value to find value
    minValue = 70
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)

    if gc.saveImageFile == True:
        extractImage(res)
    elif gc.guessGesture == True:
        returnGesture = cnn.guessGesture(gc.mod, res)
        if gc.lastGesture != returnGesture :
            gc.lastGesture = returnGesture
            # Trainded Gesture Files
            # output = ["Hi", "Stop","Spider", "Thumbsup", "Yo"]
            # Hi gesture to invoke space
            if gc.lastGesture == 1:
                print("Play/Pause")
                #pyautogui.press('space')
                time.sleep(0.25)
    return res

def extractImage(img):
    if gc.counter > (gc.numOfSamples - 1):
        gc.saveImageFile = False
        gc.gestureName = ''
        gc.counter = 0
        return
    gc.counter = gc.counter + 1
    name = gc.gestureName + str(gc.counter)
    print(("Image Number:",name))
    cv2.imwrite(gc.path + name + ".png", img)
    time.sleep(0.04)
