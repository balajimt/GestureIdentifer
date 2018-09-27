import gestureCNN as cnn
import gestureConfiguration as gc
import gestureFilters2 as gf
import cv2
#import pyautogui
import numpy as np
import os
import time

def MainInterface():
    while True:
        # try:
        #     img = cv2.imread('mainScreen.png',1)
        # except:
        #     print("Welcome screen not found")
        # cv2.imshow('image',img)
        # keyPressed = cv2.waitKey(10) & 0xff
        # if keyPressed == ord('2'):
        #     gc.mod = cnn.buildNetwork(-1)
        #     cnn.trainModel(gc.mod)
        #     input("Press any key to continue")
        #     break
        # elif keyPressed == ord('1'):
        #     print("Will load default weight file")
        #     gc.mod = cnn.buildNetwork(0)
        #     break
        # else:
        #     continue


        keyPressed = int(input('1.Predict\n2.Train'))
        if keyPressed == 2:
            gc.mod = cnn.buildNetwork(-1)
            cnn.trainModel(gc.mod)
            input("Press any key to continue")
            break
        elif keyPressed == 1:
            print("Will load default weight file")
            gc.mod = cnn.buildNetwork(0)
            break
        else:
            continue

    # Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    # Screen Dimensions
    ret = cap.set(3,640)
    ret = cap.set(4,480)

    while(True):
        ret, frame = cap.read()
        frame = cv2.flip(frame, 2)

        # Functions to be displayed on screen
        cv2.putText(frame,'GesReg v1.0',(gc.textX+520,gc.textY+6*gc.textH), gc.frameFont, gc.frameFontSize,(0,255,0),1,1)
        cv2.putText(frame,'b: Threshold     ESC: Freeze     p: Predict',(gc.textX,gc.textY + 5*gc.textH), gc.frameFont, gc.frameFontSize,(0,255,0),1,1)
        cv2.putText(frame,'a: Folder        s: Capture      h: EXIT',(gc.textX,gc.textY + 6*gc.textH), gc.frameFont, gc.frameFontSize,(0,255,0),1,1)

        if ret == True:
            if gc.binaryMode == True:
                roi = gf.thresholdFilter(frame)
            else:
                roi = gf.skinmaskFilter(frame)

        if not gc.quietMode:
            cv2.imshow('Camera',frame)
            cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        # H (Halt)
        if key == ord('h') or key == ord('H'):
            break

        # B (Threshold Toggle)
        elif key == ord('b') or key == ord('B'):
            gc.binaryMode = not gc.binaryMode
            if gc.binaryMode:
                print("Threshold filter active")
            else:
                print("SkinMask filter active")

        # P (Predict Key)
        elif key == ord('p') or key == ord('P'):
            gc.guessGesture = not gc.guessGesture
            print("Prediction - {}".format(gc.guessGesture))

        # Adjusting ROI window
        # Can be later extended to real time ROI
        elif key == ord('i') or key == ord('I'):
            gc.yROI = gc.yROI - 2
        elif key == ord('k') or key == ord('K'):
            gc.yROI = gc.yROI + 2
        elif key == ord('j') or key == ord('J'):
            gc.xROI = gc.xROI - 2
        elif key == ord('l') or key == ord('L'):
            gc.xROI = gc.xROI + 2

        # ESC (Freeze Key)
        elif key == 27:
            gc.quietMode = not gc.quietMode
            print("Freeze - {}".format(gc.quietMode))

        ## Use s key to start/pause/resume taking snapshots
        ## gc.numOfSamples has the number of snapshots to be taken
        elif key == ord('s') or key == ord('S'):
            gc.saveImageFile = not gc.saveImageFile

            if gc.gestureName != '':
                gc.saveImageFile = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                gc.saveImageFile = False

        ## Use A key to enter gesture folder name
        elif key == ord('a') or key == ord('A'):
            gc.gestureName = input("Folder name: ")
            try:
                os.makedirs(gc.gestureName)
            except:
                print('Directory Error -' + gc.gestureName)
            gc.path = "./"+gc.gestureName+"/"
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MainInterface()
