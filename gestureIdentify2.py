import gestureCNN as cnn
import gestureConfiguration as gc
import gestureFilters as gf
import cv2
import pyautogui
import numpy as np
import os
import time

# Saves an image with a particular gesture name
# Function loops till the number of samples and then
# writes them to the file directory

def saveROIImg(img):
    if gc.counter > (gc.numOfSamples - 1):
        # Reset the parameters
        gc.saveImageFile = False
        gc.gestureName = ''
        gc.counter = 0
        return

    gc.counter = gc.counter + 1
    name = gc.gestureName + str(gc.counter)
    print(("Saving img:",name))
    cv2.imwrite(gc.path + name + ".png", img)
    time.sleep(0.04)


def MainInterface():
    quietMode = False
    #Call CNN model loading callback
    while True:
        try:
            img = cv2.imread('mainScreen.png',1)
        except:
            print("Welcome screen not found")
        cv2.imshow('image',img)
        ans = cv2.waitKey(10) & 0xff
        if ans == ord('2'):
            gc.mod = cnn.buildNetwork(-1)
            cnn.trainModel(gc.mod)
            input("Press any key to continue")
            break
        elif ans == ord('1'):
            print("Will load default weight file")
            gc.mod = cnn.buildNetwork(0)
            break
        else:
            continue

    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    # set rt gc.size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)

    while(True):
        ret, frame = cap.read()
        max_area = 0

        frame = cv2.flip(frame, 3)

        if ret == True:
            if gc.binaryMode == True:
                roi = gf.binaryMask(frame)
            else:
                roi = gf.skinMask(frame)


        cv2.putText(frame,'GesReg v1.0',(gc.textX+520,gc.textY+6*gc.textH), gc.font, gc.size,(0,255,0),1,1)
        cv2.putText(frame,'b: Threshold     ESC: Freeze     p: Predict',(gc.textX,gc.textY + 5*gc.textH), gc.font, gc.size,(0,255,0),1,1)
        cv2.putText(frame,'a: Folder        s: Capture      h: EXIT',(gc.textX,gc.textY + 6*gc.textH), gc.font, gc.size,(0,255,0),1,1)

        if not quietMode:
            cv2.imshow('Camera',frame)
            cv2.imshow('ROI', roi)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xff

        ## Use Esc key to close the program
        if key == ord('h'):
            break

        ## Use b key to toggle between binary threshold or skinmask based filters
        elif key == ord('b'):
            gc.binaryMode = not gc.binaryMode
            if gc.binaryMode:
                print("Binary Threshold filter active")
            else:
                print("SkinMask filter active")

        ## Use g key to start gesture predictions via CNN
        elif key == ord('p'):
            gc.guessGesture = not gc.guessGesture
            print("Prediction Mode - {}".format(gc.guessGesture))

        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            gc.yROI = gc.yROI - 5
        elif key == ord('k'):
            gc.yROI = gc.yROI + 5
        elif key == ord('j'):
            gc.xROI = gc.xROI - 5
        elif key == ord('l'):
            gc.xROI = gc.xROI + 5

        ## Quiet mode to hide gesture window
        elif key == 27:
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))

        ## Use s key to start/pause/resume taking snapshots
        ## gc.numOfSamples controls number of snapshots to be taken PER gesture
        elif key == ord('s'):
            gc.saveImageFile = not gc.saveImageFile

            if gc.gestureName != '':
                gc.saveImageFile = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                gc.saveImageFile = False

        ## Use n key to enter gesture name
        elif key == ord('a'):
            gc.gestureName = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gc.gestureName)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Some issue while creating the directory named -' + gc.gestureName)

            gc.path = "./"+gc.gestureName+"/"
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MainInterface()
