import cv2
import pyautogui
import numpy as np
import os
import time
import gestureCNN as myNN

# Coordinates of region of Interest
x0 = 400
y0 = 200

# Height and width f region of Interest
height = 200
width = 200

# Boolean variables to keep track
saveImg = False
guessGesture = False
visualize = False
lastgesture = -1

# Which mask mode to use BinaryMask or SkinMask (True|False)
binaryMode = True
counter = 0
# This parameter controls number of image samples to be taken PER gesture

gestname = ""
path = ""
mod = 0

# Per Gesture the number of images
numOfSamples = 1000


# Saves an image with a particular gesture name
# Function loops till the number of samples and then
# writes them to the file directory

def saveROIImg(img):
    global counter, gestname, path, saveImg
    if counter > (numOfSamples - 1):
        # Reset the parameters
        saveImg = False
        gestname = ''
        counter = 0
        return

    counter = counter + 1
    name = gestname + str(counter)
    print(("Saving img:",name))
    cv2.imwrite(path + name + ".png", img)
    time.sleep(0.04)


def skinMask(frame, x0, y0, width, height ):
    global guessGesture, mod, lastgesture, saveImg
    # HSV values
    low_range = np.array([0, 50, 80])
    upper_range = np.array([30, 200, 255])

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]

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

    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True:
        retgesture = myNN.guessGesture(mod, res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            print(myNN.output[lastgesture])
            time.sleep(0.01 )
            #guessGesture = False
    return res


#%%
def binaryMask(frame, x0, y0, width, height ):
    global guessGesture, mod, lastgesture, saveImg

    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),2)

    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    #Uses Otsu's threshold value to find value
    minValue = 70
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #ret, res = cv2.threshold(blur, minValue, 255, cv2.THRESH_BINARY +cv2.THRESH_OTSU)

    if saveImg == True:
        saveROIImg(res)
    elif guessGesture == True:
        retgesture = myNN.guessAction(mod, res)
        if lastgesture != retgesture :
            lastgesture = retgesture
            # Trainded Gesture Files
            # output = ["Hi", "Stop","Spider", "Thumbsup", "Yo"]
            # if lastgesture == 3:
            #     print("Hide/View Controls")
            #     pyautogui.hotkey('ctrl', 'h')
            #     time.sleep(0.25)
            if lastgesture == 1:
                print("Play/Pause")
                pyautogui.press('space')
                time.sleep(0.25)
    return res


banner =  '''\n
    1- Use pretrained model for gesture recognition & layer visualization
    2- Train the model (you will require image samples for training under .\newtest)
    '''


def Main():
    global guessGesture, visualize, mod, binaryMode, x0, y0, width, height, saveImg, gestname, path
    quietMode = False

    font = cv2.FONT_HERSHEY_DUPLEX
    size = 0.5
    fx = 10
    fy = 355
    fh = 18

    #Call CNN model loading callback
    while True:
        try:
            ans = int(input( banner))
        except:
            print("Not an integer input")
        if ans == 2:
            mod = myNN.buildNetwork(-1)
            myNN.trainModel(mod)
            input("Press any key to continue")
            break
        elif ans == 1:
            print("Will load default weight file")
            mod = myNN.buildNetwork(0)
            break
        else:
            continue

    ## Grab camera input
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

    # set rt size as 640x480
    ret = cap.set(3,640)
    ret = cap.set(4,480)

    while(True):
        ret, frame = cap.read()
        max_area = 0

        frame = cv2.flip(frame, 3)

        if ret == True:
            if binaryMode == True:
                roi = binaryMask(frame, x0, y0, width, height)
            else:
                roi = skinMask(frame, x0, y0, width, height)


        cv2.putText(frame,'GesReg v1.0',(fx+520,fy+6*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'b: Threshold     ESC: Freeze     p: Predict',(fx,fy + 5*fh), font, size,(0,255,0),1,1)
        cv2.putText(frame,'a: Folder        s: Capture      h: EXIT',(fx,fy + 6*fh), font, size,(0,255,0),1,1)

        ## If enabled will stop updating the main openCV windows
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
            binaryMode = not binaryMode
            if binaryMode:
                print("Binary Threshold filter active")
            else:
                print("SkinMask filter active")

        ## Use g key to start gesture predictions via CNN
        elif key == ord('p'):
            guessGesture = not guessGesture
            print("Prediction Mode - {}".format(guessGesture))

        ## Use i,j,k,l to adjust ROI window
        elif key == ord('i'):
            y0 = y0 - 5
        elif key == ord('k'):
            y0 = y0 + 5
        elif key == ord('j'):
            x0 = x0 - 5
        elif key == ord('l'):
            x0 = x0 + 5

        ## Quiet mode to hide gesture window
        elif key == 27:
            quietMode = not quietMode
            print("Quiet Mode - {}".format(quietMode))

        ## Use s key to start/pause/resume taking snapshots
        ## numOfSamples controls number of snapshots to be taken PER gesture
        elif key == ord('s'):
            saveImg = not saveImg

            if gestname != '':
                saveImg = True
            else:
                print("Enter a gesture group name first, by pressing 'n'")
                saveImg = False

        ## Use n key to enter gesture name
        elif key == ord('a'):
            gestname = input("Enter the gesture folder name: ")
            try:
                os.makedirs(gestname)
            except OSError as e:
                # if directory already present
                if e.errno != 17:
                    print('Some issue while creating the directory named -' + gestname)

            path = "./"+gestname+"/"

        #elif key != 255:
        #    print key

    #Realse & destroy
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Main()
