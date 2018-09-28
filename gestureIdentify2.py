# VM Specific Interface
import gestureNetwork as cnn
import gestureConfiguration as gc

def MainInterface():
    while True:
        keyPressed = int(input('1.Train Model (Images should be under ./newtest)'))
        if keyPressed == 1:
            gc.mod = cnn.buildNetwork(-1,2)
            cnn.trainModel(gc.mod)
            input("Press any key to continue")
            break
        else:
            continue

if __name__ == "__main__":
    MainInterface()
