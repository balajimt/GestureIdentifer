
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import os
from PIL import Image

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.utils import plot_model

K.set_image_dim_ordering('th')

# Batch_size to train
batch_size = 50

# No of output classes
nb_classes = 5

# No of epochs
nb_epoch = 14

# Total number of convolutional filters to use
nb_filters = 32
# Max pooling
nb_pool = 2


#  data
path = "./"
path1 = "./gestures"    #path of folder of images

## Path2 is the folder which is fed in to training model
path2 = './newtest'

WeightFileName = ["newtestfile5000.hdf5"]

# Predicted Outputs
output = ["Hi", "Stop","Spider", "Thumbsup", "Yo"]


#%%
def returnDirectoryList(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist


# Input image dimensions
img_rows = 200
img_cols = 200
# For grayscale use 1 value and for color images use 3 (R,G,B channels)
img_channels = 1

'''
CNN Model Description
https://towardsdatascience.com/a-simple-2d-cnn-for-mnist-digit-recognition-a998dbc1e79a
http://cs231n.github.io/convolutional-networks/
'''
# Load CNN model
model = Sequential()
def buildNetwork(wf_index,index):
    global get_output
    # Convolution 2D layer addition
    # (no_of filters, rows in covolution kernel, columns in convolution kernel)
    if(index==1):
        # 15 layer Model
        model.add(Conv2D(32, (3,3),activation='relu',input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        # MaxPooling2D is a way to reduce the number of parameters in our model
        # by sliding a 2x2 pooling filter across the previous layer and taking
        # the max of the 4 values in the 2x2 filter.
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        # Output layer should have the layer corresponding to the class labels
        # For number classification it would have been 10
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
    elif(index==2):
        model.add(Conv2D(32, (3, 3),padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
    # Compilation based on a loss and entropy function
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    # Model summary
    print("Model Summary")
    model.summary()
    # Model conig details
    model.get_config()
    plot_model(model, to_file='new_model.png', show_shapes = True)

    if wf_index >= 0:
        #Load pretrained weights
        fname = WeightFileName[int(wf_index)]
        print("loading ", fname)
        model.load_weights(fname)

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    return model

# This function does the guessing work based on input images
def guessGesture(model, img):
    global output, get_output
    # Flatten it to single dimensional array
    # to reshape it later
    image = np.array(img).flatten()

    # Analyses only one image at a time
    # Reshape it to (depth, width, height)
    image = image.reshape(img_channels, img_rows,img_cols)

    # The final preprocessing step for the input data is to convert our data
    # type to float32 and normalize our data values to the range [0, 1].
    image = image.astype('float32')
    image = image / 255

    # Reshape it to (number of images, depth, width, height)
    rimage = image.reshape(1, img_channels, img_rows, img_cols)

    # Now feed it to the NN, to fetch the predictions
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)

    # get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    prob_array = get_output([rimage, 0])[0]

    #print prob_array

    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    # Get the output with maximum probability
    import operator

    guess = max(iter(d.items()), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 90.0:
        #print(d)
        return output.index(guess)

    else:
        return 1

# Splits up into test and train data
def initialiseImages():
    # Returns all files in a directory
    imlist = returnDirectoryList(path2)

    # Open one image to get size
    image1 = np.array(Image.open(path2 +'/' + imlist[0]))

    m,n = image1.shape[0:2] # get the size of the images
    total_images = len(imlist) # get the 'total' number of images

    # Create a matrix to store all flattened images
    # When translating a color image to black and white (mode “L”),
    # the library uses the ITU-R 601-2 luma transform:
    # L = R * 299/1000 + G * 587/1000 + B * 114/1000

    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in sorted(imlist)], dtype = 'f')

    print(immatrix.shape)
    input("Press any key")

    ## Label the set of images per respective gesture type.
    ## Initialize the label
    label=np.ones((total_images,),dtype = int)

    samples_per_class = total_images / nb_classes
    print("samples_per_class - ",samples_per_class)
    s = 0
    r = int(samples_per_class)
    for classIndex in range(nb_classes):
        label[s:r] = classIndex
        s = int(r)
        r = int(s + samples_per_class)

    '''
    # eg: For 301 img samples/gesture for 4 gesture types
    label[0:301]=0
    label[301:602]=1
    label[602:903]=2
    label[903:]=3
    '''

    # Generate random sequences of the matrix array
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]

    (X, y) = (train_data[0],train_data[1])


    # Split X and y into training and testing sets

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Nomrmalize entire image channel
    X_train /= 255
    X_test /= 255

    # convert class vectors to binary class matrices
    # one of the 5 values
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test


def trainModel(model):
    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initialiseImages()

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    ans = input("Saving trained weights")
    filename = input("Enter file name")
    fname = path + str(filename) + ".hdf5"
    model.save_weights(fname,overwrite=True)
