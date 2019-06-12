import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras import layers
from keras import models



#get folders name convert it to dict
lookup = dict()
reverselookup = dict()
count = 0
input_dir='input/leapGestRecog/'
epcohes=10
batch_size=64



for j in os.listdir(input_dir+'00'):
    if not j.startswith('.'): # If running this code locally, this is to 
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1
print(lookup)

#reading data and converting it
x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    print(i+1,'/',10,' loading')
    for j in os.listdir(input_dir+'0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            # Loop over the images
            for k in os.listdir(input_dir+'0' + str(i) + '/' + j + '/'):
                # Read in and convert to greyscale
                img = Image.open(input_dir+'0' + str(i) + '/' + j + '/' + k).convert('L')
                img = img.resize((320, 120)) # resize to 320 width * 120 height pixels
                arr = np.array(img)
                x_data.append(arr) 
                count = count + 1
            y_values = np.full((count, 1), lookup[j]) 
            y_data.append(y_values)
            datacount = datacount + count
print('Data loaded successfully...')
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size
print('Data reshaped successfully...')

#reshape data to 1D grey scale image (0-255) and convert the image pixel value between 0 and 1
y_data = to_categorical(y_data)
x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

#split data to train, validate,test sets
x_train,x_further,y_train,y_further = train_test_split(x_data,y_data,test_size = 0.2)
x_validate,x_test,y_validate,y_test = train_test_split(x_further,y_further,test_size = 0.5)

#CNN model architcure
model=models.Sequential()
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320,1))) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print('Started Training Data...')
#train model
model.fit(x_train, y_train, epochs=epcohes, batch_size=batch_size, verbose=1, validation_data=(x_validate, y_validate))
#save trained model
model.save('model.h5')
[loss, acc] = model.evaluate(x_test,y_test,verbose=1)
print("Accuracy:" + str(acc))
print("loss:" +str(loss))