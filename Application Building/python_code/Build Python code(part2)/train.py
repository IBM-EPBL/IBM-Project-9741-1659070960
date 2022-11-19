# -*- coding: utf-8 -*-
"""Train the model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12XMaj85LWh0fOAvn72LixqMsnoviNQ8r

<a href="https://colab.research.google.com/github/IBM-EPBL/IBM-Project-33353-1660219050/blob/main/Model%20Building/Train%20the%20model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
"""



"""# Team ID : PNT2022TMID08626

# Importing the required library
"""

import tensorflow

import numpy#for numerical analysis
import tensorflow#open source ml tool by google
from tensorflow.keras.datasets import mnist #mnist dataset
from tensorflow.keras.models import Sequential# stack for layers
from tensorflow.keras import layers#input,middle and output layers forcnn structure
from tensorflow.keras.layers import Dense,Flatten#dense and flatten layers
from tensorflow.keras.layers import Conv2D#convolutional layers
from tensorflow import keras#library for building neural networks built on tensorflow
from tensorflow.keras.optimizers import Adam#optimizers
from keras.utils import np_utils

"""# Loading the data"""

(x_train,y_train),(x_test,y_test)=mnist.load_data()

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

"""# Analyzing the data"""

x_train[5]

y_train[5]

import matplotlib.pyplot as plt
plt.imshow(x_train[5])

"""# Reshaping the data"""

x_train=x_train.reshape(60000,28,28,1).astype('float32')
x_test=x_test.reshape(10000,28,28,1).astype('float32')

print ("Shape of X_train: {}".format(x_train.shape))
print ("Shape of y_train: {}".format(y_train.shape))
print ("Shape of X_test: {}".format(x_test.shape))
print ("Shape of y_test: {}".format(y_test.shape))

"""# Applying one Hotencoding

convert numerical values to classes where 0 to 9 are 10 seperate classes if value is 5 class 5 is 1 else 0
"""

no_of_classes=10
y_train=np_utils.to_categorical(y_train,no_of_classes)
y_test=np_utils.to_categorical(y_test,no_of_classes)

y_test[3]

from keras.layers import Dense, Flatten, MaxPooling2D, Dropout

"""# Adding CNN layer"""

model = Sequential()

model.add(Conv2D(64,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(Conv2D(32,(3,3),activation='relu'))

model.add(Flatten())
model.add(Dense(no_of_classes,activation='softmax'))

"""# Compile the model"""

model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

"""# Train the model"""

model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=5,batch_size=32)