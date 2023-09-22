# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries.

### STEP 2:
Download and load the dataset

### STEP 3:
Scale the dataset between it's min and max values

### STEP 4:
Using one hot encode, encode the categorical values

### STEP-5:
Split the data into train and test

### STEP-6:
Build the convolutional neural network model

### STEP-7:
Train the model with the training data

### STEP-8:
Plot the performance plot

### STEP-9:
Evaluate the model with the testing data

### STEP-10:
Fit the model and predict the single input

## PROGRAM
~~~
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(xtrain,ytrain),(xtest,ytest)=mnist.load_data()
xtrain.shape

single_image=xtrain[10]
single_image.shape
plt.imshow(single_image,cmap="gray")
ytrain[10]

xtrain_scaled=xtrain/255.0
xtest_scaled=xtest/255.0
xtrain_scaled
xtrain_scaled.max()

ytrain_onehot=utils.to_categorical(ytrain,10)
ytest_onehot=utils.to_categorical(ytest,10)
ytrain_onehot
ytrain_onehot.shape
ytrain_onehot[10]

xtrain_scaled = xtrain_scaled.reshape(-1,28,28,1)
xtest_scaled = xtest_scaled.reshape(-1,28,28,1)
xtrain_scaled

model=keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(xtrain_scaled,ytrain_onehot,epochs=10,batch_size=64,validation_data=(xtest_scaled,ytest_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(xtest_scaled), axis=1)
print(confusion_matrix(ytest,x_test_predictions))
print(classification_report(ytest,x_test_predictions))

img = image.load_img('test_img.jpeg')
type(img)
img = image.load_img('test_img.jpeg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')
img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0
x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)
print(x_single_prediction)
~~~
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![](https://github.com/RanjithD18/mnist-classification/blob/main/1.png)
![](https://github.com/RanjithD18/mnist-classification/blob/main/2.png)
### Classification Report

![](https://github.com/RanjithD18/mnist-classification/blob/main/3.png)

### Confusion Matrix

![](https://github.com/RanjithD18/mnist-classification/blob/main/4.png)

### New Sample Data Prediction

![](https://github.com/RanjithD18/mnist-classification/blob/main/5.png)

## RESULT
Therefore a model has been successfully created for digit classification using mnist dataset.
