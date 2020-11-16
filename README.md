# Recognizing Facial Expressions with Convolutional Neural Network
Beizheng Chang, Jiachen Yan, Yuting Tang

This project aims to classify emotions from facial expressions by using CNN (Convolutional Neural Network).

### Summary
The research question that the project aims to address is whether building a machine learning model to classify emotions from facial expression images based on Convolutional Neural Network (CNN) would achieve the higher accuracy rates than a model based on Neural Network (NN).

In order to detect and classify emotional features from facial expressions, a three-phase methodology is adopted from transforming the image inputs to identifying and classifying the emotions. As gray-scale images with only front faces are fed into the model as inputs, the first stage is to decompose images into small features in the first layer of Convolutional Neural Network (CNN). Then, max pooling is adopted to group the small features to medium and large ones in the second and third layers of CNN separately. In the final phase which is the third layer of CNN, the large features are grouped and classified into the eight categories mentioned above. To evaluate the performance, the whole data set with image labels is divided into training, validation, and testing sets without any overlap in each experiment. Afterwards, based on the testing set, an accuracy rate is calculated as a performance measure by comparing the classifications produced by the model with the existing labels of the images.


### Files
- **project_main.ipynb**: the main Jupyter notebook that consists of the Python code to construct the CNN and NN model, compare the CNN models with different values of hyperparameters with the NN model, and evaluate the models by computing and plotting the training, validtaion, and testing accuracy;
- **CK+48-dataset.zip**: The Extended Cohn-Kanade (CK+) data set, pulled from the website https://www.kaggle.com/shawon10/ckplus, is adopted to train our algorithm. The CK+ data set is released in 2010 to enable the automatic detection of facial expressions. It is composed of 981 front-face images with each image being a 48x48 matrix of pixels. In the data set, each image is classified into one of the seven universal emotion categories: "anger", "disgust", "sadness", "fear", "contempt", and "happiness";
- **plot.png**: An example output which plots the training accuracy (blue) and the validation accuracy (orange) of the given CNN model.


### How to Excecute Files:
- Run the Jupyter notebook "project_main.ipynb"; 
- Before running the Jupyter notebook, unzip "CK+48-dataset.zip", and edit the variable "file_dir" to be the directory where the unzipped dataset exists;


### Required Library:
- import numpy as np
- import tensorflow as tf
- from tensorflow import keras
- import scipy as sp
- import glob
- import os
- import random
- import cv2
- from matplotlib import pyplot as plt
- import matplotlib.image as mpimg
- import pandas as pd
- import pickle
- from keras.layers.normalization import BatchNormalization


### Example Output:
- One of the outputs is the summary of the CNN model. For example:
````
Model: "sequential_11"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_19 (Conv2D)           (None, 46, 46, 32)        320       
_________________________________________________________________
conv2d_20 (Conv2D)           (None, 44, 44, 32)        9248      
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 22, 22, 32)        0         
_________________________________________________________________
conv2d_21 (Conv2D)           (None, 20, 20, 64)        18496     
_________________________________________________________________
conv2d_22 (Conv2D)           (None, 18, 18, 64)        36928     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 9, 9, 64)          0         
_________________________________________________________________
flatten_12 (Flatten)         (None, 5184)              0         
_________________________________________________________________
dense_29 (Dense)             (None, 128)               663680    
_________________________________________________________________
dense_30 (Dense)             (None, 7)                 903       
=================================================================
Total params: 729,575
Trainable params: 729,575
Non-trainable params: 0
_________________________________________________________________
````
- Another output is the plots of the training and validation accuracy of the CNN model. For example:

![Alt text](https://github.com/lauren-tang/emotion-recognition/blob/main/plot.jpg?raw=true "Example Plot")
