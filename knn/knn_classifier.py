import sys
import os
import cv2
import numpy as np
import argparse

#from sympy import arg
from _utils import DatasetLoad

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'path to input dataset')
ap.add_argument('-k', '--neighbors', type = int, default = 1, help = '# of nearest neighbor for classification')
args = vars(ap.parse_args())

pathes = args['dataset']
print(pathes)
k = args['neighbors']

# Resize
width = 64
height = 64

# Initial load data dataset
# This DatasetLoad initial, it already give size image, the default value is 64x64
# If we want to change size
# data = DatasetLoad(128, 128)
data = DatasetLoad(width, height)

# Load dataset from paths
print('[INFO] loading dataset ...')
# we know our label are cat, dog, and panda
label = ['cat', 'dog', 'panda']
# verbose = 500, it mean we want to show on screen only when it achive 500 images read
datas, labels  = data.load(pathes, verbose = 500)

# flatten image to feature vector
# from show screen, we see total image of cat is 1000, dog is 1000 and panda is 1000
# total images is 3000
print('[INFO] shape of data = ', datas.shape)
# This shape is (3000, 64,64,3)
# 3000 is total images
# 64,64 is size of image
# 3 is channel (RGB)

# We need to reshape image to a feature vector
# it means 3000 is still 3000
# but 64,64,3 need to be flatten to a vector =64*64*3
# datas.shape[0] is 3000
flat_image = datas.shape[1]*datas.shape[2]*datas.shape[3]
# use function reshape to flatten image
datas = datas.reshape((datas.shape[0], flat_image))
print('[INFO] new datas shape = ', datas.shape)

# show some information on memory consumption of the images
print('[INFO] features matrix: {:.1f}MB'.format(datas.nbytes/(1024*1000.0)))

# encode the labels as integers
le =LabelEncoder()
labels = le.fit_transform(labels) # we convert labels to integers number
# cats ---> 0
# dogs ---> 1
# pandas ---> 2
# We convert alphabet to number because we can not use string in computation
# Labels can be transform to integers number (0, 1, 2, ... so on) or transform to binary (001, 010, 100)

# partition the data into trianing and testing splits
# using 75% of the data for training and the remaining 25% for testing

# train_test_split is a function to generate training dataset and testing dataset
# in this case because of we have all the dataset in one folder
# we also can seperate by ourself
# create a folder of training dataset and testing dataset by our own
# then read those datas seperately
print('[INFO] split dataset to training and testing dataset ...')
(trainX, testX, trainY, testY) = train_test_split(datas, labels, test_size = 0.25)

# train and evaluate a k-NN classifier on the raw pixel intensities
print('[INFO] evaluating k-NN classifier ...')
# n_neighbors here is your k numbers (k = 1, focus only one nearest neighbor)
# k = 5, you focus on 5 nearest neighbor and look how many of them go to cats, dogs og pandas
# the new image will say it belong to cats if the number of cats is higher than other two
model = KNeighborsClassifier(n_neighbors=k) 
# Fit the model
model.fit(trainX, trainY)

print(classification_report(testY, model.predict(testX), target_names=le.classes_))

# ------------- Testing ------------- 
print('[INFO] Call some image to test ...')
path = 'data_1/data/Test/animals/dogs'
listfiles = os.listdir(path)
for (i, imagefile) in enumerate(listfiles):
    imagepath = path+'/'+imagefile
    image = cv2.imread(imagepath)
    img = image.copy()
    img = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    # reshape it to a feature vector 64*64*3, 3 id channel (RGB)
    img = img.reshape(1, width*height*3)
    pred = model.predict(img)
    cv2.putText(image, 'Label: {}'.format(label[pred[0]]),
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('Org', image)
    key = cv2.waitKey(1000)&0xFF
    if key == ord('q'):
        break