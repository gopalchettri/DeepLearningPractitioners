# Extract features from an arbitrary image dataset
# (provided the input dataset follows a specific directory structure)


# python extract_features.py --dataset ../2TransferLearning/dataset/flowers17/images --output ../2TransferLearning/dataset/flowers17/hdf5/features.hdf5
# python extract_features.py --dataset ../2TransferLearning/dataset/caltech-101/images --output ../2TransferLearning/dataset/caltech-101/hdf5/features.hdf5


# import required packages
import tensorflow as tf
from typing import no_type_check
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.preprocessing import LabelEncoder
from utilities.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import argparse
import random
import os

dataset_path = "../2TransferLearning/dataset/animals/images"
root_image_path = "../2TransferLearning/dataset/animals/images/"

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default=dataset_path, help="path to input dataset")
ap.add_argument("-o", "--output", required=True, help="path to output HDF5 file")
ap.add_argument("-b", "--batch_size", type=int, default=10,help="batch size of images to be passed through network")
ap.add_argument("-s", "--buffer_size", type=int, default=1000, help="size of feature extraction buffer")
args = vars(ap.parse_args())
# Store the file location in a convenience variable
dataset = args["dataset"]
# Store the batch size in a convenience variable
batch_size = args["batch_size"]

# grab the list of images that we'll be describing,
# then randomly shuffle them to allow for easy training and testing splits via
# array slicing during training time
# print("[INFO] loading imaegs ...")
# image_path = list(paths.list_images(args["dataset"]))
image_path = list(paths.list_images(dataset))
random.shuffle(image_path)

# print("Image Path : ", image_path)
# Extract the class labels from the image paths then encode the labels
labels = [p.split(os.path.sep)[-2] for p in image_path]
print('Labels : ', labels)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# load the VGG16 network
print("[INFO] loading network ...")
# we load the pre-trained VGG16 network from disk; however, the parameter include_top=False – supplying this value indicates that the final fullyconnected layers should not be included in the architecture
model = VGG16(weights="imagenet", include_top=False)

# initialize the HDF5 dataset writer, then store the class label names in the dataset
dataset = HDF5DatasetWriter(dimensions =(len(image_path), 512 * 7 * 7), 
                           output_Path = args["output"], 
                           data_Key="features", 
                           bufferSize=args["buffer_size"])
dataset.storeClassLabels(label_encoder.classes_)

# Feature Extraction
# Initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage()," ", progressbar.Bar(), " ", progressbar.ETA()]
pBar = progressbar.ProgressBar(maxval=len(image_path), widgets=widgets).start()

# loop over the images in patches
for i in np.arange(0, len(image_path), batch_size):
    # extract the batch of images and labels, then initialize the 
    # list of actual images that will be passed through the network
    # for feature extraction
    # print('value of i is : ', i)
    # print('value of batch_size is : ', batch_size)
    # print('value of image_path is : ', image_path)
    batch_Paths = image_path[i:i + batch_size]
    # print('value of batch_Paths is : ', batch_Paths)
    batch_Labels = labels[i:i + batch_size]
    # print('value of batch_Labels is : ', batch_Labels)
    batch_Images = []

    # Preparing an image for feature extraction.
    # Feature extraction is exactly the same as preparing an image for classification via a CNN

    # loop over the images and labels in the current batch
    for(j, imagePath) in enumerate(batch_Paths):
        # load the input image using the Keras helper utility
        # while ensuring the image is resized to 224x224 pixel
        # print('for loop - value of image_path is : ', imagePath)
        image = load_img(imagePath, target_size=(224, 224))
        # print('gopal')
        image = img_to_array(image)

        # preprocess the image by (1) expanding the dimensions and 
        # (2) subtracting the mean RGB pixel intensity from the 
        # ImageNet dataset
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        # add the image to the batch
        batch_Images.append(image)
    
    # Pass the images through the network and use the outputs as
    # our actual features
    batch_Images = np.vstack(batch_Images)
    features = model.predict(batch_Images, batch_size=batch_size)

    # Reshape the features so that each image is represented by 
    # a flattened feature vector of the 'MaxPooling2D' output
    features = features.reshape((features.shape[0], 512 * 7 * 7))

    # Add the features and labels to our HDF5 dataset
    dataset.add(features, batch_Labels)
    pBar.update(i)

# close the dataset
dataset.close()
pBar.finish()