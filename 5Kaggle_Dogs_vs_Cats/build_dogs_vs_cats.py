# scritp will create the HDF5 dataset from images

# import the necessary packages
from config import dogs_vs_cats_config as config
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import json
import cv2
import os

# grab the path to the images
trainPaths = list(paths.list_images(config.IMAGES_PATH))
print("trainPaths :", trainPaths)
trainLabels = [ path.split(os.path.sep)[1].split(".")[0] for path in trainPaths]
print("trainLabels :", trainLabels)
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)

# perfrom stratified sampling from the training set to build the 
# testing split from the training data
split = train_test_split(trainPaths, 
                            trainLabels, 
                            test_size=config.NUM_TEST_IMAGES, 
                            stratify=trainLabels, 
                            random_state=42)

(trainPaths, testPaths, trainLabels,testLabels) = split

# perform another stratified sampling this time to build the 
# validation data
split = train_test_split(trainPaths, 
                            trainLabels,
                            test_size = config.NUM_VALIDATION_IMAGES,
                            stratify = trainLabels,
                            random_state=42)

(trainPaths, validationPaths, trainLabels, validationLabels) = split

# construct a list pairing the training, validation and testing
# image paths along with their corresponding labels and output HDF5
# files
datasets = [
            ("train", trainPaths, trainLabels, config.TRAIN_HDF5),
            ("validation", validationPaths, validationLabels, config.VALIDATION_HDF5),
            ("test", testPaths, testLabels, config.TEST_HDF5)
            ]

# initialize the image preprocessor and the lists of RGB channel
# averages
aap = AspectAwarePreprocessor(256, 256)
(R, G, B) = ([], [], [])

# finally ready to build our HDF5 datasets

# loop over the dataset tuples
for(datasetType, paths, labels, outputPath) in datasets:
    # create the HDF5 writer
    print("[INFO] building {} ...".format(outputPath))
    writer = HDF5DatasetWriter((len(paths), 256, 256, 3), outputPath)

    # initialize the progress bar
    widgets = ["Building Dataset: ", progressbar.Percentage(), "", 
                progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(maxval=len(paths), widgets=widgets).start()

    # loop over the image path - to write each image in a given data split
    for(i, (path, label)) in enumerate(zip(paths, labels)):
        # load the image and process it
        image = cv2.imread(path)
        image = aap.preprocess(image)

        # if we are building the training dataset, then compute the 
        # mean of each channel in the image, then update the 
        # respective lists
        if datasetType == "train":
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
        
        # add the image and label # to the HDF5 dataset
        writer.add([image], [label])
        pbar.update(i)
    
    # close the HDF5 writer
    pbar.finish()
    writer.close()

# construct a dictionary of averages, then serialize the means to a
# JSON file
print("[INFO] serializing mean ...")
D = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
f = open(config.DATASET_MEAN, "w")
f.write(json.dumps(D))
f.close()