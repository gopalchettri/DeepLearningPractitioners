# import the necessary packages
from tensorflow.keras import utils
import numpy as np
import h5py

class HDF5DatasetGenerator:
    
    def __init__(self, dbPath, batchSize, preprocessors=None, 
                    dataAugmentation=None, binarize=True, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the  labels should be binarized, along with
        # the total number of classes

        # The size of mini-batches to yield when training our network

        self.batchSize = batchSize 
        self.preprocessors = preprocessors
        self.dataAugmentation = dataAugmentation
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath)
        self.numImages = self.db["labels"].shape[0]


    def generator(self, passes=np.inf):
        # np.inf -  floating point representation of (positive) infinity
        # initialize the epoch count
        epochs = 0

        # keep looping infinetly -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            for i in np.arange(0, self.numImages, self.batchSize):
                # extract the iamges and labels from the HDF dataset
                images = self.db["images"][i: i + self.batchSize]
                labels = self.db["labels"][i: i + self.batchSize]

                # check to see if the labels should be binarized
                if self.binarize:
                    labels = utils.to_categorical(labels, self.classes)

                    # check to see if our preporcessors are not None
                    if self.preprocessors is not None:
                        # initialize the list of processed images
                        processedImages = []

                        # loop over the images
                        for image in images:
                            # loop over the preprocessors and apply each
                            # to the image
                            for p in self.preprocessors:
                                image = p.preprocess(image)
                            
                            # update the list of processed images
                            processedImages.append(image)

                        # update the images array to be the processed
                        # images
                        images = np.array(processedImages)
            
                    # if the data augmentor exists, apply it
                    if self.dataAugmentation is not None:
                        (images, labels) = next(self.dataAugmentation.flow(images, 
                            labels, batch_size=self.batchSize))
                    # yield a tuple of images and labels
                    yield (images, labels)
                
                # increment the total number of epochs
                epochs += 1

    def close(self):
        # close the database
        self.db.close()