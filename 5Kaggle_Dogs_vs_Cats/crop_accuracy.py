# Evaluating AlexNet on testing set using both
# standard method and over-sampling technique
# to run - python crop_accuracy.py
# import the necessary packages
from config import dogs_vs_cats_config as config
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import MeanPreprocessor
from utilities.preprocessing import CropPreprossor
from utilities.io import HDF5DatasetGenerator
from utilities.utils.ranked import rank5_accuracy
from tensorflow.keras.models import load_model
import numpy as np
import progressbar
import json

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessor
sp = SimplePreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
cp = CropPreprossor(227, 227)
iap = ImageToArrayPreprocessor()

# load the pretrained network
print("[INFO] loading model ...")
model = load_model(config.MODEL_PATH)

# initialize the testing dataset generator, then make predictions on
# the testing data
print("[INFO] predicting on test data (no crops) ...")
testGenerator = HDF5DatasetGenerator(dbPath = config.TEST_HDF5, 
                                    batchSize=64, 
                                    preprocessors = [sp, mp, iap],
                                    classes=2)
predictions = model.predict_generator(testGenerator.generator(), steps=testGenerator.numImages // 64, max_queue_size=64*2)

# compute the rank-1 and rank-5 accuries
(rank1, _) = rank5_accuracy(predictions, testGenerator.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))
testGenerator.close()

# re-initialize the testing set generator, this time excluding the 
# 'SimplePreprocessor'
testGenerator = HDF5DatasetGenerator(dbPath = config.TEST_HDF5, 
                                    batchSize=64, 
                                    preprocessors = [mp],
                                    classes=2)
predictions = []

# initialize the progress bar
widgets = ["Evaluating : ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=testGenerator.numImages // 64,
                                widgets=widgets).start()

# loop over a single pass of the test data
for (i, (images, labels)) in enumerate(testGenerator.generator(passes=1)):
    # loop over each of the individual images
    for image in images:
        # apply the crop preprocessor to the image to generate 10
        # separate crops, then convert them from images to arrays
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops], dtype="flaot32")

        # make predictions on the crops and then average them
        # together to obtain the final prediction
        pred = model.predict(crops)
        predictions.append(pred.mean(axis=0))

    # update the progress bar
    pbar.update(i)

# compute the rank-1 accuracy
pbar.finish()
print("[INFO] predicting on test data (with crops)...")
(rank1, _) = rank5_accuracy(predictions, testGenerator.db["labels"])
print("[INFO] rank-1: {:.2f}%".format(rank1 * 100))