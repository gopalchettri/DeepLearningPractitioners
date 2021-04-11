# to run - python train_alexnet.py
# import the necessary packages
# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from config import dogs_vs_cats_config as config
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import SimplePreprocessor
from utilities.preprocessing import PatchPreprocessor
from utilities.preprocessing import MeanPreprocessor
from utilities.callbacks import TrainingMonitor
from utilities.io import HDF5DatasetGenerator
from utilities.nn.cnn import AlexNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import json
import os

# construct the training image generator for data augmentation
data_augmentation = ImageDataGenerator(rotation_range=20, 
                                        zoom_range=0.15, 
                                        width_shift_range=0.2, 
                                        height_shift_range=0.2, 
                                        shear_range=0.15, 
                                        horizontal_flip=True, 
                                        fill_mode="nearest")

# load the RGB means for the training set
means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
iap = ImageToArrayPreprocessor()

# initialize the training and validation dataset generators
trainingGenerator = HDF5DatasetGenerator(dbPath = config.TRAIN_HDF5, 
                                    batchSize=128, 
                                    dataAugmentation = data_augmentation,
                                    preprocessors = [pp, mp, iap],
                                    classes=2)
validationGenerator = HDF5DatasetGenerator(dbPath = config.VALIDATION_HDF5,
                                        batchSize=128,
                                        preprocessors=[sp, mp, iap],
                                        classes=2)

# initialize the optimizer
print("[INFO] compiling the model...")
optimizer = Adam(lr=1e-3) # default 1e-3 = 0.001
model = AlexNet.build(width=227, height=227, depth=3, classes=2, regularization=0.0002)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# construct the set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(os.getpid())])
callbacks = [TrainingMonitor(path)]

# train the network
model.fit_generator(trainingGenerator.generator(), 
                    steps_per_epoch=trainingGenerator.numImages // 128,
                    validation_data = validationGenerator.generator(),
                    validation_steps = validationGenerator.numImages // 128,
                    epochs = 75,
                    max_queue_size = 128 * 2,
                    callbacks = callbacks, 
                    verbose = 1)

# save the model to file
print("[INFO] serializing model ...")
model.save(config.MODEL_PATH, overwrite=True)
print("[INFO] serializing model done.")

# close the HDF5 datasets
trainingGenerator.close()
validationGenerator.close()