# python finetune_flowers17.py --dataset ../Vision/dataset/flowers17/images --model ../3FineTuningNetworks/flowers17.model

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from utilities.datasets import SimpleDatasetLoader
from utilities.nn.cnn import FCHeadNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from imutils import paths
import numpy as np
import argparse
import os



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to the input dataset")
ap.add_argument("-m", "--model", required=True, help="path to the output model")
args = vars(ap.parse_args())

def gpu_grow_memory():
    import tensorflow as tf
    from tensorflow.compat.v1.keras.backend import set_session
    from distutils.version import LooseVersion
    import warnings
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if not tf.test.gpu_device_name(): 
        warnings.warn('No GPU found')
    else: 
        print('Default GPU device: {}' .format(tf.test.gpu_device_name()))
            
gpu_grow_memory()

# Construct the image Generator for data augmentation
data_augmentation =  ImageDataGenerator(rotation_range=30,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip =True,
                            fill_mode="nearest")

# Grab the list of images that we'll be describing, then extract
# the class label names from the image paths
print("[INFO] loading images ...")
imagePaths = list(paths.list_images(args["dataset"]))
print("imagePaths :", imagePaths)
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classNames = [str(x) for x in np.unique(classNames)]

# initialize the image preprocessors
aap = AspectAwarePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities 
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# Partition the data into training and testing splits using 75% of the
# data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Network Surgery
# Load the VGG16 network, ensuring the head Fully Connected (FC) layer
# sets are left off
baseModel = VGG16(weights="imagenet", include_top=False,
                   input_tensor=Input(shape = (224, 224, 3)))

# Initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, len(classNames), 256)

# Place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs = baseModel.input, outputs = headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model ...")
optimizer = RMSprop(lr = 0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer,
                    metrics=["accuracy"])

# train the head of the network for a few epochs (all other layers are frozen)
# this will allow the new FC layers to start to become intialized with actual
# "learned" values versus pure random
print("[INFO] training head ...")
model.fit_generator(data_augmentation.flow(trainX, trainY, batch_size=32),
                    validation_data=(testX, testY), epochs=25,
                    steps_per_epoch=len(trainX) //32, verbose=1)

# Evaluate the network after initialization
print("[INFO] evaluating after initialization ...")
predictions = model.predict(testX, batch_size=32)
print("[INFO] Classification report ...")
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=classNames))

# Now that the head FC layer have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
    layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] Re-compiling model ...")
optimizer = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of the CONV layers along with our set of FC layers
print("[INFO] fine-tuning model ...")
model.fit_generator(data_augmentation.flow(trainX, trainY, batch_size=32), 
                    steps_per_epoch=len(trainX)// 32,
                    epochs=100, verbose=1, 
                    validation_data=(testX, testY))


# Evaluation the network on the fine-tuned model
print("[INFO] Evaluating after fine-tuning ...")
predictions = model.predict(testX, batch_size=32)
print("[INFO] Fine-tuned Classification report ...")
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), 
                            target_names=classNames))

# Save the model to disk
print("[INFO] Serializing model ...")
model.save(args["model"])