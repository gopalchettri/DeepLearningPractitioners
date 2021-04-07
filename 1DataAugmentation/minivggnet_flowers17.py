# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import optimizers
from utilities.preprocessing import ImageToArrayPreprocessor
from utilities.preprocessing import AspectAwarePreprocessor
from dataset import SimpleDatasetLoader
from utilities.nn.cnn import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
args = vars(ap.parse_args())

# Extracting the class labels from out input images
# Grabbing the list of images, then extracting
# the class label names from the image paths
image_Paths = list(paths.list_images(args["dataset"]))
class_Names = [path.split(os.path.sep)[-2] for path in image_Paths]
class_Names = [str(x) for x in np.unique(class_Names)]
print("INFO class_names :", class_Names)

# Initialize the image preprocessors
aap = AspectAwarePreprocessor(64, 64)
# to convert the image to keras-compatible arrays
iap = ImageToArrayPreprocessor()

# Load the dataset from disk then scale the raw pixel intensities
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(image_Paths, verbose=500)
data = data.astype("float") / 255.0
print("INFO Data:", data)

# Splitting the data into training and test set
(train_X, test_X, train_Y, test_Y) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Convert the labels from integers to vectors
train_Y = LabelBinarizer().fit_transform(train_Y)
test_Y = LabelBinarizer().fit_transform(test_Y)

# Initialize the Optimizer and model
print("[INFO} Compiling model...")
opt = SGD(lr=0.05)
model = MiniVGGNet.build(width=64, height=64, depth=3, classes=len(class_Names))
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train the network
print("[INFO] training network ...")
H = model.fit(train_X, train_Y, validation_data=(test_X, test_Y), batch_size=32, epochs=100, verbose=1)

# Evaluate the network
print("[INFO] Evaluating network ...")
predictions = model.predict(test_X, batch_size=32)
print(classification_report(test_Y.argmax(axis=1), predictions.argmax(axis=1), target_names=class_Names))

# Plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend()
# plt.show()