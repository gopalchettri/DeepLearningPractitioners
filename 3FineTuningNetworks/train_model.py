# I've used a pre-trained CNN to extract features from a handful of datasets, let’s see
# how discriminative these features really are, especially given that the VGG16 was trained on
# ImageNet and not Animals, CALTECH-101, or Flowers-17.

# to run the code - python train_model.py --db ../2TransferLearning/dataset/animals/hdf5/features.hdf5 --model animals.cpickle
# to run the code - python train_model.py --db ../2TransferLearning/dataset/flowers17/hdf5/features.hdf5 --model flowers17.cpickle
# to run the code - python train_model.py --db ../2TransferLearning/dataset/caltech-101/hdf5/features.hdf5 --model caltech-101.cpickle

# import the necessary packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import argparse
import pickle
import h5py

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required=True, help="path HDF5 database")
ap.add_argument("-m", "--model", required=True, help="path to output model")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# Open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was 
# already shuffled *prior* to writing it to disk
# Given that our dataset is too large to fit into memory, we need an efficient method to determine
# our training and testing split. Since we know how many entries there are in the HDF5 dataset (and
# we know we want to use 75% of the data for training and 25% for evaluation), we can simply
# compute the 75% index i into the database. Any data before the index i is considered training data
# – anything after i is testing data.
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
print("[INFO] Tuning Hyper-parameters ...")
params = {"C": [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]}
model = GridSearchCV(LogisticRegression(max_iter=2000), params, cv=3, n_jobs=args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] Best Hyper-parameters : {}".format(model.best_params_))

# Evaluate the model
print("[INFO] Evaluation ...")
preds = model.predict(db["features"][i:])
print("[INFO] Classification - report")
print(classification_report(db["labels"][i:], 
                            preds, target_names=db["label_names"]))


# Serialize the model to disk
print("[INFO] Saving model ...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()