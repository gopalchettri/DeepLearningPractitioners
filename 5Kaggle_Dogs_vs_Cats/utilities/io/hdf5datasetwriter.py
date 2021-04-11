# Is responsible for taking an input set of
# NumPy arrays (whether features, raw images, etc.) 
# and writing them to HDF5 format.

# import the necessary packages
import h5py
import os

class HDF5DatasetWriter:
    def __init__(self, dimensions, output_Path, data_Key="images", bufferSize=1000):
        # Check to see if the output path exists, and if so, raise an exception
        if os.path.exists(output_Path):
            raise ValueError("The supplied 'output_Path' already exits "
                            "and cannot be overwritten. "
                             "Manually delete the file before continuing.", output_Path)
        
        # Open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the 
        # class labels
        self.db = h5py.File(output_Path, "w")
        self.data = self.db.create_dataset(data_Key, dimensions, dtype="float")
        self.labels = self.db.create_dataset("labels",(dimensions[0], ), dtype="int")

        # Store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufferSize = bufferSize
        self.buffer = { "data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufferSize:
            self.flush()
    
    def flush(self):
        # Write the buffers to the disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx : i] = self.buffer["data"]
        self.labels[self.idx : i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, class_Labels):
        # create a dataset  to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str) # for Python 3 onwards, # vlen=unicode for python 2
        labelSet = self.db.create_dataset("label_names", (len(class_Labels), ), dtype=dt)
        labelSet[:] = class_Labels

    def close(self):
        # Check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()