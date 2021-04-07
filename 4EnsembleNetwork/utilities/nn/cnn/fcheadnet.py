# scripts used to create the fully connected head of the network

# impor the necessary packages
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

class FCHeadNet:

    @staticmethod
    def build(baseModel, classes, numberofNodes):
        # initialize the head model that will be placed on the top of
        # the base, then add a Fully Connected (FC) layer.
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(numberofNodes, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)

        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        # return the model
        return headModel