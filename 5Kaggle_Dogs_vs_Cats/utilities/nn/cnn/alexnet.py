# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K


class AlexNet:
    
    @staticmethod
    def gpu_grow_memory():
        import tensorflow as tf
        from distutils.version import LooseVersion
        import warnings
        from tensorflow.compat.v1.keras.backend import set_session
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

    @staticmethod
    def build(width, height, depth, classes, regularization=0.0002):
         # increasing the gpu memory
        AlexNet.gpu_grow_memory()
        # initialize the model along with the input shape to be
        # 'channels last' and the channels dimensions itself
        model = Sequential()
        inputShape = (height, width, depth)
        channelDimension = -1

        # if we are using 'channels first', update the input shape
        # and channels dimensions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            channelDimension = 1

        # Block # 1: first CONV => RELU => POOL layer set
        model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), 
                            input_shape=inputShape, 
                            padding="same", 
                            kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block # 2 : second CONV => RELU => POOL layer set
        model.add(Conv2D(filters=256, kernel_size=(5, 5), 
                            padding="same", 
                            kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block # 3 : CONV => RELU => CONV => RELU => CONV => RELU
        model.add(Conv2D(filters=384, kernel_size=(3, 3), 
                            padding="same", 
                            kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(Conv2D(filters=384, kernel_size=(3, 3), 
                            padding="same", 
                            kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(Conv2D(filters=256, kernel_size=(3, 3), 
                            padding="same", 
                            kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channelDimension))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.25))

        # Block # 4 : first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

         # Block # 5 : second set of FC => RELU layers
        model.add(Dense(4096, kernel_regularizer=l2(regularization)))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, kernel_regularizer=l2(regularization)))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model

        