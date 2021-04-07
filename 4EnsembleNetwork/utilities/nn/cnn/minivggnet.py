# importing required libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.layers.pooling import MaxPooling1D, MaxPooling2D
from tensorflow.python.ops.gen_math_ops import mod

class MiniVGGNet:
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
    def build(width, height, depth, classes):
        
        # increasing the gpu memory
        MiniVGGNet.gpu_grow_memory()

        # Initialize the model, input shape and the channel dimension
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1

        # if we are using 'channel_first', update the input shape and channels dimension
        if K.image_data_format() == 'channel_first':
            input_shape = (depth, height, width)
            channel_dim = 1

        # First CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

         # Second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

         # First (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # Return the constructed network architecture
        return model