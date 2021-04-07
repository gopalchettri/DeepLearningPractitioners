# import the necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from keras import backend as K

class LeNet:

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
        LeNet.gpu_grow_memory()

        # Initialize the model
        model = Sequential()
        input_shape = (height, width, depth)

        # if we are using "channels first", update the input shape
        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        # first set of CONV => RELU => POOL layers
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of Fully Connected(FC) layer  => RELU
        # coverting the multi-dimenstional representation to 1D list using Flatten
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # applying softmax classifier
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        # return the constructed network architecture
        return model