import tensorflow as tf
# from tensorflow.keras import layers, models
from keras import backend as K

class ShallowNet:

    @staticmethod
    def gpu_grow_memory():
        from distutils.version import LooseVersion
        import warnings
        from tensorflow.compat.v1.keras.backend import set_session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

        if not tf.test.gpu_device_name(): 
            warnings.warn('no gpu found')
        else: 
            print('default gpu device: {}' .format(tf.test.gpu_device_name()))
    

    @staticmethod
    def build(width, height, depth, classes):

        # increasing the gpu memory
        ShallowNet.gpu_grow_memory()

        # Initialize the model along with the input shape to be 'channels_last'
        model = tf.keras.models.Sequential()
        input_shape = (height, width, depth)

        # Update the image shape if 'channels_first' is being used
        if K.image_data_format() == 'channels_first':
            input_shape  = (depth, height, width)

        # Define the first (and only) CONV => RELU layer
        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', input_shape=input_shape))
        # model.add(layers.Conv2D(32, (3,3), padding='same', input_shape=input_shape))
        model.add(tf.keras.layers.Activation('relu'))

        # Add a softmax classifier
        model.add(tf.keras.layers.Flatten()) # coverting the multi-dimenstional representation to 1D list
        model.add(tf.keras.layers.Dense(classes))
        model.add(tf.keras.layers.Activation('softmax'))

        # Return the network architecture
        return model

   