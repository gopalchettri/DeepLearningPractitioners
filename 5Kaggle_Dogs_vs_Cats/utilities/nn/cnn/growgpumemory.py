import tensorflow as tf

class GrowGPUMemory:

    # @staticmethod
    def gpu_grow_memory():
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
            
