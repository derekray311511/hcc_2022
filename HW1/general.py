import tensorflow as tf


#Check tensorflow version and GPU available
def Check_version():
    print(tf.__version__)
    # print(tf.test.is_gpu_available())
    print(tf.config.list_physical_devices('GPU'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print("\n",len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")

def solve_cudnn_error():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("\n",len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs\n")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    
    
Check_version()
