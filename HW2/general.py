import tensorflow as tf


#Check tensorflow version and GPU available
def Check_version():
    print(tf.__version__)
    print(tf.test.is_gpu_available())
    
    
Check_version()
