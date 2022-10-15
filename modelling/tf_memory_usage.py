# https://youtu.be/wPuSAPOMVj4
"""
Author: Dr. Sreenivas Bhattiprolu 

Calculation of memory required to store a batch of images and features 
in a deep learning model. In addition, we also add the memory required
to store trainable and non trainable parameters.

Remember that you need at least this much memory but in reality you have 
other overheads. 

If you do not have this much bare minimum memory then no point in trying to train
your model. Consider working with smaller images or batch sizes. 
"""

import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
import numpy as np

def get_model_memory_usage(model, batch_size=1):
    
    features_mem = 0 # Initialize memory for features. 
    float_bytes = 4.0 #Multiplication factor as all values we store would be float32.
    
    for layer in model.layers:
        
        out_shape = layer.output_shape
        
        if type(out_shape) is list:   #e.g. input layer which is a list
            out_shape = out_shape[0]
            
        #Multiply all shapes to get the total number per layer.    
        single_layer_mem = 1 
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        
        single_layer_mem_float = single_layer_mem * float_bytes #Multiply by 4 bytes (float)
        single_layer_mem_MB = single_layer_mem_float/(1024**2)  #Convert to MB
        
        print("Memory for", out_shape, " layer in MB is:", single_layer_mem_MB)
        features_mem += single_layer_mem_MB  #Add to total feature memory count
    
# Calculate Parameter memory
    trainable_wts = np.sum([K.count_params(p) for p in model.trainable_weights])
    non_trainable_wts = np.sum([K.count_params(p) for p in model.non_trainable_weights])
    parameter_mem_MB = ((trainable_wts + non_trainable_wts) * float_bytes)/(1024**2)
    print("_________________________________________")
    print("Memory for features in MB is:", features_mem*batch_size)
    print("Memory for parameters in MB is: %.2f" %parameter_mem_MB)
    
    total_memory_MB = (batch_size * features_mem) + parameter_mem_MB  #Same number of parameters. independent of batch size
    
    total_memory_GB = total_memory_MB/1024
    
    return total_memory_GB

def keras_model_memory_usage_in_bytes(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.
    
    The model shapes are multipled by the batch size, but the weights are not.
    
    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.
    
    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += keras_model_memory_usage_in_bytes(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem
    
    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )
    
    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return total_memory 

def print_memory_usage(model, batch=1):
    mem_for_my_model = get_model_memory_usage(model, batch)
    print("_________________________________________")
    print("Minimum memory required to work with this model is: %.2f" %mem_for_my_model, "GB")

def test_vgg16_memory_usage(model):
    #Let us import VGG16 model.
    input_image_shape = (3840,2160,3)
    model = VGG16(include_top=False, input_shape=input_image_shape)
    print(model.summary())
    print_memory_usage(model, 1)
