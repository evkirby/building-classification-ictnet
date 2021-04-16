import numpy as np
from tensorflow import keras
from keras import layers
from keras import models
import tensorflow as tf

# Tenserflow
# def squeeze_excitation_layer(inputs, output_dimension, ratio, layer_name):
#     with tf.name_scope(layer_name):
#         squeeze = tf.reduce_mean(inputs, axis=[1, 2])

#         units = output_dimension/ratio
#         excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=units, layer_name=layer_name+'_fully_connected_1')
#         excitation = tf.nn.relu(excitation)
#         excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=output_dimension, layer_name=layer_name+'_fully_connected_2')
#         excitation = tf.nn.sigmoid(excitation)

#         excitation = reshape(excitation, [-1, 1, 1, output_dimension])
#         return inputs * excitation

# Keras
def squeeze_excitation_layer(inputs, output_dimension, ratio):
    squeeze = tf.reduce_mean(inputs, axis=[1, 2])

    units = output_dimension/ratio    
    excitation = layers.Dense(units, activation='relu')(squeeze)
    excitation = layers.Dense(output_dimension, activation='sigmoid')(excitation)
    excitation = tf.reshape(excitation, [-1, 1, 1, output_dimension])
    return inputs * excitation

