import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

def squeeze_excitation_layer(inputs, output_dimension, ratio, layer_name):
    with tf.name_scope(layer_name):
        squeeze = tf.reduce_mean(inputs, axis=[1, 2])

        units = output_dimension/ratio
        excitation = tf.layers.dense(inputs=squeeze, use_bias=True, units=units, layer_name=layer_name+'_fully_connected_1')
        excitation = tf.nn.relu(excitation)
        excitation = tf.layers.dense(inputs=excitation, use_bias=True, units=output_dimension, layer_name=layer_name+'_fully_connected_2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = reshape(excitation, [-1, 1, 1, output_dimension])
        return inputs * excitation


