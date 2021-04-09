import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from squeeze_excitation_block import squeeze_excitation_layer

def preactivation(inputs, num_filters, filter_shape=[3, 3], dropout_rate=0.2):
    preactivation = tf.nn.relu(slim.batch_norm(inputs))
    convolution = slim.conv2d(preactivation, num_filters, filter_shape)
    if dropout_rate != 0.0:
        convolution = slim.dropout(convolution, keep_prob=(1.0-dropout_p))
    return convolution

def DenseBlock(stack, num_layers, growth_rate, dropout_rate, path_type, output_dimension, scope=None):
    with tf.name_scope(scope) as scope:
        new_features = []
        for i in range(num_layers):
            current_layer = preactivation(stack, growth_rate, dropout_rate=dropout_rate)
            new_features.append(current_layer)
            stack = tf.concat([stack, current_layer], akis=-1)
        new_features = tf.concat(new_features, axis=-1)

        if path_type == "down":
            stack = squeeze_excitation_layer(new_features, output_dimension, 1, (scope + path_type))
        else:
            new_features = squeeze_excitation_layer(new_features, output_dimension, 1, (scope + path_type))
        
        return stack, new_features

