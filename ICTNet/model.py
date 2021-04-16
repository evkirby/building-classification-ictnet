# -- Imports -- #
from keras.layers import *
import keras
import numpy as np
import os

from squeeze_excitation_block import squeeze_excitation_layer
# ------------- #

# -- Variables -- #
IMG_SIZE = 512
N_CHANNELS = 1
N_CLASSES = 2 # binary
eps=1e-5
compress_factor = 0.5
# --------------- #

# -- Functions -- #
def pre_activation_conv(inputs, n_filters, filter_size=(3, 3), dropout_rate=0.2):
    x = BatchNormalization(epsilon=eps)(inputs)
    x = ReLU()(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(n_filters, kernel_size=filter_size, use_bias=False, kernel_initializer='he_normal')(x)
    x = Dropout(rate=dropout_rate)(x)
    return x

def dense_block(inputs, num_layers, growth_rate, dropout_rate, path_type, out_dim):
    new_features = []
    for i in range(num_layers): # num_layers is the value of 'l'
        conv_outputs = pre_activation_conv(inputs, growth_rate, dropout_rate=dropout_rate)
        inputs = Concatenate()([inputs, conv_outputs])
        new_features.append(conv_outputs)
        # n_filters += growth_rate # To increase the number of filters for each layer. (Added)
    new_features = Concatenate()(new_features)

    # Special block for SE
    if path_type == "down":
        inputs = squeeze_excitation_layer(inputs, out_dim, 1)
    else:
        new_features = squeeze_excitation_layer(new_features, out_dim, 1)

    return inputs, new_features


def TransitionDown(inputs, n_filters, dropout_rate):
    # compression_factor is the 'θ'
    # x = BatchNormalization(epsilon=eps)(inputs)
    # x = Activation('relu')(x)
    # num_feature_maps = inputs.shape[1] # The value of 'm'

    # x = Conv2D(np.floor(compression_factor * num_feature_maps).astype(np.int),
    #            kernel_size=(1, 1),
    #            use_bias=False,
    #            padding='same',
    #            kernel_initializer='he_normal',
    #            kernel_regularizer=keras.regularizers.l2(1e-4)
    #            )(x)
    # x = Dropout(rate=dropout_rate)(x)
    x = pre_activation_conv(inputs, n_filters, (1, 1), dropout_rate)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)  # Should be max pooling ?
    return x


def TransitionUp(inputs, n_filters, skip_connection):
    # skip_connection = ZeroPadding2D((1, 1))(skip_connection)
    x = Conv2DTranspose(n_filters, kernel_size=(3, 3), strides=(2, 2))(inputs)
    x = Concatenate()([x, skip_connection])
    return x


def build_model(preset_model='FC-DenseNet56', num_classes=2, n_filters_first_conv=48, n_pool=5, growth_rate=12, n_layers_per_block=4, dropout_rate=0.2):
    if preset_model == 'FC-DenseNet56':
        n_pool = 5
        growth_rate = 12
        n_layers_per_block = 4
    elif preset_model == 'FC-DenseNet67':
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = 5
    elif preset_model == 'FC-DenseNet103':
        n_pool = 5
        growth_rate = 16
        n_layers_per_block = [4, 5, 7, 10, 12, 15, 12, 10, 7, 5, 4]

    if type(n_layers_per_block) == int:
        n_layers_per_block = [n_layers_per_block] * (2 * n_pool + 1)

    input_shape = (IMG_SIZE, IMG_SIZE, N_CHANNELS)

    # Downsampling path
    inputs = Input(shape=input_shape)
    x = Conv2D(n_filters_first_conv, kernel_size=(3, 3), use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)

    n_filters = n_filters_first_conv

    skip_connection_list = []

    for i in range(n_pool):
        n_filters += growth_rate * n_layers_per_block[i]
        # Dense Block
        x, _ = dense_block(x, n_layers_per_block[i], growth_rate, dropout_rate, "down", n_filters)

        # At the end of the dense block, the current stack is stored in the skip_connections list
        skip_connection_list.append(x)
        # Transition Down
        x = TransitionDown(x, n_filters, dropout_rate)

    skip_connection_list = skip_connection_list[::-1]  # Reverse

    # Bottleneck Dense Block
    out_dim = n_layers_per_block[n_pool] * growth_rate
    # We will only upsample the new feature maps
    x, block_to_upsample = dense_block(x, n_layers_per_block[n_pool], growth_rate, dropout_rate, "bottlneck", out_dim)

    # Upsampling path
    for i in range(n_pool):
        # Transition Up ( Upsampling + concatenation with the skip connection)
        n_filters_keep = growth_rate * n_layers_per_block[n_pool + i]
        x = TransitionUp(block_to_upsample, n_filters_keep, skip_connection_list[i])

        # Dense Block
        out_dim = n_layers_per_block[n_pool + i + 1] * growth_rate
        # We will only upsample the new feature maps
        x, block_to_upsample = dense_block(x, n_layers_per_block[n_pool + 1 + i], growth_rate, dropout_rate, "up", out_dim)

    # Softmax
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(N_CLASSES)(x)  # Num Classes
    # outputs = Activation('softmax')(x)
    net = Conv2D(num_classes, kernel_size=(1, 1))(x)
    prediction = np.argmax(net, axis=3).astype(int)
    probas = Softmax()(prediction)

    model = keras.models.Model(inputs, probas)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(learning_rate=0.0001, decay=0.995), #keras.optimizers.SGD(lr=0.001),
                  metrics=['acc'])
    model.summary()
# --------------- #

# A utiliser pour le fit(x, y, callbacks=[keras.callbacks.ModelCheckpoint(filepath, ...)
build_model()