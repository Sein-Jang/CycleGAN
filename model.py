import tensorflow as tf
from tensorflow.python.keras.layers import Add, Conv2D, Input, LeakyReLU, ReLU, Conv2DTranspose
from tensorflow.python.keras.models import Model
from tensorflow_addons.layers import InstanceNormalization


def res_block(x_in, num_filters):
    x = tf.pad(x_in, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    x = Conv2D(num_filters, kernel_size=3, padding='valid', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    x = Conv2D(num_filters, kernel_size=3, padding='valid', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = Add()([x_in, x])

    return x


def generator(num_filters=64, num_res_blocks=9, num_downsamplings=2):
    x_in = Input(shape=(None, None, 3))

    x = tf.pad(x_in, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    x = Conv2D(num_filters, kernel_size=7, padding='valid', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    for _ in range(num_downsamplings):
        num_filters *= 2
        x = Conv2D(num_filters, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)

    for _ in range(num_res_blocks):
        x = res_block(x, num_filters)

    for _ in range(num_downsamplings):
        num_filters //= 2
        x = Conv2DTranspose(num_filters, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = ReLU()(x)

    x = tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], mode='REFLECT')
    x = Conv2D(3, kernel_size=7, padding='valid')(x)
    x = tf.tanh(x)

    return Model(x_in, x)


def discriminator(num_filters=64, num_downsamplings=3):
    num_filters_ = num_filters
    x_in = Input(shape=(None, None, 3))

    x = Conv2D(num_filters, kernel_size=4, padding='same')(x_in)
    x = LeakyReLU(alpha=0.2)(x)

    for _ in range(num_downsamplings - 1):
        num_filters = min(num_filters * 2, num_filters_ * 8)
        x = Conv2D(num_filters, kernel_size=4, strides=2, padding='same', use_bias=False)(x)
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

    num_filters = min(num_filters * 2, num_filters_ * 8)
    x = Conv2D(num_filters, kernel_size=4, strides=1, padding='same', use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(1, kernel_size=4, strides=1, padding='same')(x)

    return Model(x_in, x)
