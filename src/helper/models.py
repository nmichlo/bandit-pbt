
#  Copyright (c) 2019 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.


import tensorflow as tf
import tensorflow.keras.layers as layers


# ========================================================================= #
# models                                                                   #
# ========================================================================= #


def create_mnist_model(data_format):
    """
    From https://github.com/tensorflow/models/blob/master/official/mnist/mnist.py
    :param data_format: 'channels_first' is typically faster on GPUs while 'channels_last' is typically faster on CPUs.
    :return: A tf.keras.Model
    """
    if data_format == 'channels_first':
        input_shape = [1, 28, 28]
    else:
        assert data_format == 'channels_last'
        input_shape = [28, 28, 1]

    max_pool = layers.MaxPooling2D((2, 2), (2, 2), padding='same', data_format=data_format)

    return tf.keras.Sequential([
        layers.Reshape(target_shape=input_shape, input_shape=(28 * 28,)),
        layers.Conv2D(32, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
        max_pool,
        layers.Conv2D(64, 5, padding='same', data_format=data_format, activation=tf.nn.relu),
        max_pool,
        layers.Flatten(),
        layers.Dense(1024, activation=tf.nn.relu),
        layers.Dropout(0.4),
        layers.Dense(10)
    ])




# ========================================================================= \#
# END                                                                       \#
# ========================================================================= \#
