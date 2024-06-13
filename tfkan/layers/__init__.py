from .dense import DenseKAN
from .convolution import Conv1DKAN, Conv2DKAN, Conv3DKAN

import tensorflow as tf

custom_objects = {
    'DenseKAN': DenseKAN,
    'Conv1DKAN': Conv1DKAN,
    'Conv2DKAN': Conv2DKAN,
    'Conv3DKAN': Conv3DKAN
}
tf.keras.utils.get_custom_objects().update(custom_objects)

__all__ = ['DenseKAN', 'Conv1DKAN', 'Conv2DKAN', 'Conv3DKAN']