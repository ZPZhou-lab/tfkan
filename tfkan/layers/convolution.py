import tensorflow as tf
from keras.layers import Layer, Conv2D
from .dense import DenseKAN

from typing import Tuple, List, Any, Union, Callable

class Conv2DKAN(Layer):
    def __init__(
        self, 
        filters: int,
        kernel_size: Any,
        strides: Any,
        padding: str = 'VALID',
        use_bias: bool = True,
        kan_kwargs: dict = {},
        **kwargs):
        super(Conv2DKAN, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        try:
            assert padding.upper() in ['VALID', 'SAME']
            self.padding = padding.upper()
        except AssertionError:
            raise ValueError(f"expected padding to be 'VALID' or 'SAME', found {padding}")
        
        self.kan_kwargs = kan_kwargs
        self.use_bias = use_bias

    def build(self, input_shape):
        # input_shape (batch_size, dim1, dim2, ..., in_size)
        if isinstance(input_shape, int):
            in_channels = input_shape
        else:
            in_channels = input_shape[-1]

        self._in_channels = in_channels
        
        # process kernel_size and strides
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        if isinstance(self.strides, int):
            self.strides = (self.strides, self.strides)
        
        # create dnese layer for convolution kernel
        self.kernel = DenseKAN(
            units=self.filters,
            dtype=self.dtype,
            **self.kan_kwargs
        )
        self._in_size = self.kernel_size[0] * self.kernel_size[1] * in_channels
        self.kernel.build(self._in_size)

        # create bias if needed
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.filters,),
                initializer='zeros',
                trainable=True
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs, *args, **kwargs):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_channels)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.filters]], axis=0)

        # calculate the kernel output
        output = self.kernel(inputs) # shape (patch_size, filters)

        # reshape the output into the original shape
        output = tf.reshape(output, output_shape)
        # add bias
        if self.use_bias:
            output += self.bias

        return output

    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(shape)
        try:
            assert ndim == 4
        except AssertionError:
            raise ValueError(f"expected min_ndim=4, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self._in_channels
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self._in_channels}, found {shape[-1]}")
        
        # reshape the inputs into patches
        # so we can transform the convolution into a dense layer
        patches = tf.image.extract_patches(
            inputs,
            sizes=[1, *self.kernel_size, 1],
            strides=[1, *self.strides, 1],
            rates=[1, 1, 1, 1],
            padding=self.padding
        )
        orig_shape = tf.shape(patches)[:-1]
        # reshape the patches into (-1, in_size)
        inputs = tf.reshape(patches, (-1, self._in_size))

        return inputs, orig_shape