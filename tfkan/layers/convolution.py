import tensorflow as tf
import numpy as np

from keras.layers import Layer, Conv2D
from .base import LayerKAN
from .dense import DenseKAN

from typing import Tuple, List, Any, Union, Callable
from abc import ABC, abstractmethod


class ConvolutionKAN(Layer, LayerKAN):
    @abstractmethod
    def __init__(self, 
        rank: int,
        filters: int,
        kernel_size: Any,
        strides: Any,
        padding: str = 'VALID',
        use_bias: bool = True,
        kan_kwargs: dict = {},
        **kwargs):
        """
        Parameters
        ----------
        filters : int
            the number of filters (the number of output channels) of the convolutional layer
        kernel_size : Any
            the size of the convolutional kernel, can be `int` or tuple of `int` with length `rank`
        strides : Any
            the stride of the convolutional kernel, can be `int` or tuple of `int` with length `rank`
        padding : str, optional
            the padding method, by default '"VALID"', can be '"VALID"' or '"SAME"'
        use_bias : bool, optional
            whether to use bias in the convolutional layer, by default True
        kan_kwargs : dict, optional
            the keyword arguments for the KAN kernel, by default {}
        """
        super(ConvolutionKAN, self).__init__(**kwargs)
        # the rank of the convolutional layer, e.g. 1 for Conv1D, 2 for Conv2D, etc.
        self.rank = rank
        self.filters = filters

        # check kernel_size and strides
        if isinstance(kernel_size, (list, tuple)):
            try:
                assert len(kernel_size) == self.rank
            except AssertionError:
                raise ValueError(f"expected kernel_size to be of length {self.rank}, found {kernel_size} of length {len(kernel_size)}")
        if isinstance(strides, (list, tuple)):
            try:
                assert len(strides) == self.rank
            except AssertionError:
                raise ValueError(f"expected strides to be of length {self.rank}, found {strides} of length {len(strides)}")
        
        self.kernel_size = kernel_size
        self.strides = strides

        # check padding
        try:
            assert padding.upper() in ['VALID', 'SAME']
            self.padding = padding.upper()
        except AssertionError:
            raise ValueError(f"expected padding to be 'VALID' or 'SAME', found {padding}")
        
        self.use_bias = use_bias
        self.kan_kwargs = kan_kwargs

    def build(self, input_shape):
        # input_shape (batch_size, spatial_1, ..., in_channels)
        if isinstance(input_shape, int):
            in_channels = input_shape
        else:
            in_channels = input_shape[-1]

        self._in_channels = in_channels
        
        # process kernel_size and strides
        if isinstance(self.kernel_size, int):
            self.kernel_size = tuple([self.kernel_size] * self.rank)
        if isinstance(self.strides, int):
            self.strides = tuple([self.strides] * self.rank)
        
        # create dnese layer for convolution kernel
        self.kernel = DenseKAN(
            units=self.filters,
            dtype=self.dtype,
            use_bias=False,
            **self.kan_kwargs
        )
        self._in_size = int(np.prod(self.kernel_size) * in_channels)
        self.kernel.build(self._in_size)
        self.grid_size = self.kernel.grid_size

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
        output = self.kernel(inputs)

        # reshape the output into the original shape
        output = tf.reshape(output, output_shape)

        # add bias
        if self.use_bias:
            output += self.bias

        return output
    
    @abstractmethod
    def _check_and_reshape_inputs(self, inputs):
        raise NotImplementedError

    def update_grid_from_samples(self, 
        inputs: tf.Tensor, 
        **kwargs
    ):
        inputs, _ = self._check_and_reshape_inputs(inputs)
        self.kernel.update_grid_from_samples(inputs, **kwargs)
    
    def extend_grid_from_samples(self, 
        inputs: tf.Tensor, 
        extend_grid_size: int,
        **kwargs
    ):
        inputs, _ = self._check_and_reshape_inputs(inputs)
        self.kernel.extend_grid_from_samples(inputs, extend_grid_size, **kwargs)
        self.grid_size = self.kernel.grid_size

    def get_config(self):
        config = super(ConvolutionKAN, self).get_config()
        config.update({
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'kan_kwargs': self.kan_kwargs
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

class Conv2DKAN(ConvolutionKAN):
    """
    2D convolution layer using KAN kernel.

    Make sure that the channel axis is the last axis of the input, i.e. `data_format='channels_last'`.
    The Conv2DKAN layer expects input shape to be `(batch_size, spatial_1, spatial_2, channels)`.

    Examples:

    >>> # The inputs are 28x28 RGB images with `channels_last` and the batch_size is 4.
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(shape=input_shape)
    >>> y = Conv2DKAN(filters=8, kernel_size=3)(x)
    >>> print(y.shape)
    (4, 26, 26, 8)
    """
    def __init__(self, 
        filters: int,
        kernel_size: Any,
        strides: Any=(1, 1),
        padding: str = 'VALID',
        use_bias: bool = True,
        kan_kwargs: dict = {},
        **kwargs):
        super(Conv2DKAN, self).__init__(rank=2, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, kan_kwargs=kan_kwargs, **kwargs)

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

  
class Conv3DKAN(ConvolutionKAN):
    """
    3D convolution layer using KAN kernel.

    Make sure that the channel axis is the last axis of the input, i.e. `data_format='channels_last'`.
    The Conv3DKAN layer expects input shape to be `(batch_size, spatial_1, spatial_2, spatial_3, channels)`.

    Examples:

    >>> # The inputs are 28x28 RGB video with `channels_last` and the batch_size is 4.
    >>> # The input num_steps of the video is 32 at the axis 1.
    >>> input_shape = (4, 32, 28, 28, 3)
    >>> x = tf.random.normal(shape=input_shape)
    >>> y = Conv3DKAN(filters=8, kernel_size=3, strides=(1, 2, 2), padding="same")(x)
    >>> print(y.shape)
    (4, 32, 14, 14, 8)
    """
    def __init__(self, 
        filters: int,
        kernel_size: Any,
        strides: Any=(1, 1, 1),
        padding: str = 'VALID',
        use_bias: bool = True,
        kan_kwargs: dict = {},
        **kwargs):
        super(Conv3DKAN, self).__init__(rank=3, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, kan_kwargs=kan_kwargs, **kwargs)
    
    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(shape)
        try:
            assert ndim == 5
        except AssertionError:
            raise ValueError(f"expected min_ndim=5, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self._in_channels
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self._in_channels}, found {shape[-1]}")
        
        # reshape the inputs into patches
        # so we can transform the convolution into a dense layer
        patches = tf.extract_volume_patches(
            inputs,
            ksizes=[1, *self.kernel_size, 1],
            strides=[1, *self.strides, 1],
            padding=self.padding
        )
        orig_shape = tf.shape(patches)[:-1]
        # reshape the patches into (-1, in_size)
        inputs = tf.reshape(patches, (-1, self._in_size))

        return inputs, orig_shape
    

class Conv1DKAN(ConvolutionKAN):
    """
    1D convolution layer using KAN kernel.

    Make sure that the channel axis is the last axis of the input, i.e. `data_format='channels_last'`.
    The Conv3DKAN layer expects input shape to be `(batch_size, spatial, channels)`.

    Examples:

    >>> # The inputs are time series data with `channels_last` and the batch_size is 4.
    >>> # The input num_steps of the video is 32 at the axis 1.
    >>> input_shape = (4, 32, 3)
    >>> x = tf.random.normal(shape=input_shape)
    >>> y = Conv1DKAN(filters=8, kernel_size=3, strides=2, padding="same")(x)
    >>> print(y.shape)
    (4, 16, 8)
    """
    def __init__(self, 
        filters: int,
        kernel_size: Any,
        strides: Any=1,
        padding: str = 'VALID',
        use_bias: bool = True,
        kan_kwargs: dict = {},
        **kwargs):
        super(Conv1DKAN, self).__init__(rank=1, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, kan_kwargs=kan_kwargs, **kwargs)

    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(shape)
        try:
            assert ndim == 3
        except AssertionError:
            raise ValueError(f"expected min_ndim=3, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self._in_channels
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self._in_channels}, found {shape[-1]}")
        
        # reshape the inputs into patches
        # so we can transform the convolution into a dense layer
        patches = tf.signal.frame(
            inputs,
            frame_length=self.kernel_size[0],
            frame_step=self.strides[0],
            pad_end=True if self.padding == 'SAME' else False,
            axis=1
        )
        orig_shape = tf.shape(patches)[:-2]
        # reshape the patches into (-1, in_size)
        inputs = tf.reshape(patches, (-1, self._in_size))

        return inputs, orig_shape