import tensorflow as tf
from keras.layers import Layer
from ..ops.spline import calc_spline_values

from typing import Tuple, List, Any, Union, Callable


class DenseKAN(Layer):
    def __init__(
        self,
        units: int,
        grid_size: int=5,
        spline_order: int=3,
        grid_range: Union[Tuple[float], List[float]]=(-1.0, 1.0),
        spline_initialize_stddev: float=0.1,
        basis_activation: Union[str, Callable]='silu',
        dtype=tf.float32,
        **kwargs
    ):
        super(DenseKAN, self).__init__(dtype=dtype, **kwargs)
        self.units = units
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        self.basis_activation = basis_activation

        # initialize parameters
        self.spline_initialize_stddev = spline_initialize_stddev

    def build(self, input_shape: Any):
        # input_shape (batch_size, dim1, dim2, ..., in_size)
        if isinstance(input_shape, int):
            in_size = input_shape
        else:
            in_size = input_shape[-1]

        self.in_size = in_size
        self.spline_basis_size = self.grid_size + self.spline_order
        bound = self.grid_range[1] -self.grid_range[0]

        # build grid
        self.grid = tf.linspace(
            self.grid_range[0] - self.spline_order * bound / self.grid_size,
            self.grid_range[1] + self.spline_order * bound / self.grid_size,
            self.grid_size + 2 * self.spline_order + 1
        )
        # expand the grid to (in_size, -1)
        self.grid = tf.repeat(self.grid[None, :], in_size, axis=0)
        self.grid = tf.cast(self.grid, dtype=self.dtype)

        # the linear weights of the spline activation
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.RandomNormal(stddev=self.spline_initialize_stddev),
            trainable=True,
            dtype=self.dtype
        )

        # build scaler weights C
        self.scale_factor = self.add_weight(
            name="scale_factor",
            shape=(self.in_size, self.units),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
            dtype=self.dtype
        )

        # build basis activation
        if isinstance(self.basis_activation, str):
            self.basis_activation = tf.keras.activations.get(self.basis_activation)
        elif not callable(self.basis_activation):
            raise ValueError(f"expected basis_activation to be str or callable, found {type(self.basis_activation)}")

        self.built = True

    
    def call(self, inputs, *args, **kwargs):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)

        # calculate the B-spline output
        spline_in = calc_spline_values(inputs, self.grid, self.spline_order) # (B, in_size, grid_basis_size)
        # matrix multiply: (batch, in_size, grid_basis_size) @ (in_size, grid_basis_size, out_size) -> (batch, in_size, out_size)
        spline_out = tf.einsum("bik,iko->bio", spline_in, self.spline_kernel)

        # calculate the basis b(x) with shape (batch_size, in_size)
        # add basis to the spline_out: phi(x) = c * (b(x) + spline(x)) using broadcasting
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)

        # scale the output
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        
        # aggregate the output using sum (on in_size dim) and reshape into the original shape
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)


        return spline_out
    
    def _check_and_reshape_inputs(self, inputs):
        ndim = inputs.ndim
        try:
            assert ndim >= 2
        except AssertionError:
            raise ValueError(f"expected min_ndim=2, found ndim={ndim}. Full shape received: {inputs.shape}")

        try:
            assert inputs.shape[-1] == self.in_size
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self.in_size}, found {inputs.shape[-1]}")
        
        # reshape the inputs to (-1, in_size)
        orig_shape = tf.shape(inputs)[:-1]
        inputs = tf.reshape(inputs, (-1, self.in_size))

        return inputs, orig_shape