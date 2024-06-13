import tensorflow as tf
from keras.layers import Layer
from .base import LayerKAN
from ..ops.spline import fit_spline_coef
from ..ops.grid import build_adaptive_grid

from typing import Tuple, List, Any, Union, Callable


class DenseKAN(Layer, LayerKAN):
    def __init__(
        self,
        units: int,
        use_bias: bool=True,
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
        self.use_bias = use_bias

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
        self.grid = tf.Variable(
            initial_value=tf.cast(self.grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid"
        )

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

        # build bias
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.units,),
                initializer=tf.keras.initializers.Zeros(),
                trainable=True,
                dtype=self.dtype
            )
        else:
            self.bias = None

        self.built = True

    
    def call(self, inputs, *args, **kwargs):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs, orig_shape = self._check_and_reshape_inputs(inputs)
        output_shape = tf.concat([orig_shape, [self.units]], axis=0)

        # calculate the B-spline output
        spline_out = self.calc_spline_output(inputs)

        # calculate the basis b(x) with shape (batch_size, in_size)
        # add basis to the spline_out: phi(x) = c * (b(x) + spline(x)) using broadcasting
        spline_out += tf.expand_dims(self.basis_activation(inputs), axis=-1)

        # scale the output
        spline_out *= tf.expand_dims(self.scale_factor, axis=0)
        
        # aggregate the output using sum (on in_size dim) and reshape into the original shape
        spline_out = tf.reshape(tf.reduce_sum(spline_out, axis=-2), output_shape)

        # add bias
        if self.use_bias:
            spline_out += self.bias

        return spline_out
    
    def _check_and_reshape_inputs(self, inputs):
        shape = tf.shape(inputs)
        ndim = len(shape)
        try:
            assert ndim >= 2
        except AssertionError:
            raise ValueError(f"expected min_ndim=2, found ndim={ndim}. Full shape received: {shape}")

        try:
            assert inputs.shape[-1] == self.in_size
        except AssertionError:
            raise ValueError(f"expected last dimension of inputs to be {self.in_size}, found {shape[-1]}")
        
        # reshape the inputs to (-1, in_size)
        orig_shape = shape[:-1]
        inputs = tf.reshape(inputs, (-1, self.in_size))

        return inputs, orig_shape

    def update_grid_from_samples(self, 
            inputs: tf.Tensor, 
            margin: float=0.01,
            grid_eps: float=0.01
        ):
        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs, _ = self._check_and_reshape_inputs(inputs)

        # calculate the B-spline output
        spline_out = self.calc_spline_output(inputs)

        # build the adaptive grid
        grid = build_adaptive_grid(inputs, self.grid_size, self.spline_order, grid_eps, margin, self.dtype)
        
        # update the spline kernel using the new grid and LS method
        updated_kernel = fit_spline_coef(inputs, spline_out, grid, self.spline_order)

        # assign to the model
        self.grid.assign(grid)
        self.spline_kernel.assign(updated_kernel)
    

    def extend_grid_from_samples(self, 
            inputs: tf.Tensor, 
            extend_grid_size: int,
            margin: float=0.01,
            grid_eps: float=0.01,
            **kwargs
        ):
        # check extend_grid_size
        try:
            assert extend_grid_size >= self.grid_size
        except AssertionError:
            raise ValueError(f"expected extend_grid_size > grid_size, found {extend_grid_size} <= {self.grid_size}")

        # check the inputs, and reshape inputs into 2D tensor (-1, in_size)
        inputs, _ = self._check_and_reshape_inputs(inputs)

        # calculate the B-spline output
        spline_out = self.calc_spline_output(inputs)

        # build the adaptive grid
        # new shape with (in_size, extend_grid_size + 2 * spline_order + 1)
        grid = build_adaptive_grid(inputs, extend_grid_size, self.spline_order, grid_eps, margin, self.dtype)

        # update the spline kernel using the new grid and LS method
        l2_reg, fast = kwargs.pop("l2_reg", 0), kwargs.pop("fast", True)
        updated_kernel = fit_spline_coef(inputs, spline_out, grid, self.spline_order, l2_reg, fast)

        # update the grid and spline kernel
        delattr(self, "grid")
        self.grid = tf.Variable(
            initial_value=tf.cast(grid, dtype=self.dtype),
            trainable=False,
            dtype=self.dtype,
            name="spline_grid"
        )

        self.grid_size = extend_grid_size
        self.spline_basis_size = extend_grid_size + self.spline_order
        delattr(self, "spline_kernel")
        self.spline_kernel = self.add_weight(
            name="spline_kernel",
            shape=(self.in_size, self.spline_basis_size, self.units),
            initializer=tf.keras.initializers.Constant(updated_kernel),
            trainable=True,
            dtype=self.dtype
        )

    def get_config(self):
        config = super(DenseKAN, self).get_config()
        config.update({
            "units": self.units,
            "use_bias": self.use_bias,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": self.grid_range,
            "spline_initialize_stddev": self.spline_initialize_stddev,
            "basis_activation": self.basis_activation
        })

        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)