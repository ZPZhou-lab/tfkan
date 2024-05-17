import tensorflow as tf
from keras.layers import Layer
from ..ops.spline import calc_spline_values, fit_spline_coef
from ..ops.grid import build_adaptive_grid

from typing import Tuple, List, Any, Union, Callable
from abc import ABC, abstractmethod


class LayerKAN:
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    def calc_spline_output(self, inputs):
        """
        calculate the spline output, each feature of each sample is mapped to `out_size` features, \
        using `out_size` different B-spline basis functions, so the output shape is `(batch_size, in_size, out_size)`

        Parameters
        ----------
        inputs : tf.Tensor
            the input tensor with shape `(batch_size, in_size)`
        
        Returns
        -------
        spline_out : tf.Tensor
            the output tensor with shape `(batch_size, in_size, out_size)`
        """
        # calculate the B-spline output
        spline_in = calc_spline_values(inputs, self.grid, self.spline_order) # (B, in_size, grid_basis_size)
        # matrix multiply: (batch, in_size, grid_basis_size) @ (in_size, grid_basis_size, out_size) -> (batch, in_size, out_size)
        spline_out = tf.einsum("bik,iko->bio", spline_in, self.spline_kernel)

        return spline_out

    @abstractmethod
    def update_grid_from_samples(self, 
            inputs: tf.Tensor, 
            margin: float=0.01,
            grid_eps: float=0.01
        ):
        """
        update the grid based on the inputs adaptively

        Parameters
        ----------
        inputs : tf.Tensor
            the input tensor with shape (batch_size, dim1, dim2, ..., in_size)
        margin : float, optional
            the margin for extending the grid, default to `0.01`, \
            the grid range will be extended into `[min - margin, max + margin]`
        grid_eps : float, optional
            the weight for combining the adaptive grid and uniform grid, default to `0.02`, \n
            the combined grid will be `grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive`
        """

        raise NotImplementedError
    
    @abstractmethod
    def extend_grid_from_samples(self, 
            inputs: tf.Tensor, 
            extend_grid_size: int,
            margin: float=0.01,
            grid_eps: float=0.01,
            l2_reg: float=0.0,
            fast: bool=True
        ):
        """
        extend the grid based on the inputs adaptively

        Parameters
        ----------
        inputs : tf.Tensor
            the input tensor with shape (batch_size, dim1, dim2, ..., in_size)
        extend_grid_size : int
            the number of grid points after extending the grid
        margin : float, optional
            the margin for extending the grid, default to `0.01`, \
            the grid range will be extended into `[min - margin, max + margin]`
        grid_eps : float, optional
            the weight for combining the adaptive grid and uniform grid, default to `0.02`, \n
            the combined grid will be `grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive`
        l2_reg : float, optional
            The L2 regularization factor for the least square solver, by default `0.0`
        fast : bool, optional
            Whether to use the fast solver for the least square problem, by default `True`
        """

        raise NotImplementedError