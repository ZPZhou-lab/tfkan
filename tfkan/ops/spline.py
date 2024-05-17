import tensorflow as tf


def calc_spline_values(x: tf.Tensor, grid: tf.Tensor, spline_order: int):
    """
    Calculate B-spline values for the input tensor.

    Parameters
    ----------
    x : tf.Tensor
        The input tensor with shape (batch_size, in_size).
    grid : tf.Tensor
        The grid tensor with shape (in_size, grid_size + 2 * spline_order + 1).
    spline_order : int
        The spline order.

    Returns: tf.Tensor
        B-spline bases tensor of shape (batch_size, in_size, grid_size + spline_order).
    """
    assert len(tf.shape(x)) == 2
    
    # add a extra dimension to do broadcasting with shape (batch_size, in_size, 1)
    x = tf.expand_dims(x, axis=-1)

    # init the order-0 B-spline bases
    bases = tf.logical_and(
        tf.greater_equal(x, grid[:, :-1]), tf.less(x, grid[:, 1:])
    )
    bases = tf.cast(bases, x.dtype)
    
    # iter to calculate the B-spline values
    for k in range(1, spline_order + 1):
        bases = (
            (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            * bases[:, :, :-1]
        ) + (
            (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)])
            * bases[:, :, 1:]
        )

    return bases


def fit_spline_coef(
        x: tf.Tensor, 
        y: tf.Tensor, 
        grid : tf.Tensor, 
        spline_order: int,
        l2_reg: float=0.0,
        fast: bool=True
    ):
    """
    fit the spline coefficients for given spline input and spline output tensors,\n
    the formula is spline output `y_{i,j} = \sum_{k=1}^{grid_size + spline_order} coef_{i,j,k} * B_{k}(x_i)`\n
    in which, `i=1:in_size, j=1:out_size`. written in matrix form, `Y = B @ coef`,\n
    - `Y` with shape `(batch_size, in_size, out_size)`
    - `B` is the B-spline bases tensor `B_{k}(x_i)` with shape `(batch_size, in_size, grid_size + spline_order)`
    - `coef` is the spline coefficients tensor with shape `(in_size, grid_size + spline_order, out_size)`

    `in_size` is a independent dimension, `coef` transform the `grid_size + spline_order` to `out_size`

    Parameters
    ----------
    x : tf.Tensor
        The given spline input tensor with shape `(batch_size, in_size)`
    y : tf.Tensor
        The given spline output tensor with shape `(batch_size, in_size, out_size)`
    grid : tf.Tensor
        The spline grid tensor with shape `(in_size, grid_size + 2 * spline_order + 1)`
    spline_order : int
        The spline order
    l2_reg : float, optional
        The L2 regularization factor for the least square solver, by default `0.0`
    fast : bool, optional
        Whether to use the fast solver for the least square problem, by default `True`
    
    Returns
    -------
    coef : tf.Tensor
        The spline coefficients tensor with shape `(in_size, grid_size + spline_order, out_size)`
    """

    # evaluate the B-spline bases to get B_{k}(x_i)
    # B with shape (batch_size, in_size, grid_size + spline_order)
    B = calc_spline_values(x, grid, spline_order)
    B = tf.transpose(B, perm=[1, 0, 2]) # (in_size, batch_size, grid_size + spline_order)

    # reshape the output tensor to get Y, put the in_size to the first dimension
    y = tf.transpose(y, perm=[1, 0, 2]) # (in_size, batch_size, out_size)

    # solve the linear equation to get the coef
    # coef with shape (in_size, grid_size + spline_order, out_size)
    coef = tf.linalg.lstsq(B, y, l2_regularizer=l2_reg, fast=fast)

    return coef