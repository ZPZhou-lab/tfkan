import tensorflow as tf


def build_adaptive_grid(
    x: tf.Tensor, 
    grid_size: int, 
    spline_order: int, 
    grid_eps: float=0.02, 
    margin: float=0.01,
    dtype=tf.float32
) -> tf.Tensor:
    """
    construct the adaptive grid based on the input tensor

    Parameters
    ----------
    x : tf.Tensor
        the input tensor with shape (batch_size, in_size)
    grid_size : int
        the grid size
    spline_order : int
        the spline order
    grid_eps : float, optional
        the weight for combining the adaptive grid and uniform grid, default to `0.02`
    margin : float, optional
        the margin for extending the grid, default to `0.01`
    dtype : tf.DType, optional
        set the data type for the grid, default to `tf.float32`

    Returns
    -------
    grid : tf.Tensor
        The adaptive grid built from the input tensor \
        with shape `(in_size, grid_size + 2 * spline_order + 1)`
    """

    # sort the inputs and build new grid according to the quantiles
    total = tf.shape(x)[0]
    x_sorted = tf.sort(x, axis=0)

    # build the adaptive grid
    adaptive_idx = tf.cast(tf.linspace(0, total - 1, grid_size + 1), tf.int32)
    grid_adaptive = tf.gather(x_sorted, adaptive_idx) # (grid_size + 1, in_size)

    # build the uniform grid
    step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / grid_size
    grid_uniform = x_sorted[0] - margin + tf.range(grid_size + 1, dtype=dtype)[:,None] * step

    # merge the adaptive grid and uniform grid
    # grid with shape (grid_size + 1, in_size)
    grid = grid_eps * grid_uniform + (1 - grid_eps) * grid_adaptive

    # extend left and right bound according to the spline order
    # grid with shape (grid_size + 2 * spline_order + 1, in_size)
    grid = tf.concat(
        [
            grid[:1] - step * tf.range(spline_order, 0, -1, dtype=dtype)[:,None],
            grid,
            grid[-1:] + step * tf.range(1, spline_order + 1, dtype=dtype)[:,None],
        ],
        axis=0
    )

    # transpose the grid to (in_size, grid_size + 2 * spline_order + 1)
    grid = tf.transpose(grid)

    return grid