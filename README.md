# tfkan
Here is a tensorflow implementation of `KAN` ([Kolmogorov-Arnold Networks (KANs)](https://github.com/KindXiaoming/pykan))

## How to use

Download or clone the repository and run in the terminal:
```bash
cd tfkan && pip install .
```

then you can use `tfkan` packages: 

```python
from tfkan import layers
from tfkan.layers import DenseKAN, Conv2DKAN
```

## Features

The modules in `layers` are similar to those in `tf.keras.layers`, so you can use these layers as independent modules to assemble the model (in a `TensorFlow` style)

```python
from tfkan.layers import DenseKAN
# create model using KAN
model = tf.keras.models.Sequential([
    DenseKAN(4),
    DenseKAN(1)
])
model.build(input_shape=(None, 10))
```

When calling `model.summary()` you can see the model structure and its trainable parameters.

```python
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_kan (DenseKAN)        (None, 4)                 360       
                                                                 
 dense_kan_1 (DenseKAN)      (None, 1)                 36        
                                                                 
=================================================================
Total params: 396 (1.55 KB)
Trainable params: 396 (1.55 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

When getting started quickly, we can define the optimizer and loss function used for training through `model.compile()` and `model.fit()`, or you can use `tf.GradientTape()` to more finely control the model training behavior and parameter update logic. All model behaviors are the same in `Tensorflow2`, and you can truly use `KAN` as an independent module.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
    loss='mse', metrics=['mae'])
history = model.fit(x_train, y_train, epochs=100, batch_size=64)
```

## Layers

The layers currently implemented include:
- `DenseKAN`
    - The basic dense layer, corresponding to `tf.keras.layers.dense()` (or `nn.Linear()` in torch) in MLP.
    - Implement the computational logic described in the KANs paper.
    - **grid update method** is available but it will not be automatically used.

- `Conv2DKAN`
    - THe basic 2D image convolution layer, corresponding to `tf.keras.layers.Conv2D()`.
    - The implementation logic is similar to `Conv2D`, expanding convolution operations into matrix multiplication, and then replacing MLP dense layer with `DenseKAN`.

## About Grid Update

The **grid adaptive update** is an important feature mentioned in KANs paper. In this tensorflow implementation of KANs, **each KAN layer has a method** `self.update_grid_from_samples(...)` used to implement this feature. You can call it in **custom training logic** or use Tensorflow `Callbacks`

- In custom training logic
```python
def train_model(...):
    # training logic
    ...

    # call update_grid_from_samples
    for layer in model.layers:
        if hasattr(layer, 'update_grid_from_samples'):
            layer.update_grid_from_samples(x)
        x = layer(x)
```

- or use Tensorflow `Callbacks`
```python
# define update grid callback
class UpdateGridCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        """
        update grid before new epoch begins
        """
        global x_train, batch_size
        x_batch = x_train[:batch_size]
        if epoch > 0:
            for layer in self.model.layers:
                if hasattr(layer, 'update_grid_from_samples'):
                    layer.update_grid_from_samples(x_batch)
                x_batch = layer(x_batch)
```
then add it into `model.fit()`