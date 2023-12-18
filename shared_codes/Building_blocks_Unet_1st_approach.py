from keras.engine.training import optimizer
import tensorflow as tf
from tensorflow.keras import layers
from typing import List, Tuple
from typing import Callable
import os

def double_conv_block(x: tf.Tensor, n_filters: int) -> tf.Tensor:
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def downsample_block(x: tf.Tensor, n_filters: int) -> Tuple[tf.Tensor, tf.Tensor]:
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p

def upsample_block(x: tf.Tensor, conv_features: tf.Tensor, n_filters: int) -> tf.Tensor:
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    x = layers.concatenate([x, conv_features])
    x = layers.Dropout(0.3)(x)
    x = double_conv_block(x, n_filters)
    return x

def create_and_compile_unet_model(IMG_HEIGHT: int = 128,
                    IMG_WIDTH: int = 128,
                    IMG_CHANNELS: int = 1,
                    loss: Callable=None,
                    metrics: List=None,
                    optimizer: Callable=None) -> tf.keras.Model:
                    
    inputs = layers.Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    
    f1, p1 = downsample_block(inputs, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    bottleneck = double_conv_block(p4, 1024)

    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)

    outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")

    unet_model.compile(optimizer=optimizer,
                   loss=loss,
                  metrics=metrics)

    return unet_model