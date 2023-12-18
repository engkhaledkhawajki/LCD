from keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from typing import List, Tuple
from typing import Callable
import os

def create_and_compile_unet_model_pretrained_encoder(n_classes=2,
                                                    IMG_HEIGHT=128,
                                                    IMG_WIDTH=128,
                                                    IMG_CHANNELS=1,
                                                    metrics=None,
                                                    loss=None):

  BACKBONE = 'resnet34'
  preprocess_input = sm.get_preprocessing(BACKBONE)
  # define model
  model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=2, activation='softmax')
  opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
  # compile keras model with defined optimozer, loss and metrics
  #model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
  model_resnet_backbone.compile(optimizer=opt,
                                loss=loss,
                                metrics=metrics)


  return model_resnet_backbone


def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def create_and_compile_unet_model(n_classes=2,
                          IMG_HEIGHT=128,
                          IMG_WIDTH=128,
                          IMG_CHANNELS=3,
                          metrics=None,
                          loss=None,
                          optimizer=None):
    """ Input """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))

    """ Pre-trained Encoder """
    encoder = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_tensor=inputs)

    s1 = encoder.get_layer("input_4").output                      ## 128
    s2 = encoder.get_layer("block_1_expand_relu").output    ## 64
    s3 = encoder.get_layer("block_3_expand_relu").output    ## 32
    s4 = encoder.get_layer("block_6_expand_relu").output    ## 16

    """ Bottleneck """
    b1 = encoder.get_layer("block_8_expand_relu").output    ## 8

    """ Decoder """
    d1 = decoder_block(b1, s4, 256)                               ## 32
    d2 = decoder_block(d1, s3, 128)                               ## 64
    d3 = decoder_block(d2, s2, 64)                               ## 128
    d4 = decoder_block(d3, s1, 32)                                ## 256

    """ Output """
    outputs = Conv2D(n_classes, 2, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs)

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics = metrics)

    return model