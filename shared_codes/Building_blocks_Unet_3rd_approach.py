from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, Activation
import tensorflow as tf
import segmentation_models as sm
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from typing import List, Tuple
from typing import Callable
import os

def create_and_compile_unet_model(n_classes=2,
                                  IMG_HEIGHT=128,
                                  IMG_WIDTH=128,
                                  IMG_CHANNELS=1,
                                  metrics=None,
                                  loss=None,
                                  optimizer=None):

  BACKBONE = 'resnet34'
  preprocess_input = sm.get_preprocessing(BACKBONE)
  # define model
  model_resnet_backbone = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=2, activation='softmax')
  # compile keras model with defined optimozer, loss and metrics
  #model_resnet_backbone.compile(optimizer='adam', loss=focal_loss, metrics=metrics)
  model_resnet_backbone.compile(optimizer=optimizer,
                                loss=loss,
                                metrics=metrics)


  return model_resnet_backbone