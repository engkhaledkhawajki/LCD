import cv2
import numpy as np
import tensorflow as tf
from typing import List, Tuple
from sklearn.preprocessing import OneHotEncoder
from tqdm.notebook import tqdm as tq


class Preprocess2dData:
  def __init__(self) -> None:
      pass
      
  def resize(self, input_image: np.ndarray, input_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    input_image = input_image.astype(np.float32)
    input_mask = input_mask.astype(np.float32)
    input_image = cv2.resize(input_image, dsize=(128, 128))
    input_mask = cv2.resize(input_mask, dsize=(128, 128))
    return input_image, input_mask
  
  def expand_dimension(self, X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    if len(X_train.shape) < 4 and len(y_train.shape) < 4:
      X_train = np.expand_dims(X_train, axis=-1)
      y_train = np.expand_dims(y_train, axis=-1)
    return X_train, y_train
  
  def normalize(self, input_image: tf.Tensor, input_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    input_image = tf.cast(input_image, tf.float32) / 255.0
    return input_image, input_mask
  
  def make_masks_onehot2d(self, batch_ys: np.ndarray, num_classes: int) -> np.ndarray:
    batch, width, height, channels = batch_ys.shape
    reshaped_mask = batch_ys.reshape((-1, channels))
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=num_classes)
    mask_one_hot = encoder.fit_transform(reshaped_mask)
    mask_one_hot = mask_one_hot.reshape(batch_ys.shape[:-1] + (num_classes,))
    return mask_one_hot
  
  def preprocess(self, imgs: List[np.ndarray], masks: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    xt = []
    yt = []
    if len(imgs) <= len(masks):
      for i in tq(range(len(imgs))):
        x, y = self.resize(imgs[i], masks[i])
        x, y = self.normalize(x, y)
        xt.append(x)
        yt.append(y)
    else:
      for i in tq(range(len(masks))):
        x, y = self.resize(imgs[i], masks[i])
        x, y = self.normalize(x, y)
        xt.append(x)
        yt.append(y)
    return xt, yt
    