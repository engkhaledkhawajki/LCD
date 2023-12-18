import keras.backend as K
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance
from skimage.metrics import structural_similarity as ssim


def msd_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  # Convert y_true and y_pred to float32
  y_true = tf.cast(y_true, dtype='float32')
  y_pred = tf.cast(y_pred, dtype='float32')

  # Calculate the Dice coefficient
  intersection = tf.reduce_sum(y_true * y_pred)
  dice_coef = 1 - (2 * intersection + K.epsilon()) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + K.epsilon())

  # Calculate the Hausdorff distance loss
  dt_pred = tf.numpy_function(scipy.ndimage.distance_transform_edt, [y_pred], tf.float32)
  dt_true = tf.numpy_function(scipy.ndimage.distance_transform_edt, [y_true], tf.float32)
  
  boundary_pred = tf.cast(tf.greater(tf.nn.pool(y_pred, window_shape=(3, 3), strides=(1, 1), padding='SAME', pooling_type='MAX'), y_pred), dtype='float32')
  boundary_true = tf.cast(tf.greater(tf.nn.pool(y_true, window_shape=(3, 3), strides=(1, 1), padding='SAME', pooling_type='MAX'), y_true), dtype='float32')

  boundary_loss = tf.reduce_mean(boundary_pred * dt_pred * boundary_true * dt_true)

  hd_loss = tf.sqrt(dice_coef + boundary_loss)

  msd = hd_loss * (1 - tf.exp(-hd_loss))
  return msd


def iou(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
  """
  Calculate the Intersection over Union (IoU) metric.

  Args:
      y_pred: Predicted binary mask tensor of shape (batch_size, height, width, depth).
      y_true: Ground truth binary mask tensor of shape (batch_size, height, width, depth).

  Returns:
      iou: IoU values for each batch element.
  """
  smooth = 1e-6
  # Compute the intersection and union areas
  intersection = tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])
  union = tf.reduce_sum(y_pred + y_true, axis=[1, 2, 3]) - intersection

  # Compute the IoU as the ratio of intersection to union
  iou = (intersection + smooth) / (union + smooth)

  return iou


class SSIMMetric(tf.keras.metrics.Metric):
  def __init__(self, name='ssim', **kwargs):
    super(SSIMMetric, self).__init__(name=name, **kwargs)
    self.ssim_score = self.add_weight(name='ssim_score', initializer='zeros')
    self.total_samples = self.add_weight(name='total_samples', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    ssim_values = tf.py_function(self._calculate_ssim, [y_true, y_pred], tf.float32)
    self.ssim_score.assign_add(tf.reduce_sum(ssim_values))
    self.total_samples.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

  def result(self):
    return self.ssim_score / self.total_samples

  def reset_states(self):
    self.ssim_score.assign(0.0)
    self.total_samples.assign(0.0)

  def _calculate_ssim(self, y_true, y_pred):
    ssim_values = []
    for i in range(len(y_true)):
      img_true = np.array(y_true[i])
      img_pred = np.array(y_pred[i])
      ssim_values.append(ssim(img_true, img_pred, multichannel=True))
    return tf.convert_to_tensor(ssim_values, dtype=tf.float32)


def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """
  Calculate the precision metric for binary classification.

  Args:
      y_true: Ground truth binary labels tensor.
      y_pred: Predicted binary labels tensor.

  Returns:
      precision_value: Precision metric value.
  """
  # Compute true positives
  true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32))

  # Compute predicted positives
  predicted_positives = tf.reduce_sum(tf.cast(y_pred, dtype=tf.float32))

  # Compute precision
  precision_value = true_positives / (predicted_positives + tf.keras.backend.epsilon())

  return precision_value


def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  """
  Calculate the recall metric for binary classification.

  Args:
      y_true: Ground truth binary labels tensor.
      y_pred: Predicted binary labels tensor.

  Returns:
      recall_value: Recall metric value.
  """
  # Compute true positives
  true_positives = tf.reduce_sum(tf.cast(y_true * y_pred, dtype=tf.float32))

  # Compute actual positives
  actual_positives = tf.reduce_sum(tf.cast(y_true, dtype=tf.float32))

  # Compute recall
  recall_value = true_positives / (actual_positives + tf.keras.backend.epsilon())

  return recall_value