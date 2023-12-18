import keras.backend as K
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
import tensorflow as tf
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

def weighted_log_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  # Scale predictions so that the class probabilities of each sample sum to 1
  y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
  # Clip to prevent NaN's and Inf's
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
  # Weights are assigned in this order: normal, necrotic, edema, enhancing
  weights = np.array([1, 5, 2, 4], dtype=np.float32)
  # Calculate the loss
  loss = y_true * K.log(y_pred) * weights
  loss = K.mean(-K.sum(loss, axis=-1))
  return loss

def gen_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  y_true_f = K.reshape(y_true, shape=(-1, 4))
  y_pred_f = K.reshape(y_pred, shape=(-1, 4))
  sum_p = K.sum(y_pred_f, axis=-2)
  sum_r = K.sum(y_true_f, axis=-2)
  sum_pr = K.sum(y_true_f * y_pred_f, axis=-2)
  weights = K.pow(K.square(sum_r) + K.epsilon(), -1)
  generalised_dice_numerator = 2 * K.sum(weights * sum_pr)
  generalised_dice_denominator = K.sum(weights * (sum_r + sum_p))
  generalised_dice_score = generalised_dice_numerator / generalised_dice_denominator
  GDL = 1 - generalised_dice_score
  return GDL

def TverskyLoss(y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6) -> tf.Tensor:
  inputs = K.flatten(y_true)
  targets = K.flatten(y_pred)

  TP = K.sum((inputs * targets))
  FP = K.sum(((1 - targets) * inputs))
  FN = K.sum((targets * (1 - inputs)))

  Tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

  return 1 - Tversky

class SurfaceLoss:
  def __init__(self, idc: List[int]) -> None:
    self.idc = idc

  def __call__(self, probs: tf.Tensor, dist_maps: tf.Tensor) -> tf.Tensor:
    pc = tf.cast(probs[:, self.idc, ...], tf.float32)
    dc = tf.cast(dist_maps[:, self.idc, ...], tf.float32)

    multipled = tf.einsum("bkwh,bkwh->bkwh", pc, dc)

    loss = tf.reduce_mean(multipled)

    return loss

def calc_dist_map(seg: np.ndarray) -> np.ndarray:
  res = np.zeros_like(seg)
  posmask = seg.astype(np.bool)

  if posmask.any():
    negmask = ~posmask
    res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
  return res

def calc_dist_map_batch(y_true: tf.Tensor) -> np.ndarray:
  y_true_numpy = y_true.numpy()
  return np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)

def surface_loss_v2(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
  multipled = y_pred * y_true_dist_map
  multipled = tf.cast(multipled, dtype=tf.float32)
  mean = K.mean(multipled)
  return mean


def surface_loss(y_pred: tf.Tensor, y_true: tf.Tensor) -> tf.Tensor:
  y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
  multipled = y_pred * y_true_dist_map
  multipled = tf.cast(multipled, dtype=tf.float32)
  mean = K.mean(multipled)
  return mean



def modified_surface_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
  y_true = K.cast(y_true, dtype='float32')
  y_pred = K.cast(y_pred, dtype='float32')
  y_true_dist_map = tf.py_function(func=calc_dist_map_batch, inp=[y_true], Tout=tf.float32)
  multipled = y_pred * y_true_dist_map
  multipled = tf.cast(multipled, dtype=tf.float32)
  mean = K.mean(multipled)
  hd_loss = tf.divide(K.sqrt(mean), mean)
  ms_loss = hd_loss * (1 - K.exp(-hd_loss))
  return ms_loss

def dice_coef(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
  class_num = 1
  total_loss = 0.0
  for i in range(class_num):
    y_true_f = K.flatten(y_true[:, :, :, i])
    y_pred_f = K.flatten(y_pred[:, :, :, i])
    intersection = K.sum(y_true_f * y_pred_f)
    loss = ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    if i == 0:
      total_loss = loss
    else:
      total_loss = total_loss + loss
  total_loss = total_loss / class_num
  return total_loss

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
  return 1 - dice_coef(y_true, y_pred, smooth)