import tensorflow as tf


def train_model(model=None,
                NUM_EPOCHS=None,
                BATCH_SIZE=None,
                fname=None,
                X_train=None,
                y_train=None,
                X_test=None,
                y_test=None):

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
  model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=fname, monitor='val_loss', mode='min', save_best_only=True)
  callbacks = [model_checkpoint]

  STEPS_PER_EPOCH = len(X_train) // BATCH_SIZE
  VALIDATION_STEPS = (0.1 * len(X_train)) // BATCH_SIZE

  model_history = model.fit(X_train,
                            y_train,
                            epochs=NUM_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks)

  return model_history, model

def predict_after_train(model=None, test_data=None):
  y_pred = model.predict(test_data)

  return y_pred


def load_model_and_predict(model=None, path=None, test_data=None):
  model.load_weights(path)
  y_pred = model.predict(test_data)

  # y_pred = np.argmax(y_pred, axis=-1)

  return y_pred, model