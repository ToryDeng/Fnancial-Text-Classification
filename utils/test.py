from keras.metrics import sparse_categorical_accuracy
import numpy as np
import tensorflow as tf

y_true = tf.convert_to_tensor(np.array([2, 1], dtype=np.float32))
y_pred = tf.convert_to_tensor(np.array([[0.02, 0.05, 0.83, 0.1], [0.02, 0.05, 0.83, 0.1]], dtype=np.float32))
print(y_true)

print(sparse_categorical_accuracy(y_true, y_pred))