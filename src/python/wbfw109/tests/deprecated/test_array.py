# %%
import tensorflow as tf
import numpy as np

# also can use arithmetic operation
test_a = np.arange(1, 7).reshape(2, 3)
test_b = np.arange(7, 13).reshape(3, 2)
test_a
test_b
test_c_np = np.dot(test_a, test_b)
test_c_np
test_c_tf = tf.convert_to_tensor(test_c_np)
test_c_tf
test_c_tf.numpy()
tf.convert_to_tensor(test_a) * tf.convert_to_tensor(test_a)
# can convert to numpy from Tensor
# can convert to Tensor from numpy
