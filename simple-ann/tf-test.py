#!/usr/bin/env python3

import tensorflow as tf
import time

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Create a large tensor and perform a computation to utilize the GPU
with tf.device('/GPU:0'):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    start_time = time.time()
    c = tf.matmul(a, b)
    end_time = time.time()
    print("Matrix multiplication completed in:", end_time - start_time, "seconds")
