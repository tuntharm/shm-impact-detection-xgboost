import sys
import xgboost
import numpy
import pandas

print("Python:", sys.version)
print("XGBoost:", xgboost.__version__)
print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Scikit-learn:", sys.modules['sklearn'].__version__)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))

if tf.test.gpu_device_name():
    print("Default GPU:", tf.test.gpu_device_name())
else:
    print("❌ GPU not found. Running on CPU.")

from tensorflow.python.platform import build_info as tf_build_info
print("CUDA version:", tf_build_info.build_info['cuda_version'])
print("cuDNN version:", tf_build_info.build_info['cudnn_version'])
