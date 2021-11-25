import onnx
from onnx_tf.backend import prepare

ONNX_PATH = "mnist_cnn.onnx"
TF_PATH = "mnist_cnn_tf"

# Load the ONNX model
onnx_model = onnx.load(ONNX_PATH)

# Export ONNX model to TensorFlow model
tf_model = prepare(onnx_model)
tf_model.export_graph(TF_PATH)


import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load the mnist dataset
ds, info = tfds.load('mnist', split='test', with_info=True)

# Load the TensorFlow model
imported = tf.saved_model.load(TF_PATH)
f = imported.signatures["serving_default"]

# ----test with TensorFlow model----
tf_test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

for example in ds:
    image = np.array(example["image"],dtype="float32")
    label = example["label"]
    inputs = tf.reshape(image,[1,1,28,28])

    output = f(inputs)['output_0']
    tf_test_acc.update_state(y_true=label, y_pred=output)

print("TensorFlow model test accuracy: {}%".format(tf_test_acc.result() * 100))