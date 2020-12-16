import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

TF_PATH = "mnist_cnn_tf"
TFLITE_PATH = "mnist_dynamic_quantized_28x28.tflite"

# Load the mnist dataset
ds, info = tfds.load('mnist', split='train', with_info=True)

# Make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative data generator function
def representative_data_gen():
    for input_value in ds.batch(1).take(10000):
        yield [tf.reshape(np.array(input_value["image"],dtype="float32"),[1,1,28,28])]

# Set optimization Strategy and representative dataloader of converter.
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Set the input and output tensors to int8 
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Start quantize the model
tflite_model_quant = converter.convert()

# Save the model.
with open(os.path.join(os.getcwd(), TFLITE_PATH), mode='wb') as f:
        f.write(tflite_model_quant)

print("Convert Success.")

# ----test with quantize model----
ds, info = tfds.load('mnist', split='test', with_info=True)

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
scales = output_details[0]["quantization_parameters"]["scales"][0]
zero_points = output_details[0]["quantization_parameters"]["zero_points"][0]
interpreter.allocate_tensors()

quantized_test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

for example in ds:
    image = np.array((example["image"]-128),dtype="int8")
    label = example["label"]

    inputs = tf.reshape(image,[1,1,28,28])

    interpreter.set_tensor(input_details[0]["index"], tf.cast(inputs, dtype=tf.int8))
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    output = (output + (-zero_points)) * scales
    
    quantized_test_acc.update_state(y_true=label, y_pred=output)


print("Quantized test accuracy: {}%".format(quantized_test_acc.result() * 100))