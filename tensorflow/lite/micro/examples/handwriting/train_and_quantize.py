import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def main(epochs, batch_size, save_model_name):

    # ----prepare dataset----

    (train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(train_data[i], cmap="gray")
        plt.xlabel(train_label[i])
    plt.show()

    train_data = np.expand_dims(train_data.astype(np.float32) - 128.0, axis=-1)
    test_data = np.expand_dims(test_data.astype(np.float32) - 128.0, axis=-1)

    # ----create model----

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(28, 28, 1), filters=8, kernel_size=[3, 3], padding='valid',
                                     activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu))
    model.add(tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='valid', activation=None))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(10))

    model.summary()

    # ----train model----

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_data, train_label, epochs=epochs, batch_size=batch_size, validation_data=(test_data, test_label))

    results = model.evaluate(test_data, test_label, batch_size=batch_size)

    tf.saved_model.save(model, os.path.join("check_point", save_model_name))

    # ----quantize model----

    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(test_data).batch(1).take(10000):
            # Model has only one input so each data point has one element.
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join("check_point", save_model_name))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model_quant = converter.convert()

    with open(os.path.join(os.getcwd(), save_model_name+".tflite"), mode='wb') as f:
        f.write(tflite_model_quant)

    # ----test with quantize model----

    interpreter = tf.lite.Interpreter(model_path=save_model_name+".tflite")
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    scales = output_details[0]["quantization_parameters"]["scales"][0]
    zero_points = output_details[0]["quantization_parameters"]["zero_points"][0]
    interpreter.allocate_tensors()

    test_dataloader = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    test_dataloader = test_dataloader.batch(batch_size=1)

    quantized_test_acc = tf.keras.metrics.SparseCategoricalAccuracy()

    for test_data, test_label in test_dataloader:
        interpreter.set_tensor(input_details[0]["index"], tf.cast(test_data, dtype=tf.int8))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])
        output = (output + (-zero_points)) * scales

        predictions = tf.reshape(output, shape=(output.shape[0], 10))
        quantized_test_acc.update_state(y_true=test_label, y_pred=predictions)

    print("Before quantiz accuracy: {}%".format(results[1] * 100))
    print("Quantized test accuracy: {}%".format(quantized_test_acc.result() * 100))
    print("loss acc : {}%".format((results[1] - quantized_test_acc.result()) * 100))


if __name__ == "__main__":
    parser = ArgumentParser(description="Train cnn model on mnist dataset.")
    parser.add_argument("-epochs", "--epochs", help="set number of epochs", dest="epochs", type=int, default=5)
    parser.add_argument("-bsize", "--batch-size", help="set batch size", dest="batch_size", type=int, default=64)
    parser.add_argument("-save", "--save-model-name", help="set save model name", dest="save_model_name"
                        , type=str, default="mnist_cnn")

    args = parser.parse_args()

    main(args.epochs, args.batch_size, args.save_model_name)
