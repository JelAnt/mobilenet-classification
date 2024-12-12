import tensorflow as tf


model = tf.keras.models.load_model(
    "garbage_classification_dataset_model/model.h5"
)  # loads the model


converter = tf.lite.TFLiteConverter.from_keras_model(
    model
)  # creates a tftlite converter object using the previously loaded h5 model
tflite_model = converter.convert()  # performs the conversion from h5 to tflite


with open("model.tflite", "wb") as f:  # writes the converted model to a file
    f.write(tflite_model)
