import tensorflow as tf


model = tf.keras.models.load_model("C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5")


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()


with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")
