import tensorflow as tf

# Load the Keras HDF5 model
model = tf.keras.models.load_model("C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5")

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model successfully converted to TFLite format!")
