from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import tensorflow as tf


train_path = "garbage_classification_dataset_model/train"
validation_path = "garbage_classification_dataset_model/validate"  # assigning paths to the train and validate dataset


early_stopping = EarlyStopping(  # sets up a callback which will be used to stop fine tuning if the validation loss doesn't improve after 3 epochs
    monitor="val_loss",  # selects which metric to monitor
    patience=3,  # specifies number of epochs
    restore_best_weights=True,  # at the end selects weights from best epoch
)


train_datagen = ImageDataGenerator(  # creates an imagedatagenerator that specifies which augmentation techniques will be used
    preprocessing_function=preprocess_input,  # normalizes pixel values
    rotation_range=40,  # randomly rotates image
    width_shift_range=0.2,  # randomly shifts image
    height_shift_range=0.2,
    shear_range=0.2,  # applies random shear
    zoom_range=0.2,  # randomly zooms
    horizontal_flip=True,  # randomly flips image horizontally
    fill_mode="nearest",  # fills any gaps using the nearest pixel value
)

trainGenerator = (
    train_datagen.flow_from_directory(  # performs augmentation on a specified directory
        train_path,  # specifies directory path
        target_size=(224, 224),  # resizes images to fit the supported mobilenet format
        batch_size=64,  # batches images to groups of 64
    )
)


validGenerator = ImageDataGenerator(
    preprocessing_function=preprocess_input
).flow_from_directory(  # performs preprocessing on validation images to prepare them
    validation_path,  # specifies path
    target_size=(224, 224),  # resizes
    batch_size=64,  # batches images to a group of 64
)

baseModel = MobileNetV2(
    weights="imagenet", include_top=False
)  # loading the model without a top layer so a new one can be added that fits trash classification

x = baseModel.output  # assign last layer's output
x = GlobalAveragePooling2D()(x)  # reduces the spatial dimensions
x = Dense(512, activation="relu")(x)
x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(
    x
)  # add fully connected dense layers to the top of the output.

predictLayer = Dense(12, activation="softmax")(
    x
)  # adds the final classification layer with 12 classes. uses softmax to convert output values into probabilities

model = Model(
    inputs=baseModel.input, outputs=predictLayer
)  # creates new model using the input of the base model and the output of the new prediction layers

print(model.summary())  # prints a summary of the new layer


for layer in model.layers[
    :-5
]:  # freezes base layers so that they don't change. only newly added layers will be trained
    layer.trainable = False


epochs = 30  # specifies the number of epochs
optimizer = Adam(
    learning_rate=0.0001
)  # sets up an adam optimizer with a low learning rate to ensure stable training

model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)  # compiles the model

# Train the model with augmented training data
history = model.fit(
    trainGenerator,
    validation_data=validGenerator,
    epochs=epochs,
    callbacks=[early_stopping],
)  # trains the model using the trainining and validation data generator. also specifies the number of epochs and applies early stopping


model.save("model.h5")  # saves the model to the specified path


# Plotting

train_loss = history.history["loss"]
val_loss = history.history["val_loss"]
train_accuracy = history.history["accuracy"]
val_accuracy = history.history[
    "val_accuracy"
]  # extracts metrics from the training history


# plots the training and validation loss
plt.figure(figsize=(10, 5))  # creates a new figure with specified dimensions
plt.plot(train_loss, label="Training Loss")
plt.plot(val_loss, label="Validation Loss")  # plots losses
plt.title("Loss Over Epochs")  # sets the title
plt.xlabel("Epochs")
plt.ylabel("Loss")  # sets the x and y labels
plt.legend()  # enables a legend
plt.show()  # dispalys the plot

# plots the training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(val_accuracy, label="Validation Accuracy")
plt.title("Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
