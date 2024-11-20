from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam


train_path = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/train"
validation_path = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate"


trainGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(train_path, target_size=(224, 224), batch_size=32)
validGenerator = ImageDataGenerator(preprocessing_function=preprocess_input).flow_from_directory(validation_path, target_size=(224, 224), batch_size=32)

baseModel = MobileNetV2(weights="imagenet", include_top=False)

x = baseModel.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
x = Dense(128, activation='relu')(x)

predictLayer = Dense(12, activation='softmax')(x)

model = Model(inputs=baseModel.input, outputs=predictLayer)

print(model.summary())


# freeze the base model

for layer in model.layers[:-5]:
    layer.trainable = False


# compile


epochs = 5
optimizer = Adam(learning_rate = 0.0001)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])


# train
model.fit(trainGenerator,  validation_data=validGenerator, epochs=epochs)

path_for_saved_model = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5"
model.save(path_for_saved_model)