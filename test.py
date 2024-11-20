import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

categories = os.listdir("C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate")
categories.sort()
print(categories)


#load detection model

path_for_saved_model = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5"
model = tf.keras.models.load_model(path_for_saved_model)

print(model.summary())

def classify_image(imageFile):
    x = []

    img = Image.open(imageFile)
    img.load()
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(x.shape)

    pred = model.predict(x)

    categoryValue = np.argmax(pred, axis=1)

    categoryValue = categoryValue[0]
    print(categoryValue)


    result = categories[categoryValue]

    return result

imagePath = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate/clothes/clothes1176.jpg"
resultText = classify_image(imagePath)
print(resultText)


img = cv2.imread(imagePath)
img = cv2.putText(img, resultText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()