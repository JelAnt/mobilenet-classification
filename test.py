import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

categories = os.listdir("C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate") # get a list of all folders name inside the validation folder
categories.sort() #sorts all the categories
print(categories) # print to check if its correct



path_for_saved_model = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5" # path to the model
model = tf.keras.models.load_model(path_for_saved_model) #loads the model

print(model.summary()) # print the model summary to check if everything is correct

def classify_image(imageFile): # create function to classify images
    x = [] # x will hold the image information

    img = Image.open(imageFile)
    img.load() # loads an image
    img = img.resize((224, 224), Image.Resampling.LANCZOS) # resizes the image to 224x224 to fit the model

    x = image.img_to_array(img) # converts the image to an array and assigns it to x
    x = np.expand_dims(x, axis=0) # adds a batch dimension
    x = preprocess_input(x) # preprocesses the image
    print(x.shape) 

    pred = model.predict(x) # run the prediction

    categoryValue = np.argmax(pred, axis=1) # specify the category value based on the prediction
 
    categoryValue = categoryValue[0] # gets the most likely category value
    result = categories[categoryValue] # assigns the category based on the value

    return result 

imagePath = "C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate/clothes/clothes1176.jpg" # path to a test image
resultText = classify_image(imagePath) # run the classification function on this test image
print(resultText) #prints the result


img = cv2.imread(imagePath) # creates a windows to display the image
img = cv2.putText(img, resultText, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA) # puts the result text on the image window

cv2.imshow("image", img) # shows the window
cv2.waitKey(0) 
cv2.destroyAllWindows()  # windows stays up until any key is pressed which causes it to close