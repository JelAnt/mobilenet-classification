import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tkinter as tk


categories = os.listdir(
    "garbage_classification_dataset_model/validate"
)  # get a list of all folders name inside the validation folder

model = load_model("model.h5")  # load the object detection model

cap = cv2.VideoCapture(0)  # 0 is the default camera, load it up.

if not cap.isOpened():  # check if camera is working
    print("camera error")
    exit()


input_size = (224, 224)  # specify the frame size

while True:
    ret, frame = cap.read()  # read the frame from the camera
    if not ret:
        break  # if the frame was not read succesfully break out

    resized_frame = cv2.resize(frame, input_size)  # resize the frame
    input_frame = (
        np.expand_dims(resized_frame, axis=0) / 255.0
    )  # adds a batch dimension and normalizes the pixel values from 0 to 1

    prediction = model.predict(input_frame).ravel()  # predicts the current frame
    predicted_class = np.argmax(prediction)  # assigns the class based on probability
    predicted_label = categories[
        predicted_class
    ]  # assigns the label based on the previously assigned class
    confidence = (
        prediction[predicted_class] * 100
    )  # calculates and assigns the confidence score

    if (
            confidence < 60
        ):  # if the confidence score is less than 60% assign "unknown" to the prediction label
            predicted_label = "unknown"

    text = f"Category: {predicted_label}, Confidence: {confidence:.2f}%"  # create text that will be shown on the frames
    cv2.putText(
        frame,
        text,
        (10, 30),  # place the text on the frame
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )  # specify text attributes like the color, thickness, font size, font type etc.
    cv2.imshow("Video detection", frame)  # display the video

    if cv2.waitKey(1) & 0xFF == ord("q"):  # breaks out of the loop if q is pressed
        break


cap.release()  # stops using the camera
cv2.destroyAllWindows()  # closes all windows
