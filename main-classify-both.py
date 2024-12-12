import subprocess
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import tkinter as tk
from tkinter import *
from tkinter import Tk, Button, filedialog
from PIL import Image, ImageTk
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def classify_video():  # method that runs the model on a live video feed

    cap = cv2.VideoCapture(0)  # 0 is the default camera, load it up.

    if not cap.isOpened():  # check if camera is working
        print("camera error")
        exit()

    input_size = (224, 224)  # specify the frame size

    while True:
        ret, frame = cap.read()  # read the frame from the camera
        if not ret:
            break  # if the frame was not read successfully, break out

        resized_frame = cv2.resize(frame, input_size)  # resize the frame
        input_frame = (
            np.expand_dims(resized_frame, axis=0) / 255.0
        )  # adds a batch dimension and normalizes the pixel values from 0 to 1

        prediction = model.predict(input_frame).ravel()  # predicts the current frame
        predicted_class = np.argmax(
            prediction
        )  # assigns the class based on probability
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

        text1 = f"Category: {predicted_label}"  # create text that will be shown on the frames
        cv2.putText(
            frame,
            text1,
            (10, 30),  # place the text on the frame
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )  # specify text attributes like the color, thickness, font size, font type etc.


        text2 = f"Confidence: {confidence:.2f}%"  # create 2nd text with the confidence
        cv2.putText(
            frame,
            text2,
            (10, 70),  # place the text on the frame
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


def select_image():  # method that prompts the user to select an image file

    imageFile = filedialog.askopenfilename(  # opens a browse window which allows the user to select a file. only allows files in png, jpg or jpeg formats (so images)
        title="Select an Image", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*")]
    )

    if not imageFile:  # if nothing is selected return nothing
        return

    classify_selected_image(
        imageFile
    )  # calls the classify image function with the path to the selected image file


def classify_selected_image(
    imageFile,
):  # method that classifies an image using the model. takes in a image path as a parameter.
    x = []  # x will hold the image information

    img = Image.open(imageFile)
    img.load()  # loads an image
    img = img.resize(
        (224, 224), Image.Resampling.LANCZOS
    )  # resizes the image to 224x224 to fit the model

    x = image.img_to_array(img)  # converts the image to an array and assigns it to x
    x = cv2.cvtColor(
        x, cv2.COLOR_BGRA2BGR
    )  # converts image to BGR because it is a supported format by CV2
    x = np.expand_dims(x, axis=0)  # adds a batch dimension
    x = preprocess_input(x)  # preprocesses the image

    pred = model.predict(x)  # run the prediction

    categoryValue = np.argmax(
        pred, axis=1
    )  # specify the category value based on the prediction

    categoryValue = categoryValue[0]  # gets the most likely category value
    result = categories[categoryValue]  # assigns the category based on the value

    confidence = np.max(pred) * 100 # calculate confidence

    result_window = Toplevel(root)  # creates a child window of the main tk window
    result_window.title("classification result")  # changes the window's title
    result_window.geometry("500x500")  # specifies the dimensions

    img_display = ImageTk.PhotoImage(img)  # converts image to a tk supported format
    img_label = Label(
        result_window, image=img_display
    )  # creates a label widget which will hold the image
    img_label.image = (
        img_display  # keeps a reference to avoid a garbage collection issue
    )
    img_label.pack(pady=10)  # places the image on the window and adds padding

    result_label = Label(  # creates a label to display the result
        result_window, text=f"Predicted Category: {result}\n Confidence: {confidence:.2f}", font=("Helvetica", 18)
    )
    result_label.pack(pady=10)  # places the label on the window and adds padding


def exit_app():  # method to close the tk window
    root.destroy()


categories = os.listdir(
    "garbage_classification_dataset_model/validate"
)  # get a list of all folders name inside the validation folder

model = load_model("model.h5")  # load the object detection model


root = Tk()  # creates the main windows
root.title("Mobilenet classification program")  # creates a title
root.geometry("600x425")  # specifies the window dimensions


btn_video = Button(  # creates a button widget
    root,
    text="Classify Video",
    font=("Helvetica", 16),
    command=classify_video,
    width=40,
    height=4,  # specifies which function to run after the button is pressed. also width and height.
)
btn_video.pack(pady=15)  # places the button and gives a padding


btn_image = Button(
    root,
    text="Classify Image",
    font=("Helvetica", 16),
    command=select_image,
    width=40,
    height=3,
)
btn_image.pack(pady=15)


btn_exit = Button(
    root, text="Exit", font=("Helvetica", 16), command=exit_app, width=40, height=3
)
btn_exit.pack(pady=15)


root.mainloop()  # starts tk loop
