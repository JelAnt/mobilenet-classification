import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

categories = os.listdir("C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/validate")


model = load_model('C:/Users/marko/Desktop/computer vision project/garbage_classification_dataset_model/model.h5')

video_path = 0
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


input_size = (224, 224) 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    resized_frame = cv2.resize(frame, input_size)
    input_frame = np.expand_dims(resized_frame, axis=0) / 255.0 

    
    prediction = model.predict(input_frame).ravel()
    predicted_class = np.argmax(prediction)  
    predicted_label = categories[predicted_class]  
    confidence = prediction[predicted_class] * 100

    text = f"Category: {predicted_label}, Confidence: {confidence:.2f}%"
    cv2.putText(frame, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Video Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
