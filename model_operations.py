import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to your dataset folder
dataset_path = r'C:\Users\Hariharan\OneDrive\Desktop\Loghitha\Medicinal plant dataset'

def load_image(filename):
    file_path = os.path.join(dataset_path, filename)
    if not os.path.exists(file_path):
        file_path = filename  # Check in current directory if not found in dataset path
    img = load_img(file_path, color_mode="rgb", target_size=(100, 100))
    img = img_to_array(img)
    img = img.reshape(1, 100, 100, 3)
    img = img.astype('float32') / 255.0
    return img

def capture_and_predict(model, plant_categories, result_label, captured_image_label):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Capture Image (Press SPACE to capture)', frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()

    captured_image_path = "captured_leaf.jpg"
    cv2.imwrite(captured_image_path, frame)

    img = load_image(captured_image_path)
    class_prediction = model.predict(img)
    predicted_class = np.argmax(class_prediction, axis=1)
    predicted_plant = plant_categories[predicted_class[0]]

    result_label.config(text=f"Predicted Plant: {predicted_plant}")

    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(img)
    captured_image_label.config(image=img)
    captured_image_label.image = img

def select_and_predict(model, plant_categories, result_label, captured_image_label):
    file_path = filedialog.askopenfilename(initialdir=dataset_path)
    if file_path:
        relative_path = os.path.relpath(file_path, dataset_path)
        img = load_image(relative_path)
        class_prediction = model.predict(img)
        predicted_class = np.argmax(class_prediction, axis=1)
        predicted_plant = plant_categories[predicted_class[0]]

        result_label.config(text=f"Predicted Plant: {predicted_plant}")

        img = Image.open(file_path)
        img = img.resize((200, 200), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        captured_image_label.config(image=img)
        captured_image_label.image = img


