import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
import tensorflow as tf
from model_operations import load_image, capture_and_predict, select_and_predict

# Path to your dataset folder
dataset_path = r'C:\Users\Hariharan\OneDrive\Desktop\Loghitha\Medicinal plant dataset'

# Load the trained model
model_path = 'models/Indian_medicinal_plants_classification_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = tf.keras.models.load_model(model_path)

# Define the plant categories (adjust based on your dataset structure)
plant_categories = sorted(os.listdir(dataset_path))

# GUI setup
root = tk.Tk()
root.title("Leaf Identification")

# Centering the window on the screen
window_width = 670
window_height = 480
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height / 2 - window_height / 2)
position_right = int(screen_width / 2 - window_width / 2)
root.geometry(f'{window_width}x{window_height}+{position_right}+{position_top}')

# Adding background image
bg_image_path = "Medicinal Leaf.jpg"
if not os.path.exists(bg_image_path):
    raise FileNotFoundError(f"Background image file not found at {bg_image_path}")
bg_image = Image.open(bg_image_path)
bg_image = bg_image.resize((window_width, window_height), Image.ANTIALIAS)
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(relwidth=1, relheight=1)

captured_image_label = tk.Label(root)
captured_image_label.pack(pady=10)

result_label = tk.Label(root, text="Predicted Plant: None", font=("Helvetica", 16))
result_label.pack(pady=10)

def refresh():
    current_image = captured_image_label.image
    if current_image:
        refresh_prediction(model, plant_categories, result_label, captured_image_label)

capture_button = tk.Button(root, text="Capture Image", command=lambda: capture_and_predict(model, plant_categories, result_label, captured_image_label))
capture_button.pack(pady=10)

select_button = tk.Button(root, text="Select Image from File", command=lambda: select_and_predict(model, plant_categories, result_label, captured_image_label))
select_button.pack(pady=10)


root.mainloop()
