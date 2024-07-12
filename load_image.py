import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Path to your dataset folder
dataset_path = r'C:\Users\Hariharan\OneDrive\Desktop\Loghitha\Medicinal plant dataset'

def load_image(filename):
    file_path = os.path.join(dataset_path, filename)
    img = load_img(file_path, color_mode="rgb", target_size=(100, 100))
    img = img_to_array(img)
    img = img.reshape(1, 100, 100, 3)
    img = img.astype('float32') / 255.0
    return img
