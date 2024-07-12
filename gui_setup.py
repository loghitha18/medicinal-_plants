import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

root = tk.Tk()
root.title("Leaf Identification")

captured_image_label = tk.Label(root)
captured_image_label.pack(pady=10)

result_label = tk.Label(root, text="Predicted Plant: None", font=("Helvetica", 16))
result_label.pack(pady=10)

root.mainloop()
