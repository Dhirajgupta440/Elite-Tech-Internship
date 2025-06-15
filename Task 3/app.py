# app.py - Gradio Deployment Script for CIFAR-10 CNN Model

import gradio as gr
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# Load the trained model
model = load_model("cnn_cifar10_model.h5")

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Function to classify an input image
def classify_image(img):
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    img = img.reshape(1, 32, 32, 3)
    prediction = model.predict(img)[0]
    return {class_names[i]: float(prediction[i]) for i in range(10)}

# Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Label(num_top_classes=3),
    title="CIFAR-10 Image Classifier",
    description="Upload a 32x32 color image to classify it into one of the 10 CIFAR-10 categories."
)

# Launch the app
if __name__ == "__main__":
    interface.launch()
