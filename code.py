import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import gradio as gr

# Ensure the script uses UTF-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Define the path to your dataset
dataset_path = 'D:/CNN_implementation/archive (3)/COVID-19_Radiography_Dataset/'

# Safe print function to handle encoding issues
def safe_print(*args, **kwargs):
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        print(*[arg.encode('utf-8', 'ignore').decode('utf-8') for arg in args], **kwargs)

# Check the number of images in COVID and Normal folders
safe_print(len(os.listdir(os.path.join(dataset_path, 'COVID/images'))))
safe_print(len(os.listdir(os.path.join(dataset_path, 'Normal/images'))))

# Load a sample image
img = cv2.imread(os.path.join(dataset_path, 'COVID/images/COVID-32.png'))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()
safe_print(img.shape)

# Load metadata
df = pd.read_excel(os.path.join(dataset_path, 'COVID.metadata.xlsx'))
safe_print(df.head())

def loadImages(path, urls, target):
    images = []
    labels = []
    for i in range(len(urls)):
        img_path = os.path.join(path, urls[i])
        img = cv2.imread(img_path)
        img = img / 255.0
        img = cv2.resize(img, (50, 50))  # Reduce image size
        images.append(img)
        labels.append(target)
    images = np.asarray(images, dtype=np.float32)  # Use float32 to save memory
    return images, labels

covid_path = os.path.join(dataset_path, 'COVID/images/')
covidUrl = os.listdir(covid_path)
covidImages, covidTargets = loadImages(covid_path, covidUrl, 1)

normal_path = os.path.join(dataset_path, 'Normal/images')
normal_urls = os.listdir(normal_path)
normalImages, normalTargets = loadImages(normal_path, normal_urls, 0)

covidImages = np.asarray(covidImages)
normalImages = np.asarray(normalImages)
data = np.r_[covidImages, normalImages]
targets = np.r_[covidTargets, normalTargets]

x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.25)

model = Sequential([
    tf.keras.Input(shape=(50, 50, 3)),
    Conv2D(32, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Conv2D(16, 3, activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

# Plot training history
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.legend()
    
    plt.show()
    fig.savefig('training_history.png')  # Save the plot for display in Gradio

plot_history(history)

def predict_covid(model, img_path):
    # Load the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (50, 50))  # Update size to match the training size
    img = img / 255.0  # Normalize
    
    # Reshape the image for model input (add batch dimension)
    img = np.reshape(img, (1, 50, 50, 3))
    
    # Predict
    prediction = model.predict(img)
    
    # Classify based on prediction
    if prediction > 0.5:
        return "COVID"
    else:
        return "Normal"

def overlay_prediction(image, prediction):
    # Convert image to RGB for displaying with prediction
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create overlay text
    overlay_text = f"Prediction: {prediction}"
    
    # Position for the text
    position = (10, 30)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0) if prediction == "COVID" else (0, 0, 255)
    thickness = 2
    
    # Add text overlay
    cv2.putText(image_rgb, overlay_text, position, font, font_scale, color, thickness, cv2.LINE_AA)
    
    return image_rgb

def classify_images_with_display(files):
    results = []
    images_display = []
    for file in files:
        image = cv2.imread(file.name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image, (50, 50))  # Update size to match the training size
        image_resized = image_resized / 255.0  # Normalize
        image_resized = np.reshape(image_resized, (1, 50, 50, 3))
        
        prediction = model.predict(image_resized)
        confidence = prediction[0][0]
        if confidence > 0.5:
            result = "COVID"
        else:
            result = "Normal"
        
        results.append((result, confidence))
        # Overlay prediction on the original image
        image_with_overlay = overlay_prediction(image, result)
        images_display.append(image_with_overlay)
    
    last_result = results[-1] if results else None
    return results, images_display, last_result

def display_training_history():
    return plt.imread('training_history.png')

# Custom CSS for styling
css = """
h1 {
    color: #4CAF50;
}
body {
    background-color: #f8f8f8;
    font-family: Arial, sans-serif;
}
input[type=file] {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 10px 20px;
    text-align: center;
    text-decoration: none;
    display: inline-block;
    font-size: 16px;
    margin: 4px 2px;
    cursor: pointer;
}
"""

gr_interface = gr.Blocks(css=css)

with gr_interface:
    gr.Markdown("# COVID-19 Radiography Classification")
    gr.Markdown("Upload chest X-ray images to classify if they are COVID-19 positive or Normal.")
    
    with gr.Tabs():
        with gr.TabItem("Classify Image(s)"):
            images_input = gr.Files(label="Upload Images")
            results_output = gr.Dataframe(headers=["Prediction", "Confidence Score"], label="Results")
            images_display = gr.Gallery(label="Uploaded Images with Predictions")
            last_result_output = gr.Textbox(label="Last Classification Result")
            classify_button = gr.Button("Classify")
            
            classify_button.click(fn=classify_images_with_display, 
                                  inputs=images_input, 
                                  outputs=[results_output, images_display, last_result_output])
        
        with gr.TabItem("Training History"):
            gr.Markdown("### Training Accuracy and Loss")
            training_history_output = gr.Image(value='training_history.png', interactive=False)
    
    gr.Markdown("### Instructions")
    gr.Markdown("""
    1. Upload chest X-ray images.
    2. Click 'Classify' to see the predictions and confidence scores.
    3. View training accuracy and loss in the 'Training History' tab.
    """)

gr_interface.launch()