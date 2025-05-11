# 🧠 ChestX AI Classifier

ChestX AI Classifier is a deep learning-based application that detects COVID-19 from chest X-ray images using a custom Convolutional Neural Network (CNN). It features real-time prediction and image visualization via a user-friendly Gradio interface.

## 🔬 Project Highlights

- 📊 Binary classification of COVID vs Normal chest X-ray images.
- 🧠 Custom CNN architecture built with TensorFlow and Keras.
- ⚡ Efficient image preprocessing with OpenCV and NumPy.
- 🌐 Gradio web interface for real-time image upload and diagnosis.
- 📈 Training history visualization for performance tracking.

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib
- Scikit-learn
- Gradio

---

## 📦 Installation

1. **Clone the repository**

2. **Install dependencies** :

    - tensorflow
    - opencv-python
    - matplotlib
    - pandas
    - scikit-learn
    - gradio

3.**📁 Dataset**

This project uses the COVID-19 Radiography Database. You can download it from Kaggle.

Ensure the folder structure is :

COVID-19_Radiography_Dataset/
├── COVID/
│   └── images/
│   └── COVID.metadata.xlsx
├── Normal/
│   └── images/

Update the dataset_path variable in code.py accordingly.


## 🚀 Running the App

After setting up, launch the Gradio app with :

- A browser window will open with an interface to:

- Upload chest X-ray images

- Classify them as COVID or Normal




## ⚠️ Disclaimer

This project is for educational and research purposes only. It is not approved for medical use and should not be used to make healthcare decisions.



- View confidence scores

- Inspect training history (accuracy and loss)


