
# 🧠 Brain Tumor Detection (MRI Classification)

![GitHub stars](https://img.shields.io/github/stars/Sivaprasad-creator/Brain-Tumor-Detection)
![GitHub forks](https://img.shields.io/github/forks/Sivaprasad-creator/Brain-Tumor-Detection)
![GitHub license](https://img.shields.io/github/license/Sivaprasad-creator/Brain-Tumor-Detection)

## 📌 Project Overview

This project builds an **automatic Brain Tumor Detection system** using **MRI scan images**. It classifies scans into **three tumor types** — **Pituitary, Meningioma, Glioma** — and also detects if **no tumor** is present. The solution uses **Deep Learning (CNN & Transfer Learning)** and is deployed as a **Streamlit web app** for user-friendly predictions.

> 🔧 **Tools Used:** Python, Deep Learning (CNN, VGG16), Streamlit

---

## 📁 Dataset Information

- **Source:** Public dataset from Kaggle  
- **Data Type:** JPG MRI scan images  
- **Classes:** Pituitary, Meningioma, Glioma, No Tumor  
- **Key Features:** High-resolution MRI images, organized in labeled folders for training and testing

---

## 🎯 Objectives

- Predict whether an MRI image shows a brain tumor or not  
- Classify the type of tumor if detected  
- Visualize sample images and predictions  
- Deploy as a simple web application for real-time use

---

## 📊 Analysis Summary

- **Data Augmentation:** Random rotations, zoom, flips, and shifts applied for robust training  
- **Class Balance:** Checked distributions to handle any imbalance  
- **Visualization:** Displayed sample images from each category for verification  
- **Performance:** Evaluated models with confusion matrix and accuracy/loss curves

---

## 🤖 Models Used

| Step                  | Libraries & Techniques                                                                                             |
|-----------------------|--------------------------------------------------------------------------------------------------------------------|
| **Preprocessing**     | `os`, `numpy`, `seaborn`, `matplotlib`, `warnings`                                                                |
| **Data Handling**     | `ImageDataGenerator` for augmentation, `load_img`, `img_to_array` for image processing                            |
| **Custom CNN Model**  | `Sequential` model with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `BatchNormalization`             |
| **Transfer Learning** | Pre-trained **VGG16** model with `GlobalAveragePooling2D`, additional `Dense` layers, and fine-tuning             |
| **Training**          | `EarlyStopping`, `ModelCheckpoint`, `ReduceLROnPlateau` for optimized training and generalization                 |
| **Evaluation**        | `confusion_matrix` for performance analysis, plots with Seaborn and Matplotlib                                    |
| **Deployment**        | **Streamlit** web app for real-time image upload and prediction                                                    |

---

## 🚀 Streamlit Deployment

This project includes a **Streamlit web app** that allows you to:

- Upload an MRI scan image  
- Instantly predict the presence and type of tumor  
- Get easy-to-understand output on the same page  

---

## 🛠️ How to Run Locally

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/Sivaprasad-creator/Brain-Tumor-Detection.git
cd Brain-Tumor-Detection
```

2️⃣ **Create and activate a virtual environment (optional but recommended)**  
- On Windows:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

3️⃣ **Install dependencies**
```bash
pip install -r requirements.txt
```

4️⃣ **Run the Streamlit app**
```bash
streamlit run app.py
```

---

## 📬 Author Info

**Sivaprasad T.R**  
📧 Email: sivaprasadtrwork@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sivaprasad-t-r)  
💻 [GitHub](https://github.com/Sivaprasad-creator)

---

## 📜 Data Source

Data sourced from: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

---

## 🙏 Acknowledgements

Special thanks to the Kaggle community and the original dataset creators for making this work possible.

---

## 💬 Feedback

Feel free to reach out for questions, suggestions, or collaborations!
