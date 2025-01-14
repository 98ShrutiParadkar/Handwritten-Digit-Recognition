# Handwritten Digit Recognition using Neural Networks

![Project Banner](https://via.placeholder.com/800x200.png?text=Handwritten+Digit+Recognition+Project)  
*Classifying handwritten digits (0-9) using the MNIST dataset with a neural network.*



## 📋 Project Description

This project implements a neural network to recognize handwritten digits using the MNIST dataset. The dataset consists of grayscale images of digits (28x28 pixels) and corresponding labels. The goal is to preprocess the data, train a model, and achieve high accuracy in digit classification.

### **Key Features:**
- Uses the popular **MNIST dataset**.
- Built with **TensorFlow/Keras** for deep learning.
- Achieves strong performance with a simple dense neural network.
- Provides insights into machine learning workflows: data preprocessing, model training, and evaluation.



## 🛠️ Technologies Used
- **Python**: Programming language.
- **TensorFlow/Keras**: Deep learning framework.
- **Matplotlib**: Data visualization.
- **NumPy**: Numerical computations.



## 📊 Dataset Overview

The MNIST dataset is a collection of handwritten digits commonly used as a benchmark in machine learning. It includes:
- **60,000 training images** and **10,000 test images**.
- Each image is 28x28 pixels in grayscale.
- Labels range from 0 to 9, representing the digit in the image.

![mnist_samples](https://github.com/user-attachments/assets/200c8d3a-9316-4abf-b363-d651cc2622e7)



## 🧠 Model Architecture

The neural network used in this project has the following architecture:
1. **Input Layer**: Flattens the 28x28 image into a 1D array.
2. **Hidden Layer 1**: Fully connected layer with 128 neurons and ReLU activation.
3. **Hidden Layer 2**: Fully connected layer with 64 neurons and ReLU activation.
4. **Output Layer**: Fully connected layer with 10 neurons and softmax activation for multi-class classification.

### **Model Diagram:**
![model_architecture](https://github.com/user-attachments/assets/754490d2-887f-4300-86d4-51c16e83e167)

## 🚀 How to Run the Project
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```
## **Accuracy Plot:**
![accuracy_plot](https://github.com/user-attachments/assets/6f27fa51-d0d1-4f52-81fc-1759fc02d70e)

## **Output:**
![sample_predictions_grid](https://github.com/user-attachments/assets/4cc3db26-5f2e-4f74-88f4-10d613a05574)

## 📌 Future Scope
- Implement a Convolutional Neural Network (CNN) for improved accuracy.
- Extend the model for recognizing digits in complex images or handwritten notes.
- Build a web interface to deploy the model as an interactive application.

## 🙌 Acknowledgements
- MNIST Dataset: Provided by Yann LeCun et al.
- Tutorials and documentation from the TensorFlow community.
