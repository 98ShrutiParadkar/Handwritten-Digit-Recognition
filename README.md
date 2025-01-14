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
- 
![Sample](https://github.com/user-attachments/assets/c9e496d9-d1bb-446b-b6f0-d151ceff7ba3)



## 🧠 Model Architecture

The neural network used in this project has the following architecture:
1. **Input Layer**: Flattens the 28x28 image into a 1D array.
2. **Hidden Layer 1**: Fully connected layer with 128 neurons and ReLU activation.
3. **Hidden Layer 2**: Fully connected layer with 64 neurons and ReLU activation.
4. **Output Layer**: Fully connected layer with 10 neurons and softmax activation for multi-class classification.

### **Model Diagram:**
![1](https://github.com/user-attachments/assets/2a0828ce-4b96-469f-8842-c98e81b5dc11)


## 🚀 How to Run the Project
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```
## **Accuracy Plot:**
![accuracy_plot](https://github.com/user-attachments/assets/6f27fa51-d0d1-4f52-81fc-1759fc02d70e)

## **Output:**
![predicted](https://github.com/user-attachments/assets/59d08622-7754-416c-b1f6-7c82ee5fc44d)

## 📌 Future Scope
- Implement a Convolutional Neural Network (CNN) for improved accuracy.
- Extend the model for recognizing digits in complex images or handwritten notes.
- Build a web interface to deploy the model as an interactive application.

## 🙌 Acknowledgements
- MNIST Dataset: Provided by Yann LeCun et al.
- Tutorials and documentation from the TensorFlow community.
