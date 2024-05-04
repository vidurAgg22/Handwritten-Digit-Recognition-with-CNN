# Handwritten Digit Recognition with CNN
This repository contains a simple Convolutional Neural Network (CNN) model trained on the MNIST dataset for image classification. The model is implemented using PyTorch and trained to classify handwritten digits from 0 to 9.<br>

https://github.com/vidurAgg22/Handwritten-Digit-Recognition-with-CNN/assets/165144144/b538735e-7d9b-4d8a-9e2c-76823454b1d5
## Training the Model
The CNN model is trained using the MNIST dataset, which consists of 60,000 training images and 10,000 testing images of handwritten digits. The training process involves the following steps:
1. Data Preprocessing: The MNIST images are preprocessed, converting them to grayscale and normalizing the pixel values to a range of [-1, 1].
2. Model Architecture: The CNN model architecture consists of two convolutional layers followed by max-pooling layers, and two fully connected layers. ReLU activation functions are used after each convolutional layer to introduce non-linearity.
3. Loss Function and Optimizer: The model is trained using the Cross Entropy Loss function, which is suitable for multi-class classification tasks. The Adam optimizer is used to minimize the loss function during training.
4. Training Loop: The model is trained over multiple epochs, with the training dataset divided into mini-batches. In each epoch, the model predicts the output for each batch, computes the loss, and updates the model parameters using backpropagation.
5. Model Evaluation: After training, the model's performance is evaluated using the test dataset to measure its accuracy in classifying handwritten digits.

## File Description
code.ipynb: Jupyter notebook containing the code for training the CNN model and implementing the image classifier.<br>
ocrcnnmodel.pth: Pre-trained model weights saved in PyTorch format.

## Requirements
Python 3.x<br>
PyTorch<br>
torchvision<br>
ipywidgets<br>

## License
This project is licensed under the MIT License.
