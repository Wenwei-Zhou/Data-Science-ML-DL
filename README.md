### TensorFlow

TensorFlow to learn from training data and make predictions. The model consists of a single Dense layer that automatically initializes and optimizes weights and biases using the Stochastic Gradient Descent (SGD) optimizer. The training process, executed through model.fit(), adjusts parameters to minimize the Mean Squared Error (MSE) loss between predicted and target values. After training, model.predict_on_batch() is used to generate predictions based on the optimized model parameters.

### MNIST

MNIST handwritten digit recognition dataset, where a neural network model is built and trained to automatically classify images of digits. The model uses Softmax in the output layer to convert results into probability distributions across 10 classes and applies One-hot encoding for the true labels. By using cross-entropy loss, the model measures the difference between predictions and actual values to optimize its performance. This project demonstrates the fundamental process and core principles of neural networks in image classification tasks.

### 🎧 Audiobooks Re-Purchase Prediction

A neural network model built with TensorFlow/Keras to predict whether audiobook customers will make another purchase.  
The dataset was balanced to address class imbalance, standardized, and split into training, validation, and test sets.  

The model achieved **83.7% test accuracy**, showing strong generalization and effective performance in identifying potential returning customers.
