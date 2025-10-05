# Machine Learning & Deep Learning

### Linear Regression with sklearn

This project focuses on predicting car prices using linear regression based on real-world car sales data. The model was trained on a dataset containing car prices and mileage (distance driven). After training, predictions were compared against actual test data to evaluate performance.

The results show that the model predicts mid-range car prices with reasonable accuracy. However, it performs less accurately for cheaper cars and some high-end vehicles. This is likely because price variations in these ranges depend on additional factors such as brand, condition, manufacturing year, and market demand, which are not included in the dataset.

Through visualization of predicted vs. actual prices, the model’s strengths and limitations were analyzed, providing insights for potential improvements using more complex models or additional data features.

### Logistic Regression – University Admission Prediction

Built a logistic regression model to predict university admission based on SAT scores and gender. The model used dummy variable encoding and maximum likelihood estimation with statsmodels.Logit(). Results showed that SAT had a strong positive effect on admission probability, while gender also had a significant impact. Model performance was evaluated using a confusion matrix and accuracy score.

### 🧠 K-means

This project applies K-Means clustering to analyze patterns within multidimensional data and identify meaningful groups based on feature similarity. The optimal number of clusters is determined using the Elbow Method, which evaluates the Within-Cluster Sum of Squares (WCSS) to balance model simplicity and cluster compactness.

The clustering results are visualized through scatter plots to illustrate data distribution and heatmaps to highlight correlations and relationships between features. These visualizations provide valuable insights into the underlying structure of the dataset, helping reveal hidden trends, group behaviors, and feature interactions.

### TensorFlow

Use TensorFlow to learn from training data and make predictions. The model consists of a single Dense layer that automatically initializes and optimizes weights and biases using the Stochastic Gradient Descent (SGD) optimizer. The training process, executed through model.fit(), adjusts parameters to minimize the Mean Squared Error (MSE) loss between predicted and target values. After training, model.predict_on_batch() is used to generate predictions based on the optimized model parameters.

### MNIST

MNIST handwritten digit recognition dataset, where a neural network model is built and trained to automatically classify images of digits. The model uses Softmax in the output layer to convert results into probability distributions across 10 classes and applies One-hot encoding for the true labels. By using cross-entropy loss, the model measures the difference between predictions and actual values to optimize its performance. This project demonstrates the fundamental process and core principles of neural networks in image classification tasks.

### 🎧 Audiobooks Re-Purchase Prediction

A neural network model built with TensorFlow/Keras to predict whether audiobook customers will make another purchase.  
The dataset was balanced to address class imbalance, standardized, and split into training, validation, and test sets.  

The model achieved **83.7% test accuracy**, showing strong generalization and effective performance in identifying potential returning customers.
