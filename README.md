# Fashion-MNIST-Classifiers
Design and implementation of machine learning classifiers for data classification using the Fashion MNIST dataset.

# Intro
This project focuses on the classification of the Fashion MNIST dataset, which consists of grayscale images of 10 different types of clothing. The dataset contains 60,000 training samples and 10,000 testing samples, with each image being 28x28 pixels in size, converted to vectors of 784 features.

# How it works
The assignment explores various classification models, both traditional and neural networks. Specifically, the project includes:

  # A. Vector Representation Models
  1. Max polling with a 4x4 window to reduce image dimensions.
  2. Nearest Neighbor Classifier with K=1, 3, or 5 and Euclidean distance.
  3. Decision Tree and Random Forest models with specific hyperparameters.
  4. SVM classifiers with different C values and kernels (linear, RBF).
  5. Feed-forward neural network with 3 hidden layers, trained using Adam optimizer and evaluated for accuracy.
  # B. Image Representation Models
  1. Convolutional Neural Networks (CNN) with 3 convolutional layers, max-pooling, and dropout, tested with different filter sizes.
  2. The CNN architecture ends with a fully connected layer of 100 neurons and a softmax output for classification.

Each model is trained and evaluated on both training and testing datasets, and their performance is analyzed through various metrics, including loss and accuracy.

# How to use
To run the models, simply open the provided Jupyter notebook in an appropriate environment, such as Google Colab or a local machine with Python installed.

  1. Install necessary libraries: The notebook uses libraries such as TensorFlow, Keras, and scikit-learn.
  2. Preprocess the data: Normalize and reshape the Fashion MNIST dataset as described.
  3. Run the models: Execute each cell in sequence to train and test the models on the dataset. Hyperparameters can be adjusted in the relevant sections.
  4. Evaluate results: The notebook will produce plots of loss and accuracy for both the training and testing sets for all models.

