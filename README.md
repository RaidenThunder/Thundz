**Cat vs. Dog Image Classification using Deep Neural Networks**
This repository contains a Jupyter Notebook that explores the use of various Deep Neural Network (DNN) architectures to classify images of cats and dogs. The project utilizes the cats_vs_dogs dataset from TensorFlow Datasets and compares five different models to identify the most effective one for this binary classification task.

**About the Project**
The core of this project is a Jupyter Notebook (cats_vs_dogs_classification.ipynb) that walks through the entire process of building and evaluating image classification models.

**The key steps covered are:**

**Data Loading & Pre-processing:** The cats_vs_dogs dataset is loaded and split into training (80%), validation (10%), and testing (10%) sets. Images are resized to 150x150 pixels and pixel values are normalized to the range [-1, 1] for model compatibility.
**Model Building:** Five distinct DNN models are constructed to compare their performance.
**Training & Evaluation:** Each model is trained for 10 epochs, and their training/validation accuracy and loss are plotted to visualize performance.
**Testing:** The best-performing model is evaluated on the unseen test dataset, with a detailed classification report and confusion matrix.  

**Getting Started**
To get a local copy up and running, follow these simple steps.
Prerequisites
Ensure you have Python 3 installed, along with pip for package management. This project relies on several Python libraries.
Installation
Clone the repo
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
Install required packages
pip install tensorflow tensorflow-datasets matplotlib numpy scikit-learn
USAGE
Open the Jupyter Notebook:
jupyter notebook cats_vs_dogs_classification.ipynb
Run the cells sequentially from top to bottom to see the entire process, from data loading to model evaluation.

**Models Explored**
The notebook implements and compares five different fully connected neural network architectures:
Basic DNN: A simple baseline model with a single hidden dense layer.
Deeper DNN: A model with three hidden dense layers to explore the effect of network depth.
DNN with Dropout: Introduces dropout layers to the deeper model to combat overfitting.
DNN with Batch Normalization: Incorporates batch normalization layers to improve training stability and performance.
DNN with L2 Regularization: Adds L2 regularization to penalize large weights and reduce overfitting.

**Results**
After training all five models, their performance was compared. The CNN with Batch Normalization (Model 4) was identified as the most effective model, demonstrating stable training and high accuracy on the validation set.
The final evaluation on the test set yielded a detailed classification report and a confusion matrix, providing insights into the model's predictive capabilities for both "Cat" and "Dog" classes.

**Conclusion**
This project successfully demonstrates the process of building and comparing multiple DNN architectures for a common image classification problem. It highlights that techniques like Batch Normalization can significantly improve model stability and performance.
Future work could involve exploring more advanced architectures (like actual Convolutional Neural Networks with Conv2D layers), fine-tuning hyperparameters, or employing data augmentation techniques to further enhance accuracy.
