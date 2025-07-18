DNN: Cats vs. Dogs Image Classification
This project uses a Deep Neural Network (DNN) to classify images from the cats_vs_dogs dataset, demonstrating skills in deep learning, model optimization, and performance evaluation. These techniques are applicable to workforce analytics tasks, such as classifying employee performance data or sentiment from visual feedback.
Files

DNN_Assignment_1_Group_112_1_(1).ipynb: Jupyter Notebook with the DNN implementation, including data preprocessing, model architectures, regularization, optimizer comparisons, and visualizations.
No separate dataset file is required as the cats_vs_dogs dataset is loaded via TensorFlow Datasets.

Project Overview
The notebook performs the following tasks:

Data Acquisition: Loads the cats_vs_dogs dataset using TensorFlow Datasets, split into 80% training, 10% validation, and 10% testing.
Data Preprocessing: Resizes images and converts them to a format suitable for DNN input.
Model Architectures:
Model 1 (Original): Baseline DNN with multiple layers.
Model 2 (Decreased Layers): Simplified architecture for comparison.
Model 3 (Increased Layers): Deeper architecture to test complexity.


Regularization:
Model 4: Adds Dropout (0.25) to reduce overfitting.
Model 5: Combines Dropout (0.25) with L2 regularization.


Optimizers:
Model 1: SGD (baseline).
Model 6: RMSprop.
Model 7: Adam.


Visualizations:
Sample images from the dataset.
Training/validation accuracy and loss curves.
Confusion matrix and classification report for model evaluation.



Key Insights

Best Model: Model 4 (Original Architecture with Dropout 0.25 and SGD optimizer) achieved the highest validation accuracy, indicating effective regularization.
Architecture Impact: Simplified architecture (Model 2) outperformed the deeper Model 3, suggesting overfitting with increased complexity.
Regularization: Dropout (0.25) improved stability, but combining with L2 regularization (Model 5) increased loss, indicating suboptimal L2 strength.
Optimizers: SGD outperformed RMSprop and Adam, which showed fluctuating validation accuracy.
Applications: Techniques can be applied to workforce management, such as classifying employee performance metrics or visual feedback for sentiment analysis.

How to Run

Environment: Open the notebook in Google Colab (recommended due to GPU support) or Jupyter Notebook with a GPU.
Install Dependencies:pip install tensorflow tensorflow-datasets matplotlib numpy scikit-learn


Run the Notebook:
Ensure an internet connection for TensorFlow Datasets to download cats_vs_dogs.
Run all cells sequentially. The dataset is automatically loaded via tfds.load('cats_vs_dogs').


GPU Setup (Optional):
In Colab: Runtime > Change runtime type > GPU.
Locally: Ensure TensorFlow-GPU is installed and a compatible GPU is available.



Dependencies
See ../requirements.txt for a complete list. Key packages:

Python 3.8+
tensorflow, tensorflow-datasets
matplotlib, numpy, scikit-learn

Notes

The notebook assumes access to TensorFlow Datasets. No external dataset file is needed.
Visualizations (e.g., accuracy/loss curves, confusion matrix) are optimized for clarity and recruiter appeal.
For large-scale applications, consider using a CNN (e.g., in the CNN folder) for better image classification performance.
