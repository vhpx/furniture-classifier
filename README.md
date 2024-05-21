# RMIT Machine Learning (COSC2753) Assignment 2

## Introduction

This project tackles the development of an image-based furniture recommendation system for an e-commerce platform. It was created by:

- Vo Hoang Phuc (s3926761)
- Nguyen Cong Gia Hien (s38884308)
- Phan Trong Nguyen (s3927189)
- Ho Quoc Thai (s3927025)

The system is designed to:

1.  **Classify:** Categorize furniture images into six classes (beds, chairs, dressers, lamps, sofas, tables).
2.  **Recommend Similar Items:** Suggest 10 furniture items visually similar to a user-provided image.
3.  **Recommend by Style:**  Extend recommendations to include items matching the interior style of the input image.

The project leverages Python, TensorFlow (version < 2.11, mainly to train our models on native Windows 11 machines), and various machine learning libraries to build and evaluate multiple models for these tasks.

## Key Features

- **Image Classification:** Employs Convolutional Neural Networks (CNNs), including custom and ResNet architectures, to classify furniture categories.
- **Similarity Search:** Explores K-means clustering for finding visually similar items.
- **Style Recognition:** Extends CNN models and ResNet architectures to recognize interior styles.
- **Data Augmentation:** Utilizes image augmentation techniques to address class imbalances and improve model generalization.
- **Caching:** Employs caching mechanisms to store intermediate results and avoid redundant computations.

## Repository Structure

```
Assignment 2
├── notebooks             # Jupyter notebooks for different stages of analysis
├── cache                 # Cached data for faster execution
├── data
│   ├── datasets          # Original and processed image datasets
│   └── zipped            # Zipped util functions for Google Colab Deployment
├── cache                 # Cached data for faster execution
│   └── models            # Trained models for classification and recommendation
├── utils                 # Utility functions for data preprocessing, model training, etc.
├── README.md             # This file
├── pipeline.ipynb        # Main pipeline for training and evaluating models
└── requirements.txt      # Required libraries for the project
```

## Getting Started

### Prerequisites

- **Python:** Version 3.10.14 (recommend using Anaconda for easy package management)
- **Libraries:** Install required libraries using `pip`:

  ```bash
  pip install -r requirements.txt
  ```

  **Required Libraries:**
  - pandas
  - numpy
  - matplotlib
  - tensorflow<2.11
  - tensorflow_hub
  - seaborn
  - scikit-learn
  - setuptools
  - joblib
  - IPython
  - imagehash
  - ipywidgets
  - tqdm

## Usage

1.  **Data Preparation:**
    *   Place your raw dataset in the `data/datasets/raw` directory.
    *   Run the preprocessing scripts in the `pipeline` to clean, augment, and split the data.

2.  **Model Training:**
    *   Execute the Jupyter notebooks in the `pipeline` or `notebooks` directory to train and evaluate different models for each task.
    *   Trained models will be saved in the `cache/models` directory.

3.  **Recommendations:**
    *   Use the trained models to generate furniture recommendations based on input images.
    *   Refer to the notebooks for examples and guidance.

## Disclaimer

This project is developed for educational purposes. The code is not optimized for production use. The primary goal is to demonstrate the understanding and application of machine learning concepts and techniques for furniture recommendation.

## Acknowledgements

We would like to thank Dr. Nguyen Thien Bao for providing the dataset and guidance for this assignment. We also acknowledge the open-source libraries and resources that were instrumental in the development of this project.