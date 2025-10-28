# Iris Flower Classification using Neural Network (Keras)

## Overview
This project demonstrates how to build a **Neural Network classifier** to predict the species of Iris flowers (Setosa, Versicolor, Virginica) based on their sepal and petal measurements.  
The dataset used is the **Iris dataset**, one of the most famous datasets in machine learning.


## Dataset Information
The dataset contains **150 rows** and **6 columns**:

| Column Name     | Description |
|-----------------|--------------|
| Id              | Sample number |
| SepalLengthCm   | Length of the sepal in cm |
| SepalWidthCm    | Width of the sepal in cm |
| PetalLengthCm   | Length of the petal in cm |
| PetalWidthCm    | Width of the petal in cm |
| Species         | Type of Iris flower (`Iris-setosa`, `Iris-versicolor`, `Iris-virginica`) |

There are **50 samples** of each species, and **no missing or duplicate values**.


##  Data Preprocessing Steps
1. **Loaded the dataset** using Pandas.
2. **Checked for missing values, duplicates, and datatypes.**
3. **Visualized outliers** using boxplots.
4. **Checked data distribution and skewness** using histograms.
5. **Encoded target variable (`Species`)** using `LabelEncoder`.
6. **Standardized features** (`SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`) using `StandardScaler`.
7. **Split the data** into:
   - Training set: 80%
   - Testing set: 20%


## Model Architecture

A simple **Feedforward Neural Network (ANN)** built with **Keras Sequential API**:

| Layer | Type | Units | Activation | Input Dim |
|--------|------|--------|-------------|------------|
| 1 | Dense | 8 | ReLU | 4 |
| 2 | Dense | 8 | ReLU | — |
| 3 | Dense | 3 | Softmax | — |

- **ReLU (Rectified Linear Unit):** introduces non-linearity.
- **Softmax:** outputs probabilities for 3 classes.
