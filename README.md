# House Price Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.8+-green)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-blue)
![Pandas](https://img.shields.io/badge/Library-Pandas-yellow)
![Matplotlib](https://img.shields.io/badge/Library-Matplotlib-lightblue)

## Project Overview

This project focuses on predicting house prices using a Linear Regression model. The dataset contains various features that influence the price of a house, such as square footage, the number of bedrooms, and the location. The goal is to build a model that can accurately predict house prices based on these features.

## Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Conclusion](#conclusion)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)
- [License](#license)

## Introduction

House prices are a significant economic factor and are influenced by various factors. This project aims to explore and model these factors to predict house prices. Linear Regression is a fundamental technique in predictive modeling and will be used to build the predictive model in this project.

## Dataset Description

we'll be using the Boston Housing dataset. There are 506 rows in the dataset. The target variable is the median home price. There are 13 predictor variables including an average number of rooms per dwelling, crime rate by town, etc. More information about this dataset can be found at  https://www.kaggle.com/datasets/muhammadyasirsaleem/boston-house-price/data

This data frame contains the following columns:

- **crim** Per capita crime rate by town

- **zn** proportion of residential land zoned for lots over 25,000 sq. ft.

- **indus** proportion of non-retail business acres per town.

- **chas** Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

- **nox** nitrogen oxide concentration (parts per 10 million).

- **rm** average number of rooms per dwelling.

- **age** proportion of owner-occupied units built before 1940.

- **dis** weighted mean of distances to five Boston employment centres.

- **rad** index of accessibility to radial highways.

- **tax** full-value property-tax rate per $10,000.

- **ptratio** pupil-teacher ratio by town.

- **black** 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

- **lstat** lower status of the population (per cent).

- **medv** median value of owner-occupied homes in $1000s.


## Data Preprocessing

Before modelling, the data was cleaned and preprocessed. Steps included:
- Handling missing values
- Encoding categorical variables
- Normalizing and scaling features

## Exploratory Data Analysis

In this section, we analyze the relationships between different features and house prices. Visualizations and statistical summaries are used to identify trends and correlations in the data.

## Model Building

A Linear Regression model was constructed using the Scikit-learn library. The model was trained on the processed dataset, with the target variable being the house prices.

## Model Evaluation

The model's performance was evaluated using metrics such as:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R-squared (R²)

These metrics provide insights into the accuracy and reliability of the model.

## Conclusion

The Linear Regression model provided a reasonable prediction of house prices, demonstrating the relationship between the features and the target variable. Future work could involve using more advanced algorithms or additional data for improved accuracy.

## How to Run

To run the project:
1. Clone this repository.
2. Install the required dependencies.
3. Open the Jupyter Notebook and run the cells to train and evaluate the model.

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
cd House-Price-Prediction
pip install -r requirements.txt
jupyter notebook
