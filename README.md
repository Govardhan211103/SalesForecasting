# Walmart Sales Forecasting Project

This repository contains a complete Walmart sales forecasting project using modular programming and machine learning techniques. The project aims to predict future sales for different products at Walmart stores.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Prediction interface](#prediction-interface)

## Introduction

The goal of this project is to forecast sales for various products at Walmart stores. The prediction is based on historical sales data and other relevant features. The project uses modular programming principles to ensure code reusability and maintainability. Additionally, machine learning techniques are employed, specifically the CatBoost regressor, to develop an accurate prediction model.

## Project Structure

The project follows a modular structure to improve code organization and readability. Here is an overview of the main directories and files:

- `artifacts/`: This directory contains the datasets used for training and testing the model, preprocessing and model pickle files.
- `notebooks/`: This directory contains Jupyter notebooks used for exploratory data analysis, feature engineering, and model evaluation along with original data retrieved from Kaggle.
- `src/`: This directory contains the main Python scripts.
<br>        `src/components/`: This directory contains scripts for training the model, transforming the data, and evaluating the model's performance.
<br>        `src/pipeline/`: This directory contains the modular pipeline components used for data preprocessing, feature engineering, model training, and evaluation.      
- `templates/`: This directory contains the interface files(HTML) for user interaction.

## Dependencies

The following dependencies are required to run the project:

- Python 3.7 or higher
- CatBoost 0.26.1 or higher
- Pandas 1.1.5 or higher
- NumPy 1.19.5 or higher
- Scikit-learn 0.24.2 or higher
- Matplotlib 3.3.4 or higher
- Jupyter Notebook (optional, for running the notebooks)

It is recommended to use a virtual environment to manage the dependencies.

## Installation

To install the necessary dependencies, follow these steps:

1. Clone this repository: `git clone https://github.com/Govardhan211103/SalesForecasting.git`
2. Navigate to the project directory: `cd SalesForecasting`
3. Create a virtual environment (optional but recommended): `conda create -p venv python==3.9`
4. Activate the virtual environment: `conda activate venv`
5. Install the dependencies: `pip install -r requirements.txt`

## Usage

To train the sales forecasting model and generate predictions, follow these steps:

1. Ensure that you have installed the required dependencies (see the [Installation](#installation) section).
2. Place the relevant dataset files in the `notebook/data/` directory.
3. Execute the `data_ingestion.py` script to train the CatBoost regression model: `python src/components/data_ingestion.py`
4. After training, To use the web application for predicting new data, execute the following: `python app.py`
5. Optionally, use the provided Jupyter notebooks in the `notebooks/` directory for exploratory data analysis and model evaluation on different models.

## Contributing

Contributions to this project are welcome. If you encounter any issues or have suggestions for improvement, please open an issue or submit a pull request. 

## Prediction interface

![SalesForecast](https://github.com/Govardhan211103/SalesForecasting/assets/112187319/c91818c6-62d5-489b-8ea8-29629e2310a6)

