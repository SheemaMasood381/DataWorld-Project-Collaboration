# ⚽ Soccer League Database (EDA) Match Outcome Prediction

This project is part of the DataWorld Project Collaboration. Despite never watching a football match in my life, this project gave me a deep understanding of the game, including key stats like possession, shots on target, and team performance metrics.

## 📝 Project Overview

### 📊 Dataset Details

- **Dataset Name:** Soccer League Database
- **Source:** [Kaggle - Soccer Database](https://www.kaggle.com/hugomathien/soccer)
- **Description:** The dataset contains detailed information on over 25,000 soccer matches from the top European leagues. It includes data on match outcomes, team statistics, player statistics, and more.

### 🎯 The Challenge

Predicting match outcomes using team data (win rates, goals, etc.) and match-specific features (shots, possession) to determine whether the home or away team would win.

### 🔑 Key Steps Taken

- **Exploratory Data Analysis (EDA):** Analyzed patterns and trends in match data.
- **Feature Engineering:** Selected key features like team stats and match-specific data.
- **Data Merging & Extraction:** Linked team data to match outcomes using API IDs.
- **Model Training & Evaluation:** Trained models (Logistic Regression, Random Forest, XGBoost, SVM) and evaluated using accuracy, precision, recall, and confusion matrices.

### 🛠️ Challenges Faced

- Data extraction and feature selection were tricky, requiring me to identify the most relevant columns for predictions.
- Merging team data required precision to ensure accurate and effective data alignment.
- Handling missing data and outliers was critical for accurate modeling.
- Hyperparameter tuning was essential to choose the best model.

### 🚀 Next Steps

I plan to enhance the model with advanced features, explore other algorithms, and even build an interactive web app for anyone to predict match outcomes!

## 📂 Project Structure

- `03_Soccer_League_DAtaBase_(EDA)_Match_Prediction_Model.ipynb`: Notebook for data cleaning, exploratory data analysis, and building the match prediction model.

## ⚙️ Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
