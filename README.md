# Titanic Machine Learning Project

This project is a submission for the Titanic Machine Learning competition on Kaggle. The goal of the competition is to predict which passengers survived the Titanic shipwreck based on a set of features.

## Project Overview

In this project, I:
- Performed data preprocessing and exploratory data analysis (EDA).
- Applied feature engineering techniques to improve the dataset.
- Trained and evaluated machine learning models.
- Achieved an accuracy score of **78%** on the test dataset.

## Dataset

The dataset consists of the following files:
- `train.csv`: Training data with survival labels.
- `test.csv`: Testing data without survival labels (used for predictions).

Key features in the dataset include:
- `Pclass`: Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd).
- `Sex`: Gender of the passenger.
- `Age`: Age of the passenger.
- `SibSp`: Number of siblings/spouses aboard.
- `Parch`: Number of parents/children aboard.
- `Fare`: Fare paid for the ticket.
- `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## Methods and Models

### Data Preprocessing
- Handled missing values (e.g., imputing `Age` and `Embarked`).
- Encoded categorical variables (e.g., `Sex` and `Embarked`).

### Feature Engineering
- Created new features (e.g., `FamilySize` combining `SibSp` and `Parch`).
- Extracted titles from passenger names for additional insights.

### Machine Learning Models
The following models were trained and evaluated:
- Logistic Regression
- Random Forest
- Gradient Boosting
- Bagging
- AdaBoostingClassifier
- SVC
-  XGBClassifier

The final model achieved an accuracy score of **78%** on the test dataset.

## Repository Structure

```
├── titanic-ml.ipynb         # Jupyter Notebook with code and analysis
├── train.csv               # Training dataset
├── test.csv                # Test dataset
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Shrouk-Adel/titanic-ml-project.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
## Results
- Achieved a test accuracy score of **78%**.
 
