![banner](https://github.com/PJURC-data-science/spaceship-titanic/blob/main/media/banner.png)

# Spaceship Titanic: Dimensional Transport Prediction
[View Notebook](https://github.com/PJURC-data-science/spaceship-titanic/blob/main/Spaceship%20Titanic.ipynb)

A machine learning analysis to predict passenger transportation to alternate dimensions in the Spaceship Titanic incident. This study compares multiple ML models and ensemble methods to achieve optimal prediction accuracy in a Kaggle competition.

## Overview

### Business Question 
Can we predict which Spaceship Titanic passengers were transported to an alternate dimension using available passenger data?

### Key Findings
- CryoSleep status significantly impacts transport
- Gradient boosting models show best performance
- XGBoost achieves 80.43% competition accuracy
- Ensemble methods show promise but don't outperform XGBoost
- Missing data impacts ~16% of records

### Impact/Results
- Achieved 80.43% prediction accuracy
- Compared 6 ML models
- Tested 3 ensemble methods
- Handled missing data
- Created feature engineering pipeline

## Data

### Source Information
- Dataset: Spaceship Titanic
- Source: Kaggle Competition
- Size: ~9000 training records
- Balance: Well-balanced classes
- Missing Data: 16% of records

### Variables Analyzed
- Passenger demographics
- Travel arrangements
- Luxury amenity usage
- Group associations
- Cabin information
- Service selections
- Transport status

## Methods

### Analysis Approach
1. Data Preprocessing
   - Missing value imputation
   - Feature engineering
   - Data transformation
2. Model Development
   - Individual models
   - Ensemble methods
   - Performance comparison
3. Competition Submission
   - Model selection
   - Prediction generation
   - Score validation

### Tools Used
- Python (Data Science)
  - XGBoost: Primary model
  - LightGBM: Comparison model
  - Scikit-learn:
    - Voting Classifier
    - Stacking Classifier
    - Base models
    - Preprocessing
  - Model Performance:
    - Competition score: 80.43%
    - Cross-validation
    - Ensemble testing
  - Additional Tools:
    - Feature engineering
    - Data imputation
    - Weighted averaging

## Getting Started

### Prerequisites
```python
ipython==8.12.3
lightgbm==4.5.0
matplotlib==3.8.4
numpy==2.2.0
pandas==2.2.3
phik==0.12.4
scikit_learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
xgboost==2.1.3
```

### Installation & Usage
```bash
git clone git@github.com:PJURC-data-science/spaceship-titanic.git
cd spaceship-titanic
pip install -r requirements.txt
jupyter notebook "Spaceship Titanic.ipynb"
```

## Project Structure
```
spaceship-titanic/
│   README.md
│   requirements.txt
│   Spaceship Titanic.ipynb
|   utils.py
|   styles.css
└── data/
    └── test.csv
    └── train.csv
└── exports/
    └── submission_LightGBM.csv
    └── submission_StackingClassifier.csv
    └── submission_VotingClassifier.csv
    └── submission_WeightedEnsemble.csv
    └── submission_XGBoost.csv
```

## Strategic Recommendations
1. **Model Selection**
   - Use XGBoost for best accuracy
   - Consider ensemble for robustness

2. **Feature Engineering**
   - Focus on CryoSleep data
   - Try other methods of handling missing values
   - Create group features
   - Transform amenity data
   - Extensively check for missing data patterns

3. **Implementation Strategy**
   - Optimize preprocessing
   - Balance complexity/performance
   - Maintain model interpretability
   - Consider computational costs

## Future Improvements
- Test AdaBoost implementation
- Implement Bayesian search
- Try Optuna optimization
- Enhance feature engineering
- Improve missing data handling
- Test other VotingClassifier optimization methods (e.g., optimal weights for specific performance metric)