# Customer Churn Prediction Using Machine Learning

## Project Overview
This project uses machine learning to predict whether a telecom customer is likely to leave a service. It analyzes customer information, contract details, internet services, tenure, and billing information to identify potential churn.

## Problem Statement
Customer churn can affect a company's revenue. Identifying customers who may leave can help businesses plan customer-retention strategies.

## Objectives
- Clean and prepare customer data.
- Explore patterns in customer churn.
- Train and compare classification models.
- Evaluate models using multiple performance metrics.
- Build a web application for interactive predictions.

## Dataset
The project uses the Telco Customer Churn dataset. It contains customer demographics, subscribed services, contract information, billing details, and the churn target.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

## Workflow
1. Load and inspect the dataset.
2. Handle missing values and clean the data.
3. Perform exploratory data analysis.
4. Separate features and target.
5. Encode categorical features and scale numerical features.
6. Split the data into training and testing sets.
7. Train and compare machine learning models.
8. Tune Logistic Regression using GridSearchCV.
9. Save the trained pipeline.
10. Build a Streamlit application for predictions.

## Models Compared
- Logistic Regression
- Decision Tree
- Random Forest

## Initial Model Results
On the project's current test split, the initial models achieved:

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.4% | 64.8% | 57.2% | 60.8% | 0.836 |
| Decision Tree | 73.3% | 49.9% | 79.4% | 61.3% | 0.828 |
| Random Forest | 73.9% | 50.6% | 76.2% | 60.8% | 0.830 |

These are results from one train/test split and should not be interpreted as guaranteed performance on future customers.

## Run the Application

1. Install Python and the required dependencies.
2. Clone or download this repository.
3. Install the dependencies:

   `pip install -r requirements.txt`

4. Ensure the trained model is available at `models/customer_churn_model.pkl`.
5. Run the application from the project root:

   `python -m streamlit run app/app.py`

## Limitations
- Predictions depend on the dataset and model quality.
- Test-set performance may differ from performance on future customers.
- A churn prediction is an estimate, not a guarantee.
- Further validation and tuning may improve performance.

## Future Improvements
- Compare additional classification algorithms.
- Improve recall while monitoring false positives.
- Add model interpretation and feature importance.
- Deploy the application online.

## Author
BTech Computer Science and Engineering (AI & ML) graduate.