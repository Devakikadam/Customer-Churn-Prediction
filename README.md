# Customer Churn Prediction

## Project Overview
Customer Churn Prediction is a Machine Learning project that predicts whether a telecom customer is likely to leave a service. It helps businesses understand customer churn and improve customer retention.

## Objectives
- Analyze customer data and identify churn patterns.
- Clean and preprocess the dataset.
- Train and compare Machine Learning models.
- Predict churn for new customer details.
- Display results through an interactive Streamlit web application.

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset
This project uses the IBM Telco Customer Churn dataset.

The cleaned dataset contains **7,032 customer records** and 20 columns, including the target column `Churn`.

- `Yes`: Customer churned.
- `No`: Customer did not churn.

## Project Workflow
1. Data collection
2. Exploratory Data Analysis (EDA)
3. Data cleaning and preprocessing
4. Train-test split
5. Model training
6. Model evaluation
7. Model saving
8. Streamlit web application

## Machine Learning Models
Two classification algorithms were trained:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 80.38% |
| Random Forest | 76.05% |

Logistic Regression achieved higher accuracy on the current test split. Random Forest achieved higher recall for the churn class.

## Key Dataset Insights
- Total cleaned customer records: 7,032
- Customers who churned: 1,869
- Historical churn rate: approximately 26.6%

These figures describe the dataset, not predictions for individual customers.

## Project Structure
```text
Customer_Churn_Prediction/
├── app/
│   └── app.py
├── data/
│   ├── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   └── cleaned_customer_churn.csv
├── models/
│   ├── logistic_model.pkl
│   └── random_forest_model.pkl
├── notebooks/
│   ├── data_analysis.py
│   └── train_model.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR-USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment
```bash
python -m venv venv
```

Activate it on Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
python -m streamlit run app/app.py
```

The application should open in your browser.

## How to Use
1. Open the Streamlit application.
2. View customer churn statistics on the Dashboard.
3. Navigate to Predict Churn.
4. Enter customer details.
5. Click the prediction button to view the predicted churn class and estimated probability.

## Evaluation
The models were evaluated using accuracy, precision, recall, and F1-score.
