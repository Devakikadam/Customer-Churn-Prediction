
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load cleaned dataset
df = pd.read_csv("data/cleaned_customer_churn.csv")

# Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"].map({"No": 0, "Yes": 1})

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "str"]).columns
numerical_cols = X.select_dtypes(exclude=["object", "str"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_cols),
        ("numerical", StandardScaler(), numerical_cols)
    ]
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Logistic Regression model
logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

logistic_model.fit(X_train, y_train)
logistic_predictions = logistic_model.predict(X_test)

print("Logistic Regression Accuracy:",
      round(accuracy_score(y_test, logistic_predictions) * 100, 2), "%")

print("\nLogistic Regression Report:")
print(classification_report(
    y_test, logistic_predictions, zero_division=0
))

# Random Forest model
random_forest_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    ))
])

random_forest_model.fit(X_train, y_train)
rf_predictions = random_forest_model.predict(X_test)

print("\nRandom Forest Accuracy:",
      round(accuracy_score(y_test, rf_predictions) * 100, 2), "%")

print("\nRandom Forest Report:")
print(classification_report(
    y_test, rf_predictions, zero_division=0
))

# Save both models
joblib.dump(logistic_model, "models/logistic_model.pkl")
joblib.dump(random_forest_model, "models/random_forest_model.pkl")

print("\nBoth models saved successfully!")
