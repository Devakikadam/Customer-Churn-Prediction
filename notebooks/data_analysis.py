
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:", df.duplicated().sum())

# Check target column distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())




# Data Cleaning
print("\nDataset Shape:", df.shape)

# Remove customer ID column
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# Check missing values after conversion
print("\nMissing Values:")
print(df.isnull().sum())

# Remove rows with missing values
df.dropna(inplace=True)

print("\nCleaned Dataset Shape:", df.shape)

# Save cleaned dataset
df.to_csv("data/cleaned_customer_churn.csv", index=False)

print("\nCleaned dataset saved successfully!")

# Plot churn distribution
sns.countplot(data=df, x="Churn")
plt.title("Customer Churn Distribution")
plt.savefig("data/churn_distribution.png")
plt.show()
