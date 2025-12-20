# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/nsa9/bank-customer-churn/bank_customer_churn.csv"
bank_dataset = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Define the target variable for the classification task
target = 'Exited'

# List of numerical features in the dataset
numeric_features = [
    'CreditScore',       # Customer's credit score
    'Age',               # Customer's age
    'Tenure',            # Number of years the customer has been with the bank
    'Balance',           # Customer’s account balance
    'NumOfProducts',     # Number of products the customer has with the bank
    'HasCrCard',         # Whether the customer has a credit card (binary: 0 or 1)
    'IsActiveMember',    # Whether the customer is an active member (binary: 0 or 1)
    'EstimatedSalary'    # Customer’s estimated salary
]

# List of categorical features in the dataset
categorical_features = [
    'Geography',         # Country where the customer resides
]

# Define predictor matrix (X) using selected numeric and categorical features
X = bank_dataset[numeric_features + categorical_features]

# Define target variable
y = bank_dataset[target]


# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,              # Predictors (X) and target variable (y)
    test_size=0.2,     # 20% of the data is reserved for testing
    random_state=42    # Ensures reproducibility by setting a fixed random seed
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="nsa9/bank-customer-churn",
        repo_type="dataset",
    )
