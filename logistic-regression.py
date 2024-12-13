import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Load the dataset from a GitHub URL
def load_data():
    # Replace 'yourusername' and 'repository' with your GitHub details
    url = "https://raw.githubusercontent.com/Agamjot27/logistic-regression/main/framingham.csv"
    return pd.read_csv(url)

# Preprocess the dataset
def preprocess_data(data):
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Split into features (X) and target (y)
    X = data_imputed.drop(columns='TenYearCHD')
    y = data_imputed['TenYearCHD']

    # Standardize continuous features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

# Train and evaluate the model
def train_and_evaluate(X, y):
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Evaluate the model
    report = classification_report(y_test, y_pred, target_names=["No CHD", "CHD"])
    roc_auc = roc_auc_score(y_test, y_prob)

    return model, report, roc_auc

# Main function
def main():
    # Load and preprocess the data
    data = load_data()
    X, y = preprocess_data(data)

    # Train and evaluate the model
    model, report, roc_auc = train_and_evaluate(X, y)

    # Output the results
    print("Classification Report:")
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
