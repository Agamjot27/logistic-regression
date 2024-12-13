import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def load_data():
    
    url = "https://raw.githubusercontent.com/Agamjot27/logistic-regression/main/framingham.csv"
    return pd.read_csv(url)


def preprocess_data(data):
   
    imputer = SimpleImputer(strategy='mean')
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    
    X = data_imputed.drop(columns='TenYearCHD')
    y = data_imputed['TenYearCHD']

   
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def train_and_evaluate(X, y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

   
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    
    report = classification_report(y_test, y_pred, target_names=["No CHD", "CHD"])
    roc_auc = roc_auc_score(y_test, y_prob)

    return model, report, roc_auc


def main():
   
    data = load_data()
    X, y = preprocess_data(data)

   
    model, report, roc_auc = train_and_evaluate(X, y)

   
    print("Classification Report:")
    print(report)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    main()
