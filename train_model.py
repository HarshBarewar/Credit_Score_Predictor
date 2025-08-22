import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv(r'C:\Users\Admin\Desktop\credit score\dataset\credit_score_modified.csv')

# Features to use
features = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
            'Num_of_Delayed_Payment', 'Num_Credit_Inquiries', 'Outstanding_Debt',
            'Credit_Utilization_Ratio', 'Payment_of_Min_Amount', 'Payment_Behaviour',
            'Monthly_Balance']

X = df[features]
y = df['Credit_Score']

# Encode categorical features
le_min_amount = LabelEncoder()
le_behaviour = LabelEncoder()

X['Payment_of_Min_Amount'] = le_min_amount.fit_transform(X['Payment_of_Min_Amount'])
X['Payment_Behaviour'] = le_behaviour.fit_transform(X['Payment_Behaviour'])

# Ensure all features are numeric
X = X.apply(pd.to_numeric, errors='coerce')
X = X.dropna()
y = y[X.index]  # Align target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/credit_score_model.pkl')
print("✅ Model trained and saved successfully.")
