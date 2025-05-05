import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

# ---------------------- Load and Preprocess Dataset ----------------------
file_path = "Large_Synthetic_Layoff_Prediction_Dataset.csv"
data = pd.read_csv(file_path)

# Encode categorical features
cat_cols = ['Department', 'JobLevel']
encoder = LabelEncoder()
for col in cat_cols:
    data[col] = encoder.fit_transform(data[col])

# Define features
features = ['PerformanceScore', 'AttendanceRate', 'SalaryImpact', 'MarketTrend', 'YearsAtCompany', 'ProjectContributions']
scaler = MinMaxScaler()
data[features] = scaler.fit_transform(data[features])

# Define input/output
X = data[features]
y = data['LayoffRisk']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Oversample minority class
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)

print("\nData Preprocessing Completed!")
print(f"Training Samples: {X_train_bal.shape[0]}, Testing Samples: {X_test.shape[0]}")
print("Class Distribution After Resampling:", np.bincount(y_train_bal))

# ---------------------- Train BiLSTM ----------------------
X_train_bilstm = np.expand_dims(X_train_bal.values, axis=1)
X_test_bilstm = np.expand_dims(X_test.values, axis=1)

bilstm_model = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(X_train_bilstm.shape[1], X_train_bilstm.shape[2])),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

bilstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
bilstm_model.fit(X_train_bilstm, y_train_bal, epochs=20, batch_size=16, validation_data=(X_test_bilstm, y_test))

# ---------------------- XGBoost Feature Importance ----------------------
dummy_model = xgb.XGBClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
dummy_model.fit(X_train_bal, y_train_bal)

perm = permutation_importance(dummy_model, X_test, y_test, scoring='accuracy')
feature_importance = perm.importances_mean
important_features = np.argsort(feature_importance)[::-1]
selected_features = [features[i] for i in important_features[:5]]

print("\nTop 5 Selected Features:", selected_features)

# ---------------------- Final XGBoost Model ----------------------
X_train_final = X_train_bal[selected_features]
X_test_final = X_test[selected_features]

xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
xgb_model.fit(X_train_final, y_train_bal)

# Evaluation
xgb_preds = xgb_model.predict(X_test_final)
accuracy = accuracy_score(y_test, xgb_preds)
print(f"\nFinal XGBoost Accuracy: {accuracy:.4f}")



# ---------------------- Save Model and Features ----------------------
joblib.dump(xgb_model, "layoff_model.pkl")
joblib.dump(selected_features, "model_features.pkl")
print("\nModel and Features Saved Successfully!")
