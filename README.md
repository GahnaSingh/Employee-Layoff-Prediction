# Employee Layoff Prediction

A full-stack AI-powered layoff prediction system that leverages machine learning and deep learning models to forecast potential employee layoffs. This project includes a React + TailwindCSS frontend and a
Python FastAPI backend, integrated with real-world layoff datasets and powerful ML algorithms.

## Project Structure
```
HR/
├── backend/
│ ├── main.py
│ ├── model_train.py
│ └── requirements.txt
├── frontend/
│ ├── public/
│ └── src/
│ ├── assets/
│ ├── App.jsx
│ ├── App.css
│ ├── index.css
│ ├── main.tsx
│ └── ...
├── Large_Synthetic_Layoff_Prediction_Dataset.csv
└── README.md
```

## Features

-  Predicts employee layoffs using ML, DL, and hybrid models
-  Uses structured layoff datasets for training
-  Frontend: React + TailwindCSS
-  Backend: FastAPI with scikit-learn, XGBoost, LightGBM, etc.
-  Real-time user input and predictions
-  High accuracy, precision, recall, and F1-score
-  API integration between UI and model endpoints


##  Tech Stack

### Frontend
- React
- TailwindCSS
- TypeScript
- Vite

### Backend
- Python 3.x
- FastAPI
- scikit-learn, XGBoost, LightGBM
- Pandas, NumPy
- 

##  Machine Learning Models Used

- Random Forest
- XGBoost
- LightGBM
- Feedforward Neural Networks
- Hybrid model (stacking/ensemble)

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## Dataset
The project uses a synthetic dataset:Large_Synthetic_Layoff_Prediction_Dataset.csv

Includes features like:

- Company size
- Department
- Tenure
- Previous layoff history
- Revenue changes
- Layoff decision (Target)
