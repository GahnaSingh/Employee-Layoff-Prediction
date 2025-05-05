pip install -r requirements.txt
/
from fastapi import FastAPI, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
import io

app = FastAPI(title="Layoff Risk API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Set to specific domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and selected features
model = joblib.load("layoff_model.pkl")
selected_features = joblib.load("model_features.pkl")

@app.post("/upload")
async def upload_csv(file: UploadFile):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files supported.")
    df = pd.read_csv(io.BytesIO(await file.read()))
    return {"columns": df.columns.tolist(), "data": df.to_dict(orient="records")}

@app.post("/predict")
async def predict(data: list[dict], layoffs: int = Query(...)):
    df = pd.DataFrame(data)
    
    # Check for missing columns
    missing = set(selected_features) - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")
    
    # Predict layoff risk
    X = df[selected_features]
    df["layoff_risk"] = model.predict_proba(X)[:, 1]
    
    # Return top N risky employees
    result = df.sort_values("layoff_risk", ascending=False).head(layoffs)
    return JSONResponse(result.to_dict(orient="records"))
