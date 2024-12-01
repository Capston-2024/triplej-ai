# pickin.py 파일 작성 (로컬에서)
from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# 로컬 경로에서 모델 로딩
model = joblib.load('pickin_model.pkl')  # 로컬 경로로 수정
scaler = joblib.load('scaler.pkl')  # 스케일러도 로딩
label_encoders = joblib.load('label_encoders.pkl')  # 레이블 인코더 로딩

class PredictionRequest(BaseModel):
    nationality: str
    education: str
    topik: str
    work: str

@app.post("/predict")
def predict(request: PredictionRequest):
    input_data = np.array([[
        label_encoders['국적'].transform([request.nationality])[0],
        label_encoders['최종학력'].transform([request.education])[0],
        label_encoders['TOPIK'].transform([request.topik])[0],
        label_encoders['관심직무'].transform([request.work])[0]
    ]])
    
    # 데이터 정규화 (훈련된 스케일러 사용)
    input_data_scaled = scaler.transform(input_data)
    
    # 예측 수행
    prediction = model.predict(input_data_scaled)
    
    return {"prediction": int(prediction[0])}
