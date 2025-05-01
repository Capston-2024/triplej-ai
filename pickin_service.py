import joblib
import numpy as np

model = joblib.load("model/pickin_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

def predict_label(request):
    input_data = np.array([[
        label_encoders['국적'].transform([request.nationality])[0],
        label_encoders['최종학력'].transform([request.education])[0],
        label_encoders['TOPIK'].transform([request.topik])[0],
        label_encoders['관심직무'].transform([request.work])[0]
    ]])

    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    return int(prediction[0])
