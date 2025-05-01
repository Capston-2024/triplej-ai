from fastapi import APIRouter
from pickin_service import predict_label
from schema.prediction_request import prediction_request

router = APIRouter()

@router.post("/predict")
def predict(request: prediction_request):
    prediction = predict_label(request)
    return {"prediction": prediction}
