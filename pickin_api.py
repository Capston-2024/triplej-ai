from fastapi import APIRouter
from pickin_service import predict_pickin_score, extract_keywords_from_text, extract_keywords_from_sheet, calc_similarity, feedback_service
from schema.prediction_request import prediction_request
from schema.text_keyword_extraction_request import text_keyword_extraction_request
from schema.text_similarity_request import text_similarity_request
from schema.feedback_request import feedback_request

router = APIRouter()

@router.post("/predict")
def predict(request: prediction_request):
    return predict_pickin_score(request)

@router.post("/feedback") # 키워드 및 유사도 기반 정량적 피드백 & 텍스트 기반 정성적 피드백
def feedback(request: feedback_request):
    return feedback_service(request)