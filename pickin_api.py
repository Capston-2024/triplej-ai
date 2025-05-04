from fastapi import APIRouter
from pickin_service import predict_label, extract_keywords_from_text, extract_keywords_from_sheet, extract_sentence_embeddings, calc_similarity, feedback
from schema.prediction_request import prediction_request
from schema.text_keyword_extraction_request import text_keyword_extraction_request
from schema.text_similarity_request import text_similarity_request

router = APIRouter()

@router.post("/predict")
def predict(request: prediction_request):
    prediction = predict_label(request)
    return {"prediction": prediction}

@router.post("/keywords/text") # 텍스트 기반 키워드 추출, 채용공고/자기소개서 공통
def extract_keywords_from_text(request: text_keyword_extraction_request):
    return extract_keywords_from_text(request)

@router.post("/keywords/sheet") # 스프레드시트 기반 키워드 추출, 채용공고/자기소개서 공통
def extract_keywords_from_sheet(request):
    return extract_keywords_from_sheet(request)

@router.post("/similarity") # 텍스트 기반 유사도 계산
def calc_similarity(request: text_similarity_request):
    return calc_similarity(request)

@router.post("/feedback") # 키워드 및 유사도 기반 정량적 피드백 & 텍스트 기반 정성적 피드백
def feedback(request):
    return feedback(request)
