import joblib
import numpy as np
from keybert import KeyBERT
from konlpy.tag import Okt

model = joblib.load("model/pickin_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

kb_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
okt = Okt() # 형태소 분석기 초기화

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

def extract_keywords_from_text(request): # 텍스트 기반 키워드 추출, 채용공고/자기소개서 공통
    noun_tokens = okt.nouns(request.text) # 명사 추출
    noun_text = " ".join(noun_tokens) 
    
    keywords = kb_model.extract_keywords(noun_text, top_n=request.size, stop_words=None)
    
    cleaned_keywords = []
    for keyword, _ in keywords: # 중복 제거
        kw = keyword.strip()
        if kw not in cleaned_keywords:
            cleaned_keywords.append(kw)

    return cleaned_keywords

def extract_keywords_from_sheet(request): # 스프레드시트 기반 키워드 추출, 채용공고/자기소개서 공통
    ### todo ###
    return request

def extract_sentence_embeddings(request): # 텍스트 기반 문장 임베딩 추출, 채용공고/자기소개서 공통
    return request

def calc_similarity(request): # 문장 임베딩 기반 유사도 계산
    return request

def feedback(request): # 키워드 및 유사도 기반 정량적 피드백 & 텍스트 기반 정성적 피드백
    return request