import joblib
import numpy as np
from keybert import KeyBERT
from konlpy.tag import Okt
import torch
from transformers import BertModel, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os

model = joblib.load("model/pickin_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_encoders = joblib.load("model/label_encoders.pkl")

kb_model = KeyBERT(model="distiluse-base-multilingual-cased-v1")
okt = Okt() # 형태소 분석기 초기화

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
bert_model = BertModel.from_pretrained('monologg/kobert')
bert_model.eval()

client = OpenAI(api_key=os.getenv('OPENAI_APIKEY'))

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

def get_kobert_embedding(text):
    """
    return_tensors="pt": PyTorch tensor로 변환
    truncation=True: 512 토큰을 초과하는 경우 잘라내기
    padding=True: 길이를 맞추기 위해 padding 추가
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad(): # 불필요한 grad 계산 비활성화
        outputs = bert_model(**inputs)
    
    last_hidden_state = outputs.last_hidden_state 
    attention_mask = inputs['attention_mask']

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked_embeddings = last_hidden_state * mask
    summed = torch.sum(masked_embeddings, dim=1)
    summed_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
    mean_pooled = summed / summed_mask # 평균 풀링 방식을 활용하여 텍스트 전체의 의미 요약

    return mean_pooled.squeeze().numpy() # numpy 배열로 변환 후 return

def calc_similarity(request): # 텍스트 기반 유사도 계산
    job = request.job
    letter = request.letter

    job_emb = get_kobert_embedding(job)
    letter_emb = get_kobert_embedding(letter)

    similarity = cosine_similarity([job_emb], [letter_emb])
    
    return similarity[0][0]

def feedback(request): # 키워드 및 유사도 기반 정량적 피드백 & 텍스트 기반 정성적 피드백
    prompt = f"""
    지원자의 자기소개서 내용: {request.letter}
    채용공고 내용: {request.job}
    지원자의 자기소개서에 포함되지 않거나 강조되지 않은 키워드: {', '.join(request.missing_keywords)}
    채용공고와 지원자의 자기소개서 간 의미적 유사도: {request.similarity}
    
    위의 정보들을 바탕으로, 지원자의 자기소개서를 첨삭해줘. (아래와 같은 방향으로)
    - 부족한 키워드를 자기소개서에 자연스럽게 추가할 수 있도록
    - 채용공고에서 요구하는 인재상에 부합하도록
    """

    gpt = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )

    response = gpt.choices[0].message.content.strip()
    
    return response