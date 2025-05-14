import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch


## 1. 데이터 전처리
# # 1-1. 채용 공고 데이터 
df_job = pd.read_excel("data/jobpost.xlsx")
df_job['job_data'] = (
    '본 직무는 ' + df_job['직무'].fillna('') + ' 를 주 업무로 하며, '
    + '우대사항으로는 ' + df_job['우대사항'].fillna('') + '을(를) 보유한 지원자를 선호합니다.'
)

# # 1-2. 지원자 데이터 
df_app = pd.read_excel("data/applicantsInfo.xlsx")
df_app['app_data'] = (
    df_app['국적'] + ' 국적을 가진 지원자는 '
    + df_app['제 1언어'] + '를 제1언어로 사용하며, '
    + df_app['전공'] + '을 전공하였습니다. '
    + '대외 활동으로는 ' + df_app['대외 활동'] + '을(를) 수행하였고, '
    + '한국어 능력은 TOPIK ' + df_app['TOPIK 등급'] + ' 수준입니다.'
)

# print(df_app[['app_data']].head())
# print(df_job[['job_data']].head())


## 2. 임베딩 
# KoBERT 기반으로 학습된 문장 임베딩 전용 모델 사용 
model = SentenceTransformer('snunlp/KR-SBERT-V40K-klueNLI-augSTS')

job_texts = df_job['job_data'].tolist()
app_texts = df_app['app_data'].tolist()

job_embeddings = model.encode(job_texts, convert_to_tensor=True)
app_embeddings = model.encode(app_texts, convert_to_tensor=True)

## 3. Pickin 지수 계산 함수
def get_pickin_scores(applicant_id):
    """
    applicant_id: 1부터 시작하는 지원자 ID (엑셀 기준 인덱스 + 1)
    return: list of dicts -> [{채용공고id, pickin지수}, ...]
    """

    idx = applicant_id - 1  # 리스트 인덱스는 0부터 시작

    # 해당 지원자의 임베딩과 전체 채용공고 간 유사도 계산
    similarities = util.cos_sim(app_embeddings[idx], job_embeddings)
    sim_scores = similarities.squeeze().tolist()

    max_sim = max(sim_scores)
    min_sim = min(sim_scores)

    # 유사도 정규화
    if max_sim != min_sim:
        scaled_scores = [(s - min_sim) / (max_sim - min_sim) for s in sim_scores]
    else:
        scaled_scores = [0.0 for _ in sim_scores]

    # 결과 매핑
    pickin_scores = [
        {"채용공고id": i + 1, "pickin지수": round(score, 4)}
        for i, score in enumerate(scaled_scores)
    ]

    return pickin_scores


if __name__ == "__main__":
    applicant_id = 1  # 원하는 지원자 ID
    result = get_pickin_scores(applicant_id)
    for r in result:
        print(r)

# # app_embeddings[0] : 첫 번째 지원자와 각 공고 간의 cosine similarity 벡터
# similarities = util.cos_sim(app_embeddings[0], job_embeddings)

# print(similarities)

# # 유사도 스케일링
# sim_scores = similarities.squeeze().tolist()
# max_sim = max(sim_scores)
# min_sim = min(sim_scores)

# scaled_scores = [(s - min_sim) / (max_sim - min_sim) for s in sim_scores]

# print(scaled_scores)


