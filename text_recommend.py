# 필요 라이브러리
import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained("beomi/kcbert-large")
device = torch.device('cpu')
# 데이터 호출
data=pd.read_csv('sentimenttalk.xlsx')

# 토크나이저 함수 정의
def Build_X (sents, tokenizer, device):
    X = tokenizer(sents, padding=True, truncation=True, return_tensors='pt')
    return torch.stack([
        X['input_ids'],
        X['token_type_ids'],
        X['attention_mask']
    ], dim=1).to(device)

# 데이퍼프레임에 데이터 추가
def data_add(df,sentence):
    new_data = [0,0,0,0,0,0,sentence,0,0,0,0,0] # 이거슨...컬럼 위치찾기..
    df.loc[len(df)] = new_data

# 입력 df # sentence는 입력할 문장
def text_recommend(df, sentence):
    data_add(df, sentence) # 데이터 추가
    docs = df[df.columns[6]] # person['사람문장1']
    sen = [sent for sent in docs] # 안에 있는 문장들 리스트화
    x = Build_X(sen, tokenizer, device) # 토크나이저 함수 적용
    input_ids = x[:, 0] # ids 추출
    input_ids_numpy = input_ids.numpy()
    # matrix = np.asmatrix(input_ids_numpy)
    matrix = np.array(input_ids_numpy) # ids matrix화
    cosine_matrix = cosine_similarity(matrix, matrix) # 코사인유사도 적용
    np.round(cosine_matrix, 4)
    idx2sent = {} # index 입력 -> 문장 출력
    for i, c in enumerate(docs): idx2sent[i] = c
    sent2idx = {} # 문장 입력 -> index 출력
    for i, c in idx2sent.items(): sent2idx[c] = i

    idx = sent2idx[sentence] # 문장을 넣어 index 출력
    sim_score = [(i, c) for i, c in enumerate(cosine_matrix[idx]) if i != idx]
    sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
    sim_score = [(idx2sent[i], c) for i, c in sim_score[:10]]
    return print(sim_score)
text_recommend(data,'우리집에서 고양이보고갈래.')
