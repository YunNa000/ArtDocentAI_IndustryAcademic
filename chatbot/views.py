from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

import json
import numpy as np
import pandas as pd
import os
import openai

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()
# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터 로드 및 모델 초기화
train_df_new = pd.read_excel('chatbot/static/assets/xlsx/tokenized_semart_train_combined.xlsx')
val_df_new = pd.read_excel('chatbot/static/assets/xlsx/tokenized_semart_val_combined.xlsx')
test_df_new = pd.read_excel('chatbot/static/assets/xlsx/tokenized_semart_test_combined.xlsx')
all_df = pd.concat([train_df_new, val_df_new, test_df_new], ignore_index=True)

def home(request):
    context = {}
    return render(request, "home.html", context)

def favorite(request):
    context = {}
    return render(request, "favorite.html", context)

def IntroTeam(request):
    context = {}
    return render(request, "IntroTeam.html", context)

def IntroArtist(request):
    context = {}
    return render(request, "IntroArtist.html", context)

from IPython.display import Image, display

# Function to display the image from the IMAGE_FILE path
def display_artwork_image_proportionally(most_similar_artwork, width=None):
    image_path = f"Images/{most_similar_artwork['IMAGE_FILE']}"
    display(Image(filename=image_path, width=width))

import json

class ArtData:
    def __init__(self):
        self.title = None
        self.description = None
        self.technique = None
        self.author = None
        self.shape = None
        self.type = None
        self.school = None
        self.timeframe = None
        self.depiction = None


def json_to_artdata(json_str):
    try:
        print("JSON 字符串内容:", json_str)
        data = json.loads(json_str)
        
        art = ArtData()
        art.title = data.get('title')
        art.description = data.get('description')
        art.technique = data.get('technique')
        art.author = data.get('author')
        art.shape = data.get('shape')
        art.type = data.get('type')
        art.school = data.get('school')
        art.timeframe = data.get('timeframe')
        art.depiction = data.get('depiction')
        
        return art
    except json.JSONDecodeError as e:
        print("JSON 解析错误:", e)
        return None
    
# 점수 계산을 위한 함수
def calculate_director_score(directors,directors_to_check):
    score = 0
    for director in directors_to_check:
        if director in directors:
            score += 1
    return score

def calculate_string_score(string, substrings):
    for substring in substrings:
        if substring in string:
            return 1
    return 0


# GPT-3 API 호출 함수
def call_gpt_api(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response['choices'][0]['message']['content'].strip()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Hugging Face 임베딩 모델 생성
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 인덱스 경로
faiss_index_path = './chatbot/db/faiss'
faiss1_index_path = './chatbot/db/faiss1'

vectorstore = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
vectorstore2 = FAISS.load_local(faiss1_index_path, embeddings_model, allow_dangerous_deserialization=True)
def search_reviews(query):
    # 검색어를 기반으로 벡터 저장소에서 유사한 리뷰 검색
    search_result = vectorstore.similarity_search(query, k=10)
    
    # 검색 결과 출력
    results = []
    for i, review in enumerate(search_result):
        result = {
            'title': review.metadata['title'],
            'content': review.page_content
        }
        results.append(result)
    
    return results

def get_json_text(query_text):
    prompt = f'Generate a JSON object that includes the title, author, technique, shape, type, school, and timeframe of the artwork described in the following query: "{query_text}"'
    
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    answer_text = response['choices'][0]['message']['content'].strip()
    print("Generated JSON Text: ", answer_text)
    return answer_text
    

def get_answer(query_text):
    scoring = all_df.copy()
    answer = get_json_text(query_text)
    if answer is None:
        return pd.DataFrame(), "Empty or invalid response from OpenAI API"
    
    try:
        art_data = json_to_artdata(answer)
    except json.JSONDecodeError as e:
        print("JSONDecodeError: ", str(e))
        return pd.DataFrame(), "Failed to parse JSON data"

    if art_data is None:
        return pd.DataFrame(), "Failed to parse JSON data"
    
    scoring['total_score'] = 0
    alpha = 1.2

    if art_data.title is not None:
        scoring['scoreTITLE'] = scoring['TITLE'].apply(lambda x: calculate_string_score(x, art_data.title))
        scoring['total_score'] += scoring['scoreTITLE'] * alpha

    if art_data.author is not None:
        scoring['scoreAUTHOR'] = scoring['AUTHOR'].apply(lambda x: calculate_string_score(x, art_data.author))
        scoring['total_score'] += scoring['scoreAUTHOR'] * alpha

    if art_data.technique is not None:
        scoring['scoreTECHNIQUE'] = scoring['TECHNIQUE'].apply(lambda x: calculate_string_score(x, art_data.technique))
        scoring['total_score'] += scoring['scoreTECHNIQUE'] * alpha

    if art_data.shape is not None:
        scoring['scoreSHAPE'] = scoring['SHAPE'].apply(lambda x: calculate_string_score(x, art_data.shape))
        scoring['total_score'] += scoring['scoreSHAPE'] * alpha

    if art_data.type is not None:
        scoring['scoreTYPE'] = scoring['TYPE'].apply(lambda x: calculate_string_score(x, art_data.type))
        scoring['total_score'] += scoring['scoreTYPE'] * alpha

    if art_data.school is not None:
        scoring['scoreSCHOOL'] = scoring['SCHOOL'].apply(lambda x: calculate_string_score(x, art_data.school))
        scoring['total_score'] += scoring['scoreSCHOOL'] * alpha

    review_data = {}
    search_result = vectorstore.similarity_search(query_text)
    for i, review in enumerate(search_result):
        review_data[review.metadata['title']] = {
            'author': review.metadata['author'],
            'content': review.page_content
        }
        index = scoring[scoring['TITLE'] == review.metadata['title']].index
        if not index.empty:
            scoring.loc[index, 'total_score'] += 2

    review_data2 = {}
    search_result = vectorstore2.similarity_search(query_text)
    for i, review in enumerate(search_result):
        review_data2[review.metadata['title']] = {
            'author': review.metadata['author'],
            'content': review.page_content
        }
        index = scoring[scoring['TITLE'] == review.metadata['title']].index
        if not index.empty:
            scoring.loc[index, 'total_score'] = scoring.loc[index, 'total_score'] + 2

    result_df = scoring.sort_values('total_score', ascending=False).head(20)
    most_similar_index = result_df['total_score'].idxmax()
    most_similar_title = result_df['TITLE'][most_similar_index]
    total = "요구사항:이 정보에 기반해서 추천된 작품이야 질문에 맞춰 한국어로 답변을 생성해줘" + "질문:" + query_text + most_similar_title

    if art_data.title != None:
        total += "제목: " + str(result_df['TITLE'][most_similar_index]) + "\n"
    if art_data.author != None:
        total += "작가: " + str(result_df['AUTHOR'][most_similar_index]) + "\n"
    if art_data.technique != None:
        total += "technique: " + str(result_df['TECHNIQUE'][most_similar_index]) + "\n"
    if art_data.shape != None:
        total += "SHAPE: " + str(result_df['SHAPE'][most_similar_index]) + "\n"
    if art_data.type != None:
        total += "TYPE: " + str(result_df['TYPE'][most_similar_index]) + "\n"
    if art_data.school != None:
        total += "학교: " + str(result_df['SCHOOL'][most_similar_index]) + "\n"

    prompt = "질문:" + query_text + total
    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True
    )
    answer_text = ""
    for chunk in chat_completion:
        if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
            content = chunk.choices[0].delta.get("content", "")
            if content:
                answer_text += content
    return result_df, answer_text

@csrf_exempt
def chatanswer(request):
    if request.method == 'POST':
        context = {}
        chattext = request.POST.get('chattext', '')

        # get_answer 함수 호출
        result_df, answer = get_answer(chattext)
        if result_df is None:
            result_df = pd.DataFrame()  # 빈 데이터프레임으로 초기화

        # 데이터프레임을 JSON 형식으로 변환
        result_df_json = result_df.to_json(orient='records')

        context['result_df'] = json.loads(result_df_json)  # JSON 문자열을 딕셔너리로 변환하여 추가
        context['answer'] = answer  # GPT-3 응답

        # 이미지 경로 추가
        if not result_df.empty and 'IMAGE_FILE' in result_df.columns:
            image_paths = [f"static/Images/{result_df['IMAGE_FILE'].iloc[0]}"]
        else:
            image_paths = []

        context['image_paths'] = image_paths

        return JsonResponse(context, content_type="application/json")
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)