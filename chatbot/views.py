from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import random
from django.conf import settings

import markdown2
import json
import numpy as np
import pandas as pd
import os
from openai import OpenAI

from transformers import BertModel, BertTokenizer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os
key="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"
이전질문=""
# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] =key 

load_dotenv()
# OpenAI API 키 설정
client = OpenAI(api_key = key)

# 데이터 로드 및 모델 초기화
all_df = pd.read_excel('chatbot\static/xlsx/finaldata_cleaned.xlsx')
    
def home(request):
    if settings.DEBUG:
        image_directory = os.path.join(settings.STATICFILES_DIRS[0], 'art-images')
    else:
        image_directory = os.path.join(settings.STATIC_ROOT, 'art-images')

    all_images = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    random_images = random.sample(all_images, 9) if len(all_images) >= 9 else all_images
    image_urls = [os.path.join(settings.STATIC_URL, 'art-images', img) for img in random_images]

    return render(request, "home.html", {'image_urls': json.dumps(image_urls)})

def favorite(request):
    context = {}
    return render(request, "favorite.html", context)

def IntroTeam(request):
    context = {}
    return render(request, "IntroTeam.html", context)

def IntroArtist(request):
    df = pd.read_csv(r'chatbot\static/xlsx/grouped_artist_table_sample.csv', encoding='cp949')
    
    # 데이터 파싱 및 리스트 구성
    artists = []
    for _, row in df.iterrows():
        # 각 row에 대한 artist dictionary를 생성
        artist = {
            'name': row['AUTHOR'],  # 작가의 이름을 'AUTHOR' 열에서 가져옴
            'description': row['DESCRIPTION'],  # 작가의 설명을 'DESCRIPTION' 열에서 가져옴
            'artworks': [artwork.strip() for artwork in str(row['ARTWORK']).split(',')] if pd.notna(row['ARTWORK']) else []
        }
        # artist dictionary를 artists 리스트에 추가
        artists.append(artist)
    
    # artists 리스트를 컨텍스트로 IntroArtist.html 템플릿 렌더링
    return render(request, "IntroArtist.html", {'artists': artists})

from IPython.display import Image, display

# Function to display the image from the IMAGE_FILE path
def display_artwork_image_proportionally(most_similar_artwork, width=None):
    image_path = f"art-images/{most_similar_artwork['IMAGE_FILE']}"
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
        art.title = data.get('description')
        #art.description = data.get('description')
        art.technique = data.get('technique')
        art.author = data.get('author')
        art.shape = data.get('shape')
        art.type = data.get('type')
        art.school = data.get('nationality')
        art.timeframe = data.get('art form')
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

def calculate_string_score1(string, substrings):
    score = 0
    for substring in substrings:
        if substring.lower() in string:
            score += 0.5
    return score

def calculate_string_score(string_list, substring):
    for string in string_list:
        if substring in string:
            return 1
    return 0

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Hugging Face 임베딩 모델 생성
embeddings_model = HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli',
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# FAISS 인덱스 경로
faiss_index_path = './chatbot/db/faiss'
faiss1_index_path = './chatbot/db/faiss1'

# vectorstore = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
# vectorstore2 = FAISS.load_local(faiss1_index_path, embeddings_model, allow_dangerous_deserialization=True)
# FAISS 데이터베이스 경로 설정
FAISS_DB_PATH = os.path.join(settings.BASE_DIR, "chatbot", "db", "school_images_index")

# FAISS 데이터베이스 불러오기
embeddings = OpenAIEmbeddings(openai_api_key=key)
vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings,allow_dangerous_deserialization=True)

def search_reviews(query):
    # 검색어를 기반으로 벡터 저장소에서 유사한 리뷰 검색
    search_result = vectorstore.similarity_search(query, k=10)
    
    # 검색 결과 출력
    results = []
    for i, review in enumerate(search_result):
        result = {
            'title': review.metadata['IMAGE_FILE'],
            'content': review.page_content
        }
        results.append(result)
    
    return results


def get_json_text(query_text):
    prompt = """
    다음 질문을 기반으로 미술 작품 관련 정보를 키워드로 분류해 주세요. 한국어 키워드는 영어로 적절하게 번역해서 넣어 주세요.

    예시 질문: "예수와 동방박사가 나오는 르네상스 시대 프랑스산 가로형 유화 그림을 찾아줘"

    가능한 값들:
    {
        "author": null,  # 작가를 모를 경우 null
        "technique": ["Oil", "Tempera", "Watercolor", "Gouache", "Ink", "Charcoal", "Pastel", "Pencil", "Lithograph", "Etching", "Enamel", "Egg", "Graphite", "Distemper", "Bronze", "Gold", "Silver", "Woodcu"],
        "shape": ["Horizontal", "Vertical"],
        "type": ["Genre", "Historical", "Interior", "Landscape", "Mythological", "Portrait", "Religious", "Still life", "Study"],
        "school": ["American", "Austrian", "Belgian", "Bohemian", "Catalan", "Danish", "Dutch", "English", "Finnish", "Flemish", "French", "German", "Greek", "Hungarian", "Irish", "Italian", "Netherlandish", "Norwegian", "Polish", "Portuguese", "Russian", "Scottish", "Spanish", "Swedish", "Swiss"],
        "art_form": ["Art", "Baroque", "Impressionism", "Medieval", "Neoclassicism", "Nouveau", "Period", "Post-Impressionism", "Realism", "Renaissance", "Rococo", "Romanticism"]
    }

    출력 형식은 다음과 같아야 합니다:
    {
        "author": null,  # 작가 미상일 경우
        "technique": "Oil", # 하나의 문자열만 입력 or null
        "shape":"Horizontal",  # 하나의 문자열만 입력 or null
        "type": "Religious",   # 하나의 문자열만 입력 or null
        "school": "French",   # 하나의 문자열만 입력 or null
        "art_form": "Renaissance"  # 하나의 문자열만 입력
    }

    주의사항:
    1. 확실하지 않은 정보는 반드시 null로 처리
    2. 각 필드당 최대 2개까지만 값 입력 가능
    3. 모든 값은 제시된 옵션 중에서만 선택
    4. 한국어 키워드는 반드시 영어로 변환
    5. 각 배열은 오름차순으로 정렬

    질문:
    """ + query_text
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # 또는 "gpt-3.5-turbo"
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )
        
        answer_text = response.choices[0].message.content.strip()
        print("Generated JSON Text: ", answer_text)

        # Remove backticks and any extra text around the JSON object
        start_idx = answer_text.find('{')
        end_idx = answer_text.rfind('}') + 1
        json_text = answer_text[start_idx:end_idx]
        
        try:
            answer_json = json.loads(json_text)
        except json.JSONDecodeError as e:
            print("JSONDecodeError: ", str(e))
            return None
        
        return answer_json
        
    except Exception as e:
        print(f"Error in API call: {e}")
        return None
    

def get_answer(query_text):
    scoring = all_df.copy()
    answer_json = get_json_text(query_text)
    if answer_json is None:
        return pd.DataFrame(), "Empty or invalid response from OpenAI API"
    
    art_data = json_to_artdata(json.dumps(answer_json))  # JSON 객체를 문자열로 변환하여 전달

    if art_data is None:
        return pd.DataFrame(), "Failed to parse JSON data"
    
    scoring['total_score'] = 0
    alpha = 1.1

    if art_data.title is not None:
        titles=art_data.title[0].split()
        scoring['scoreTITLE'] = scoring['TITLE'].apply(lambda x: calculate_string_score1(x, titles))
        scoring['total_score'] += scoring['scoreTITLE'] * alpha
    print(art_data.author)
    if art_data.author is not None:
        Author = art_data.author.lower().split()
        scoring['scoreAUTHOR'] = scoring['AUTHOR'].apply(lambda x: calculate_string_score1(x, Author))
        scoring['total_score'] += scoring['scoreAUTHOR'] * alpha

    if art_data.technique is not None:
        scoring['scoreTECHNIQUE'] = scoring['TECHNIQUE'].apply(lambda x: calculate_string_score([x], art_data.technique))
        scoring['total_score'] += scoring['scoreTECHNIQUE'] * alpha
 
    if art_data.shape is not None:
        scoring['scoreSHAPE'] = scoring['SHAPE'].apply(lambda x: calculate_string_score([x], art_data.shape))
        scoring['total_score'] += scoring['scoreSHAPE'] * alpha

    if art_data.type is not None:
        scoring['scoreTYPE'] = scoring['TYPE'].apply(lambda x: calculate_string_score([x], art_data.type))
        scoring['total_score'] += scoring['scoreTYPE'] * alpha
    if art_data.school is not None:
        scoring['scoreSCHOOL'] = scoring['SCHOOL'].apply(lambda x: calculate_string_score([x], art_data.school))
        scoring['total_score'] += scoring['scoreSCHOOL'] * alpha

    # if art_data.timeframe is not None:
    #     scoring['scoreTIMEFRAME'] = scoring['TIMEFRAME'].apply(lambda x: calculate_string_score(x, art_data.timeframe))
    #     scoring['total_score'] += scoring['scoreTIMEFRAME'] * alpha
    #scoring.to_csv(r'C:\result_datatitle_scoreing.csv', index=False)  
    
    if query_text is not None:
        review_data = {}
        search_result = vectorstore.similarity_search(query_text)

        for i, review in enumerate(search_result):
            review_data[review.metadata['IMAGE_FILE']] = {
                'content': review.page_content
            }
            index = scoring[scoring['IMAGE_FILE'] == review.metadata['IMAGE_FILE']].index
            if not index.empty:
                scoring.loc[index, 'total_score'] += 1.123
        print(query_text)


    

    result_df = scoring.sort_values('total_score', ascending=False).head(18).reset_index(drop=True)
    most_similar_title = result_df['TITLE'][0]
    total = "요구사항:이 질문에 기반해서 추천된 작품들이야 질문에 맞춰서 묻는 말에만 관련된 정보로 한국어로 답변을 생성해줘" + "질문:" + query_text + most_similar_title
    total +="추천 창에 출력된 그림 리스트"
    for i in range(6):
        total+= str(i)+":"
        if i == 0:
            total +="가장 크게 출력됨"
        if i == 1:
            total +="오른쪽 위에 출력됨"
        if i == 2:
            total +="오른쪽 중앙에 출력됨"
        if i == 3:
            total +="왼쪽 아래에 출력됨"
        if i == 4:
            total +=" 아래쪽 중앙에 출력됨"
        if i == 5:
            total +="오른쪽 아래에 출력됨"
        total += "제목: " + str(result_df['TITLE'][i]) + "\n"
        if art_data.author != None:
            total += "작가: " + str(result_df['AUTHOR'][i]) + "\n"
        if art_data.technique != None:
            total += "technique: " + str(result_df['TECHNIQUE'][i]) + "\n"
        if art_data.shape != None:
            total += "SHAPE: " + str(result_df['SHAPE'][i]) + "\n"
        if art_data.type != None:
            total += "TYPE: " + str(result_df['TYPE'][i]) + "\n"
        if art_data.school != None:
            total += "학교: " + str(result_df['SCHOOL'][i]) + "\n"
        total += "설명: " + str(result_df['description_final'][i]) + "\n"
    prompt = "질문:" + query_text + total

    # 새로운 API 버전으로 수정
    stream = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # 모델명 수정
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True
    )
    
    answer_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            answer_text += chunk.choices[0].delta.content

    # 결과 저장 경로 수정
    result_path = os.path.join(settings.BASE_DIR, 'chatbot', 'static', 'results')
    os.makedirs(result_path, exist_ok=True)
    
    result_df['제목'] = art_data.title
    try:
        result_df.to_excel(os.path.join(result_path, 'result_datatitle.xlsx'), index=False)
    except Exception as e:
        print(f"Error in saving result_df: {e}")

    return result_df, answer_text

def get_normal_answer(query_text, 프롬프트="당신은 미술 관련 지식을 전달하는 전문가입니다"):
    prompt = 프롬프트+" 질문:" + query_text 

    # 새로운 API 버전으로 수정
    stream = client.chat.completions.create(
        model="gpt-4o-2024-08-06",  # 모델명 수정
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=True
    )
    
    answer_text = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            answer_text += chunk.choices[0].delta.content
    
    return answer_text

    # except Exception as e:
    #     print(f"Error in get_answer: {e}")
    #     return None, "죄송합니다. 응답을 처리하는 중에 오류가 발생했습니다."

    # prompt = "질문:" + query_text + total
    # chat_completion = openai.ChatCompletion.create(
    #     model='gpt-4o-2024-05-13',
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": prompt,
    #         }
    #     ],
    #     stream=True
    # )
    # answer_text = ""
    # for chunk in chat_completion:
    #     if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
    #         content = chunk.choices[0].delta.get("content", "")
    #         if content:
    #             answer_text += content
    # result_df['시발']=art_data.title
    # # Raw 문자열을 사용하는 방법
    # result_df.to_csv(r'C:\Users\Admin\Desktop\result_datatitle.csv', index=False)
    # return result_df, answer_text

@csrf_exempt
def chatanswer(request):
    if request.method == 'POST':
        global 이전질문
        context = {}
        chattext = request.POST.get('chattext', '')
        지시문 = "이것은 추천 서비스에 들어온 연속된 질문입니다. 해당 질문 상황이 이전 질문과 연관되어 추가로 요구한 것일 경우.'이전질문'을 입력해주세요. '추천'과 관련이 있을 경우 '추천'을 입력해주세요. 두 가지 모두 콤마로 구분해 답변하거나 공백으로 제출도 가능합니다."
        분기용정보="이전질문: "+이전질문+" 현재질문: "+chattext
        분기=get_normal_answer(분기용정보,프롬프트=지시문)
        result_df=None
        print(분기)
        
        이전질문_임시=chattext
        if ("이전질문" in 분기) and 이전질문!="":
            # get_answer 함수 호출
            chattext= "이전질문: "+이전질문+" 현재질문: "+chattext

        # if "추천" in 분기:
        #     # get_answer 함수 호출
        result_df, answer = get_answer(chattext)
        #else:
        #    answer = get_normal_answer(chattext)

        
        answer = markdown2.markdown(answer.replace("\n", "  \n")) # !!!!!!! Markdown 변환을 사용하여 답변 가독성 개선

        print(answer)
        print("분기:",분기)

        if result_df is None:
            result_df = pd.DataFrame()  # 빈 데이터프레임으로 초기화
        print("질문:",chattext)
        이전질문=이전질문_임시
        # 데이터프레임을 JSON 형식으로 변환
        result_df_json = result_df.to_json(orient='records')

        context['result_df'] = json.loads(result_df_json)  # JSON 문자열을 딕셔너리로 변환하여 추가
        context['answer'] = answer  # GPT-3 응답

        # 이미지 경로 추가
        image_paths = []
        # image_paths = [f"static/Images/{result_df['IMAGE_FILE'].iloc[0]}"]
        if not result_df.empty and 'IMAGE_FILE' in result_df.columns:
            result_images = result_df.head(6)
            print(result_df['IMAGE_FILE'])
            image_paths = [f"static/art-images/{row['IMAGE_FILE']}" for _, row in result_images.iterrows()]
        else:
            image_paths = []
        context['image_paths'] = image_paths

        return JsonResponse(context, content_type="application/json")
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)