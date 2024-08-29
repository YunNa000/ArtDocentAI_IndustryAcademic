from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import random
from django.conf import settings

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
    if settings.DEBUG:
        image_directory = os.path.join(settings.STATICFILES_DIRS[0], 'Images')
    else:
        image_directory = os.path.join(settings.STATIC_ROOT, 'Images')

    all_images = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f))]
    random_images = random.sample(all_images, 9) if len(all_images) >= 9 else all_images
    image_urls = [os.path.join(settings.STATIC_URL, 'Images', img) for img in random_images]

    return render(request, "home.html", {'image_urls': json.dumps(image_urls)})

def favorite(request):
    context = {}
    return render(request, "favorite.html", context)

def IntroTeam(request):
    context = {}
    return render(request, "IntroTeam.html", context)

def IntroArtist(request):
    artists = [
        {'name': 'AACHEN, Hans von', 'image': 'artist_images/aachen.jpg', 
         'description': 'AACHEN, Hans von (1552–1615) - 독일 출신의 화가로, 주로 종교적 주제와 야경을 다루었습니다. 그의 작품은 마네리즘과 베네치아 르네상스의 영향을 받았으며, 세밀한 묘사와 생동감 넘치는 색채, 그리고 극적인 구도로 유명합니다. 특히 그의 그림은 인물들의 신비로움과 감정적 풍부함을 잘 담아내었습니다.' },
        {'name': 'AAGAARD, Carl Frederik', 'image': 'artist_images/aagaard.jpg',
         'description': 'AAGAARD, Carl Frederik (1833–1895) - 덴마크 풍경 화가로, 자연의 아름다움을 세밀하게 담아내어 유명합니다. 바르비존 학파와 현실주의 운동의 영향을 받으며, 그의 작품은 자연의 공기와 빛의 변화를 섬세하게 표현합니다. 덴마크의 시골 풍경을 주로 소재로 삼아서, 그의 화풍은 자연의 조용한 아름다움을 감각적으로 전달합니다.'},
        {'name': 'ABADIA, Juan de la', 'image': 'artist_images/abadia.jpg',
         'description': 'ABADIA, Juan de la (15세기 후반-16세기 초반 활동) - 스페인 출신의 고딕 화가로, 종교적 구도와 알타르피스를 주로 다루었습니다. 그의 작품은 세밀하고 다채로운 색채, 복잡한 세부 사항과 함께, 고딕 예술의 전통을 잘 이어받았습니다. 특히 그의 그림은 신성함과 동시에 인간의 존엄성을 강조하는데 주력했습니다.'},
        {'name': 'ABBATE, Niccolò dell', 'image': 'artist_images/abbate.jpg',
         'description': 'ABBATE, Niccolò dell (약 1509–1571) - 이탈리아 출신의 마네리즘 화가로, 신화적이고 애굽적인 테마를 주로 다루었습니다. 델 아바테의 작품은 우아하고 정교한 스타일로 유명하며, 움직임과 감정을 특징으로 합니다. 그의 그림은 신화와 역사의 경계를 넘나들며, 당시의 문화적인 다양성을 반영합니다.'},
        {'name': 'ABBATI, Giuseppe', 'image': 'artist_images/abbati.jpg',
         'description': 'ABBATI, Giuseppe (1836–1868) - 이탈리아 풍경 화가로, 막키아올리 그룹에 속했으며, 자연의 아름다움을 생동감 있게 표현했습니다. 그의 풍경화는 대담한 색채와 감각적인 브러시워크로 자연의 빛과 효과를 포착하며, 당대 이탈리아 자연주의 회화의 중요한 사례로 평가받고 있습니다.'},
        {'name': 'ABBATINI, Guido Ubaldo', 'image': 'artist_images/abbatini.jpg',
         'description': 'ABBATINI, Guido Ubaldo (1602–1656) - 이탈리아 출신의 바로크 화가로, 종교적이고 역사적인 주제를 주로 다뤘습니다. 그의 작품은 강렬한 조명과 그림자의 사용으로 유명하며, 감정과 정신적 깊이를 표현했습니다. 특히 그의 화풍은 바로크 예술의 특성을 잘 대변하며, 신앙과 예술적 표현의 조화를 추구했습니다.'},
        {'name': 'ABILDGAARD, Nicolai', 'image': 'artist_images/abildgaard.jpg',
         'description': 'ABILDGAARD, Nicolai (1743–1809) - 덴마크 출신의 네오클래식 화가로, 역사적이고 신화적인 구도를 그립니다. 그의 작품은 형태의 명확성과 고전적 아름다움을 강조했습니다. 또한 자연과 인간의 상호작용을 중시하며, 그의 화풍은 현실과 이상 사이의 상호작용을 탐구했습니다.'},
        {'name': 'ACHENBACH, Oswald', 'image': 'artist_images/achenbach.jpg',
         'description': 'ACHENBACH, Oswald (1827–1905) - 독일 출신의 풍경 화가로, 뒤셀도르프 학파에 속했으며, 바다와 해변의 아름다움을 특히 잘 표현했습니다. 그의 작품은 빛과 물의 상호작용을 섬세하게 그려내어, 자연의 순수한 아름다움을 감각적으로 전달했습니다. 특히 그의 바다 풍경은 깊은 감성과 자연의 근원적인 아름다움을 잘 담아내었습니다.'},
        {'name': 'ADAM, Albrecht', 'image': 'artist_images/adam.jpg',
         'description': 'ADAM, Albrecht (1786–1862) - 독일 출신의 화가 및 리소그라퍼로, 낭만주의와 사실주의를 결합한 풍경과 역사적 주제를 다뤘습니다. 그의 작품은 문학적 감성과 풍경의 몽환적 표현으로 유명합니다. 특히 그의 풍경화는 자연의 장엄함과 인간의 존재감을 아우르며, 그의 예술적 비전을 잘 대변했습니다.'},
        {'name': 'ADEMOLLO, Luigi', 'image': 'artist_images/ademollo.jpg',
         'description': 'ADEMOLLO, Luigi (1764–1849) - 이탈리아 출신의 네오클래식과 낭만주의 화가로, 역사적이고 신화적인 주제를 다뤘습니다. 그의 작품은 구성의 섬세함과 조화로운 색감이 돋보입니다. 특히 그의 풍경화는 자연과 인간의 조화를 중시하며, 당대의 이탈리아 미학의 상징적인 예술가로 평가받고 있습니다.'},
        {'name': 'ADLER, Salomon', 'image': 'artist_images/adler.jpg',
         'description': 'ADLER, Salomon (1630–1709) - 독일 바로크 화가로, 종교적이고 일상적인 주제를 현실적으로 그렸습니다. 그의 작품은 세밀한 묘사와 감정 표현이 특징입니다. 특히 그의 그림은 바로크 예술의 특성을 잘 대변하며, 그 시대의 문화적 풍경을 잘 보여줍니다.'},
        {'name': 'ADRIAENSSEN, Alexander', 'image': 'artist_images/adriaenssen.jpg',
         'description': 'ADRIAENSSEN, Alexander (1587–1661) - 플랑드르 출신의 바로크 화가로, 여전사(바니타스)를 주제로 한 작품으로 유명합니다. 그의 작품은 물건의 세밀한 묘사와 빛과 그림자의 상징적 사용으로 알려져 있습니다. 특히 그의 그림은 인간 존재의 무상한 플레이에 대한 깊은 사색을 제안합니다.'}
    ]
    return render(request, "IntroArtist.html", {'artists': artists})

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
        if substring.lower() in string.lower():
            score += 0.5
    return score

def calculate_string_score(string_list, substring):
    for string in string_list:
        if substring.lower() in string.lower():
            return 1
    return 0

# GPT-3 API 호출 함수
def call_gpt_api(prompt):
    response = openai.ChatCompletion.create(
        model='gpt-4o-2024-05-13',
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
    search_result = vectorstore.similarity_search(query, k=5)
    
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
    prompt="""
다음 질문을 기반으로 정보를 키워드 분류해 주세요. 한국어 키워드는 꼭 영어로 적절하게 번역해서 넣어 주세요
예시 질문:""예수와 동방박사가 나오는 르네상스 시대 프랑스산 가로형 유화 그림 추천해줘. 기법은 점묘화로 이 모든 값들은 알수 없는 것은 null이고 shape는 Vertical,Horizontal만 있어 유일하게 description만 "Jesus Magi Birth"처럼 이 질문을 묘사할 가장 적절한 키워드들을 띄어쓰기로 분류해 검색할 수 있어
결과를 다음 형식으로 제공해 주세요 절대 주석을 넣거나 영어 번역을 어기면 안됨 코드에 넣을거라:
예시 답변:
{
    "description": "Jesus Magi",
    "author": null,
    "technique": "Oil",
    "shape": "Horizonta",'
    "type": "Still life",
    "nationality": "French",
    "art form": "Renaissance"
}
질문:
"""+query_text
    
    response = openai.ChatCompletion.create(
        model='gpt-4o-2024-05-13',
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )
    
    answer_text = response['choices'][0]['message']['content'].strip()
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
    


def get_answer(query_text):

    # if '추천' not in query_text:
    #     chat_completion = openai.ChatCompletion.create(
    #     model='gpt-4o-2024-05-13',
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": query_text,
    #         }
    #     ],
    #     stream=True
    #     )
    #     answer_text = ""
    #     for chunk in chat_completion:
    #         if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
    #             content = chunk.choices[0].delta.get("content", "")
    #             if content:
    #                 answer_text += content

    #     # 추천 데이터 특징 확인용 지워도됨
    #     #result_df.to_csv(r'C:\Users\Admin\Desktop\result_datatitle.csv', index=False)
    #     return None, answer_text

    scoring = all_df.copy()
    answer_json = get_json_text(query_text)
    if answer_json is None:
        return pd.DataFrame(), "Empty or invalid response from OpenAI API"
    
    art_data = json_to_artdata(json.dumps(answer_json))  # JSON 객체를 문자열로 변환하여 전달

    if art_data is None:
        return pd.DataFrame(), "Failed to parse JSON data"
    
    scoring['total_score'] = 0
    alpha = 1.2

    if art_data.title is not None:
        titles=art_data.title.split()
        scoring['scoreTITLE'] = scoring['TITLE'].apply(lambda x: calculate_string_score1(x, titles))
        scoring['total_score'] += scoring['scoreTITLE'] * alpha

    if art_data.author is not None:
        scoring['scoreAUTHOR'] = scoring['AUTHOR'].apply(lambda x: calculate_string_score([x], art_data.author))
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
    scoring.to_csv(r'C:\Users\Admin\Desktop\result_datatitle_scoreing.csv', index=False)  
    
    if art_data.title is not None:
        review_data = {}
        search_result = vectorstore.similarity_search(art_data.title)
        for i, review in enumerate(search_result):
            review_data[review.metadata['title']] = {
                'author': review.metadata['author'],
                'content': review.page_content
            }
            index = scoring[scoring['TITLE'] == review.metadata['title']].index
            if not index.empty:
                scoring.loc[index, 'total_score'] += 1.1
        print(art_data.title[0])
        review_data2 = {}
        search_result = vectorstore2.similarity_search(art_data.title)
        for i, review in enumerate(search_result):
            review_data2[review.metadata['title']] = {
                'author': review.metadata['author'],
                'content': review.page_content
            }
            index = scoring[scoring['TITLE'] == review.metadata['title']].index
            if not index.empty:
                scoring.loc[index, 'total_score'] = scoring.loc[index, 'total_score'] + 1.21

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
        model='gpt-4o-2024-05-13',
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

    # 추천 데이터 특징 확인용 지워도됨
    #result_df.to_csv(r'C:\Users\Admin\Desktop\result_datatitle.csv', index=False)
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
        # image_paths = [f"static/Images/{result_df['IMAGE_FILE'].iloc[0]}"]
        if not result_df.empty and 'IMAGE_FILE' in result_df.columns:
            image_paths = [f"static/Images/{row['IMAGE_FILE']}" for _, row in result_df.iterrows()]
        else:
            image_paths = []

        context['image_paths'] = image_paths

        return JsonResponse(context, content_type="application/json")
    else:
        return JsonResponse({"error": "Invalid request method."}, status=400)