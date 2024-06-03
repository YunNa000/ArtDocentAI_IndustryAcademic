from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

def home(request):
    context={}

    return render(request, "home.html", context)

def favorite(request):
    context={}

    return render(request, "favorite.html", context)

def IntroTeam(request):
    context={}
    
    return render(request, "IntroTeam.html", context)

def IntroArtist(request):
    context={}
    
    return render(request, "IntroArtist.html", context)


# csrf 토큰을 받으면 실행
@csrf_exempt
def chattrain(request):
    context={}

    file = open(f"chatbot\static\intents.json", encoding="UTF-8")
    data = json.load(file)
    
    training_sentences = []
    training_labels = []
    labels = []
    responses = []


    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses.append(intent['responses'])
        
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
            
    num_classes = len(labels)

    lbl_encoder = LabelEncoder()
    lbl_encoder.fit(training_labels)
    training_labels = lbl_encoder.transform(training_labels)

    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

    # Model Training

    #model = Sequential()
    #model.add(Embedding(vocab_size, embedding_dim, input_length=max_len))
    #model.add(GlobalAveragePooling1D())
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(16, activation='relu'))
    #model.add(Dense(num_classes, activation='softmax'))

    # Sequential 모델 생성 시 Input 레이어로 입력 크기를 명시
    model = Sequential([
        Input(shape=(max_len,)), # 입력 길이를 여기에 명시
        Embedding(vocab_size, embedding_dim), # input_length 매개변수를 제거
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', 
                optimizer='adam', metrics=['accuracy'])

    model.summary()

    epochs = 500
    history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs)

    # to save the trained model
    model.save("chatbot/static/chat_model.keras")

    import pickle

    # to save the fitted tokenizer
    with open('chatbot/static/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    # to save the fitted label encoder
    with open('chatbot/static/label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    context["result"] = "Success..."

    return JsonResponse(context, content_type="application/json")


@csrf_exempt
def chatanswer(request):
    context={}

    inp = request.GET['chattext']

    import colorama 
    colorama.init()
    from colorama import Fore, Style, Back

    import random
    import pickle

    file = open(f"./static/intents.json", encoding="UTF-8")
    data = json.load(file)


    # load trained model
    model = keras.models.load_model('static/chat_model.keras')

    # load tokenizer object
    with open('static/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('static/label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    

    print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        
    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            txt1 = np.random.choice(i['responses'])
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,txt1)

    # txt1을 context의 anstext에 저장하여 JsonResponse로 보내고,
    # chathome.html의 sendAsk에서 obj.anstext로 활용
    context['anstext'] = txt1

    return JsonResponse(context, content_type="application/json")
