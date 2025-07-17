import json
import nltk
import string
import pickle
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from unidecode import unidecode

# carregar modelo e vetor
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# carregar intents
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

# baixar recursos necessários
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

stemmer = RSLPStemmer()
stop_words = set(stopwords.words('portuguese'))

# mesma função de preprocessamento usada no train.py
def preprocess(text):
    text = unidecode(text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)

# função para encontrar resposta com base na tag
def get_response(tag):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Desculpe, algo deu errado na minha resposta."

import random

print("🤖 Chatbot UFERSA iniciado! Digite 'sair' para encerrar.\n")

while True:
    user_input = input("Você: ")

    if user_input.lower() == 'sair':
        print("Chatbot: Até mais! Bons estudos na UFERSA.")
        break

    processed = preprocess(user_input)
    vectorized = vectorizer.transform([processed])
    proba = max(model.predict_proba(vectorized)[0])
    tag = model.predict(vectorized)[0]

    # print(f"[DEBUG] Confiança: {proba:.2f} | Tag: {tag}")  # opcional

    if proba < 0.15:
        print("Chatbot: Desculpe, não entendi sua pergunta. Pode reformular?")
    else:
        resposta = get_response(tag)
        print("Chatbot:", resposta)