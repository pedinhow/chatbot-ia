import json
import nltk
import string
import pickle
import random
from nltk.stem import PorterStemmer

nltk.download('punkt')

stemmer = PorterStemmer()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in string.punctuation]
    return ' '.join(tokens)

# Carregar intents, modelo e vetor
with open('intents.json') as f:
    data = json.load(f)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def get_response(user_input):
    processed = preprocess(user_input)
    X = vectorizer.transform([processed])
    tag = model.predict(X)[0]

    # Se o modelo não reconhecer bem, usar fallback
    proba = model.predict_proba(X).max()
    if proba < 0.25:
        tag = "fallback"

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Desculpe, não entendi."

print("Chatbot: Olá! Digite 'sair' para encerrar.")
while True:
    user_input = input("Você: ")
    if user_input.lower() == "sair":
        print("Chatbot: Até logo!")
        break

    resposta = get_response(user_input)
    print("Chatbot:", resposta)
