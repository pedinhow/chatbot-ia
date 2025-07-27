from flask import Flask, render_template, request, jsonify
import json
import nltk
import string
import pickle
import random
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from unidecode import unidecode

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

stemmer = RSLPStemmer()
default_stopwords = set(stopwords.words('portuguese'))
palavras_para_manter = {'o', 'a', 'os', 'as', 'qual', 'quais', 'como', 'onde', 'que', 'é'}
stop_words = default_stopwords - palavras_para_manter


def preprocess(text):
    text = unidecode(text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)


def get_response_from_tag(tag, intents_data):
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            response_text = random.choice(intent['responses'])
            image_file = intent.get('image', None)
            return {"answer": response_text, "image": image_file}
    return {"answer": "Desculpe, algo deu errado na minha resposta.", "image": None}


def chatbot_response(user_input):
    saudacoes_simples = ["oi", "ola", "olá", "e aí", "eae", "bom dia", "boa tarde", "boa noite"]
    if user_input.lower() in saudacoes_simples:
        return get_response_from_tag('boas_vindas', data)

    processed_input = preprocess(user_input)

    if not processed_input:
        return get_response_from_tag('fallback', data)

    vectorized_input = vectorizer.transform([processed_input])
    scores = model.decision_function(vectorized_input)[0]
    max_score = max(scores)

    CONFIDENCE_THRESHOLD = 0.2

    if max_score < CONFIDENCE_THRESHOLD:
        return get_response_from_tag('fallback', data)
    else:
        predicted_tag = model.predict(vectorized_input)[0]
        return get_response_from_tag(predicted_tag, data)


# --- ROTAS DO SITE ---

@app.route("/")
def home():
    return render_template("index.html")


# --- ROTA MODIFICADA ---
@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    response_data = chatbot_response(user_text)
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)