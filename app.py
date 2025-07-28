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

try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('stemmers/rslp')
except nltk.downloader.DownloadError as e:
    print(f"Erro ao encontrar pacotes NLTK, tentando baixar: {e}")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('rslp')

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


def find_suggestions(user_input, intents_data):
    processed_input_tokens = set(preprocess(user_input).split())
    if not processed_input_tokens:
        return []

    intent_scores = {}
    for intent in intents_data['intents']:
        if intent['tag'] in ['fallback', 'boas_vindas', 'agradecimento']:
            continue

        all_patterns_text = " ".join(intent['patterns'])
        processed_patterns_tokens = set(preprocess(all_patterns_text).split())

        common_tokens = processed_input_tokens.intersection(processed_patterns_tokens)
        score = len(common_tokens)

        if score > 0:
            intent_scores[intent['tag']] = score

    if not intent_scores:
        return []

    sorted_intents = sorted(intent_scores.items(), key=lambda item: item[1], reverse=True)
    best_tag = sorted_intents[0][0]

    for intent in intents_data['intents']:
        if intent['tag'] == best_tag:
            return random.sample(intent['patterns'], min(3, len(intent['patterns'])))

    return []


def get_response_from_tag(tag, intents_data):
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            response_text = random.choice(intent['responses'])
            images = intent.get('image', [])
            if not isinstance(images, list):
                images = [images]
            return {"answer": response_text, "images": images, "suggestions": []}

    fallback_response = random.choice([i for i in intents_data['intents'] if i['tag'] == 'fallback'][0]['responses'])
    return {"answer": fallback_response, "images": [], "suggestions": []}


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
        suggestions = find_suggestions(user_input, data)
        fallback_message = random.choice([i for i in data['intents'] if i['tag'] == 'fallback'][0]['responses'])
        return {"answer": fallback_message, "images": [], "suggestions": suggestions}
    else:
        predicted_tag = model.predict(vectorized_input)[0]
        return get_response_from_tag(predicted_tag, data)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    user_text = request.args.get('msg')
    response_data = chatbot_response(user_text)
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True)