import json
import nltk
import string
import pickle
import random
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from unidecode import unidecode

# (O resto do código de setup continua igual...)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('stemmers/rslp')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('rslp')
stemmer = RSLPStemmer()
default_stopwords = set(stopwords.words('portuguese'))
palavras_para_manter = {'o', 'a', 'os', 'as', 'qual', 'quais', 'como', 'onde', 'que', 'é'}
stop_words = default_stopwords - palavras_para_manter


def preprocess(text):
    text = unidecode(text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)


def get_response(tag, intents_data):
    for intent in intents_data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Desculpe, algo deu errado na minha resposta."


def chatbot_main():
    print("🤖 Chatbot UFERSA iniciado! Digite 'sair' para encerrar.\n")
    saudacoes_simples = ["oi", "ola", "olá", "e aí", "eae", "bom dia", "boa tarde", "boa noite"]

    while True:
        user_input = input("Você: ")

        if user_input.lower() == 'sair':
            print("Chatbot: Até mais! Bons estudos na UFERSA.")
            break

        if user_input.lower() in saudacoes_simples:
            response = get_response('boas_vindas', data)
            print("Chatbot:", response)
            continue

        processed_input = preprocess(user_input)

        if not processed_input:
            response = get_response('fallback', data)
            print("Chatbot:", response)
            continue

        vectorized_input = vectorizer.transform([processed_input])

        # --- MUDANÇA NA LÓGICA DE CONFIANÇA PARA O LinearSVC ---
        # Obter os scores da decision_function em vez de probabilidades
        scores = model.decision_function(vectorized_input)[0]
        max_score = max(scores)

        # O limiar para o score é diferente. Um score > 0 já indica confiança.
        # Vamos usar 0.2 para ter um pouco mais de certeza.
        CONFIDENCE_THRESHOLD = 0.2

        if max_score < CONFIDENCE_THRESHOLD:
            response = get_response('fallback', data)
        else:
            # A previsão da tag continua igual
            predicted_tag = model.predict(vectorized_input)[0]
            response = get_response(predicted_tag, data)
        # --------------------------------------------------------

        print("Chatbot:", response)


if __name__ == "__main__":
    chatbot_main()