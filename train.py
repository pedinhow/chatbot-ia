import json
import nltk
import string
import pickle
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from unidecode import unidecode

# baixar recursos
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('rslp')

stemmer = RSLPStemmer()
stop_words = set(stopwords.words('portuguese'))

# pré-processamento: acentos, minúsculo, stopwords, stem
def preprocess(text):
    text = unidecode(text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return ' '.join(tokens)

# carregar dados
with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)

corpus = []
labels = []

# processar frases
for intent in data['intents']:
    for pattern in intent['patterns']:
        processed = preprocess(pattern)
        corpus.append(processed)
        labels.append(intent['tag'])

# vetorizar
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)

# treinar modelo
model = MultinomialNB()
model.fit(X, labels)

# salvar arquivos
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Treinamento finalizado com sucesso!")