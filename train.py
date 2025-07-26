import json
import nltk
import string
import pickle
from nltk.stem import RSLPStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
# MUDANÇA 1: Importar o novo modelo
from sklearn.svm import LinearSVC
from unidecode import unidecode

# (O resto do código de setup continua igual...)
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

with open('intents.json', encoding='utf-8') as f:
    data = json.load(f)
corpus = []
labels = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        processed = preprocess(pattern)
        corpus.append(processed)
        labels.append(intent['tag'])

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)

# MUDANÇA 2: Usar o LinearSVC no lugar do MultinomialNB
model = LinearSVC()
model.fit(X, labels)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Treinamento com o novo modelo (LinearSVC) finalizado com sucesso!")