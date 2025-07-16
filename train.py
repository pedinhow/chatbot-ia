import json
import nltk
import string
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# baixar recursos do NLTK
nltk.download('punkt')

# inicializa stemmer para reduzir palavras às raízes
stemmer = PorterStemmer()

# função para pré-processar texto (tokenizar, stem, remover pontuação)
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [stemmer.stem(t) for t in tokens if t not in string.punctuation]
    return ' '.join(tokens)

# carregar dados
with open('intents.json') as f:
    data = json.load(f)

corpus = []
labels = []

# preparar dados para treinamento
for intent in data['intents']:
    for pattern in intent['patterns']:
        processed = preprocess(pattern)
        corpus.append(processed)
        labels.append(intent['tag'])

# vetorização TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

# modelo Naive Bayes
model = MultinomialNB()
model.fit(X, labels)

# salvando modelo e vetor
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("Treinamento finalizado e modelo salvo!")
