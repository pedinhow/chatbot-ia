# 🤖 Chatbot de Suporte ao Cliente (Projeto de Inteligência Artificial)

Este é um projeto simples de **Chatbot de Suporte ao Cliente** desenvolvido em Python, utilizando as bibliotecas **NLTK** e **scikit-learn** como parte da disciplina de Inteligência Artificial. O objetivo é simular uma conversa humana, oferecendo respostas automáticas a perguntas frequentes e operações simples relacionadas a UFERSA baseadas no Manual do Aluno.

---

## 🚀 Funcionalidades

- Respostas automáticas para saudações, e perguntas sobre a UFERSA.
- Treinamento com exemplos de perguntas (intents) em formato JSON.
- Uso de processamento de linguagem natural com NLTK.
- Classificação de intenções com modelo Naive Bayes (scikit-learn).
- Respostas aleatórias para cada intenção.
- Fallback inteligente para frases não reconhecidas.

---

## 🧠 Tecnologias utilizadas

- Python 3.x
- [NLTK](https://www.nltk.org/) (Natural Language Toolkit)
- [scikit-learn](https://scikit-learn.org/)
- TF-IDF (vetorização de texto)
- Modelo Naive Bayes (Multinomial)

---

## 🤔 Como utilizar?

1. Instale as dependências:
   
<pre> pip install -r requirements.txt </pre>
   
2. Baixe os dados do NLTK (se ainda não tiver)
No Python Console:

<pre>import nltk
nltk.download('punkt')</pre>

3. Treine o modelo

<pre>python train.py</pre>

4. Execute o chatbot

<pre>python chatbot.py</pre>

---
  
