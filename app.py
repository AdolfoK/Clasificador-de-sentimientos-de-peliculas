import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
import os

# Descargar stopwords 
nltk.download('stopwords')

# Cargar datos (ajusta la ruta del archivo si es necesario)
reviews = []
sentiments = []
for folder in ['pos', 'neg']:
    path = os.path.join('aclImdb', 'train', folder)
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as f:
            reviews.append(f.read())
            sentiments.append(1 if folder == 'pos' else 0)

df = pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Preprocesamiento de texto y vectorización
stop_words = 'english'  # Usar stopwords en inglés predefinidas (o list(stopwords.words('english')))
tfidf = TfidfVectorizer(stop_words=stop_words)
X = tfidf.fit_transform(df['review'])
y = df['sentiment']

# Entrenar modelo 
clf = SVC(kernel='linear')
clf.fit(X, y)

# Interfaz de Streamlit
st.title('Clasificador de Sentimiento de Reseñas de Películas')
review_input = st.text_area('Ingresa una reseña de película:')

if st.button('Predecir'):
    if review_input:
        # Preprocesar la reseña ingresada
        review_input_tfidf = tfidf.transform([review_input])

        # Predecir el sentimiento
        prediction = clf.predict(review_input_tfidf)[0]

        # Mostrar resultado
        if prediction == 1:
            st.success('Esta reseña es positiva 😃')
        else:
            st.error('Esta reseña es negativa 😞')
    else:
        st.warning('Por favor, ingresa una reseña.')
