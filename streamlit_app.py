import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Função de limpeza do texto


def limpar_texto_simplificado(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto


# Título do aplicativo
st.title('Classificador de Texto com F1-Score - PEDRO ERNESTO')

# Upload de arquivo
uploaded_file = st.file_uploader(
    "Escolha o arquivo Excel com as descrições e categorias", type=["xlsx"])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)

    # Limpeza e preparação dos dados
    data['descricao_reclamacao_limpa'] = data['descricao_reclamacao'].apply(
        limpar_texto_simplificado)
    X_train, X_test, y_train, y_test = train_test_split(
        data['descricao_reclamacao_limpa'], data['categoria'], test_size=0.2, random_state=42)

    # Processamento e classificação
    vectorizer = TfidfVectorizer(max_df=0.5, ngram_range=(1, 1))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train_tfidf, y_train)
    predicoes = modelo.predict(X_test_tfidf)
    report = classification_report(y_test, predicoes, output_dict=True)

    # Exibindo resultados
    st.text(classification_report(y_test, predicoes))
    st.write("F1-Score médio (Macro-average):",
             report['macro avg']['f1-score'])
