
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import streamlit as st

# Загрузка данных
df = pd.read_csv("abstracts_for_search.csv")

# Загрузка модели
model = SentenceTransformer('all-MiniLM-L6-v2')

# Создание эмбеддингов для всех абстрактов
@st.cache_data(show_spinner=False)
def compute_embeddings():
    return model.encode(df['abstract'].tolist(), convert_to_tensor=True)

abstract_embeddings = compute_embeddings()

# Интерфейс
st.title("🔍 Семантический поиск по научным статьям для финтеха")
query = st.text_input("Введите запрос (например: диджитализация финансов):")

if query:
    with st.spinner("Ищем релевантные статьи..."):
        query_embedding = model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, abstract_embeddings)[0]
        top_results = torch.topk(scores, k=3)

        st.subheader("📌 Топ-3 релевантные статьи:")
        for score, idx in zip(top_results.values, top_results.indices):
            paper = df.iloc[idx]
            st.markdown(f"### {paper['paper_name']}")
            st.markdown(f"**Релевантность:** {score.item():.2f}")
            st.markdown(f"**Аннотация:** {paper['abstract']}")
            st.markdown(f"**Польза для финтеха:** {paper['fintech_relevance_summary']}")
            st.markdown("---")
