
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Загрузка данных
df = pd.read_csv("abstracts_for_search.csv")

# Загрузка модели
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# Кэшируем эмбеддинги
@st.cache_data(show_spinner=False)
def compute_embeddings():
    return model.encode(df['abstract'].tolist())

abstract_embeddings = compute_embeddings()

# Интерфейс
st.title("🔍 Семантический поиск по научным статьям для финтеха")
query = st.text_input("Введите запрос (например: диджитализация финансов):")

if query:
    with st.spinner("Ищем релевантные статьи..."):
        query_embedding = model.encode([query])
        scores = cosine_similarity(query_embedding, abstract_embeddings)[0]
        top_indices = scores.argsort()[-3:][::-1]

        st.subheader("📌 Топ-3 релевантные статьи:")
        for idx in top_indices:
            paper = df.iloc[idx]
            score = scores[idx]
            st.markdown(f"### {paper['paper_name']}")
            st.markdown(f"**Релевантность:** {score:.2f}")
            st.markdown(f"**Аннотация:** {paper['abstract']}")
            st.markdown(f"**Польза для финтеха:** {paper['fintech_relevance_summary']}")
            st.markdown("---")
