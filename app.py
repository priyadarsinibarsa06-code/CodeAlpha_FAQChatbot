import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?",
    "What is Streamlit?",
    "What is Chatbot?"
]

answers = [
    "AI stands for Artificial Intelligence.",
    "Python is a programming language.",
    "Machine Learning is a subset of AI.",
    "Streamlit is a Python framework for web apps.",
    "A chatbot is a software that responds to user queries."
]

st.set_page_config(page_title="FAQ Chatbot")

st.title("🤖 FAQ Chatbot")

user_input = st.text_input("Ask Your Question")

if user_input:

    vectorizer = TfidfVectorizer()

    vectors = vectorizer.fit_transform(questions + [user_input])

    similarity = cosine_similarity(vectors[-1], vectors[:-1])

    index = similarity.argmax()

    st.success(answers[index])