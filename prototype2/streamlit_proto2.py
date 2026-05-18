"""
프로토타입 2 — Enhanced RAG
- DB: seoultech_v1 (ChaTech DB 공유)
- 검색: child_summary similarity_search k=3
- 프롬프트: ChaTech 13규칙 (build_answer)
"""
import os
import sys
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

_ROOT = str(Path(__file__).resolve().parent.parent)
_CHROMA_DIR = str(Path(_ROOT) / "chroma_db")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.web_ui import build_answer

K = 3


@st.cache_resource
def load_db():
    return Chroma(
        collection_name="seoultech_v1",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=_CHROMA_DIR,
    )


db = load_db()

st.set_page_config(page_title="챗봇 B", page_icon="🤖")
st.title("챗봇 B")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    docs = db.similarity_search(query, k=K)
    ctx_str = "\n\n".join(
        f"[문서 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs)
    )

    answer = build_answer(query, ctx_str, [])

    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)
