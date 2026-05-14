# app/services/rag_chain.py
import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# 나중에 파인튜닝 모델로 교체 시 MODEL_NAME 환경변수만 변경
# Ollama: MODEL_ENDPOINT=http://localhost:11434/v1, MODEL_NAME=llama3
# vLLM:   MODEL_ENDPOINT=http://localhost:8000/v1, MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct
_SYSTEM_PROMPT = """당신은 서울과학기술대학교 학사 및 청년 정책 전문 어드바이저입니다.

[엄격한 답변 규칙]
1. 반드시 아래 [컨텍스트] 안의 정보만 사용하여 답변하십시오.
2. 컨텍스트에 없는 정보는 절대 추측하거나 지어내지 말고, "제공된 정보 내에서는 확인이 어렵습니다"라고 답변하십시오.
3. 답변 마지막에는 반드시 아래 형식으로 출처를 명시하십시오:
   📌 출처: [공지 제목](URL)
   (출처가 여러 개일 경우 줄바꿈으로 모두 나열)"""


def generate_answer(query: str, retrieved_docs: list[dict]) -> str:
    """Llama-3-8B-Instruct로 RAG 답변을 생성합니다.

    파인튜닝된 모델로 교체 시 MODEL_ENDPOINT, MODEL_NAME 환경변수만 수정하면 됩니다.

    Args:
        query: 사용자 질의 문자열.
        retrieved_docs: ParentChildRetriever.retrieve()가 반환한 문서 리스트.

    Returns:
        LLM이 생성한 답변 문자열.
    """
    if not retrieved_docs:
        logger.warning("⚠️ 검색된 문서가 없어 답변을 생성할 수 없습니다.")
        return "제공된 정보 내에서는 확인이 어렵습니다."

    context_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        context_parts.append(
            f"[문서 {i}]\n"
            f"제목: {doc['title']}\n"
            f"URL: {doc['url']}\n"
            f"내용:\n{doc['parent_context']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    user_message = f"[컨텍스트]\n{context_str}\n\n[질문]\n{query}"

    endpoint   = os.getenv("MODEL_ENDPOINT")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    api_key    = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")

    if endpoint:
        logger.info("🤖 LLM 호출 | 커스텀 엔드포인트: %s | 모델: %s", endpoint, model_name)
        client = OpenAI(base_url=endpoint, api_key=api_key or "ollama")
    else:
        logger.info("🤖 LLM 호출 | OpenAI API | 모델: %s", model_name)
        client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content
        logger.info("✅ LLM 답변 생성 완료.")
        return answer
    except Exception as e:
        logger.error(f"❌ LLM 답변 생성 실패: {e}")
        return f"답변 생성 중 오류가 발생했습니다: {e}"
