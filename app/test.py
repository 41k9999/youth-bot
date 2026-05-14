# app/test.py — RAG 검증 전용 스크립트 (스크래핑/API 호출 없음)
import os
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.services.retriever import ParentChildRetriever
from app.services.rag_chain import generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

_DIVIDER = "=" * 60


def run_rag_test(query: str, retriever: ParentChildRetriever) -> None:
    """chroma_db에 저장된 데이터만으로 RAG 답변 품질을 검증합니다.

    Args:
        query: 검증할 질의 문자열.
        retriever: 초기화된 ParentChildRetriever 인스턴스.
    """
    logger.info(_DIVIDER)
    logger.info("❓ 질문: %s", query)
    logger.info(_DIVIDER)

    retrieved = retriever.retrieve(query=query)

    if not retrieved:
        logger.warning("⚠️  검색된 문서가 없습니다. main.py로 데이터를 먼저 수집하세요.")
        return

    logger.info("📄 검색된 문서 %d건:", len(retrieved))
    for i, doc in enumerate(retrieved, 1):
        parent_preview = doc["parent_context"][:120].replace("\n", " ")
        logger.info("  [%d] [%s] %s", i, doc["category"], doc["title"])
        logger.info("      parent_context 미리보기: %s...", parent_preview)

    answer = generate_answer(query=query, retrieved_docs=retrieved)

    logger.info(_DIVIDER)
    logger.info("💬 최종 답변:\n%s", answer)
    logger.info(_DIVIDER)


def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        logger.error("❌ OPENAI_API_KEY가 없습니다. .env를 확인하세요.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(
        collection_name="seoultech_v1",
        embedding_function=embeddings,
        persist_directory="./chroma_db",
    )

    total = vector_db.get()
    logger.info("📦 현재 DB 문서 수: %d건", len(total["ids"]))
    if not total["ids"]:
        logger.error("❌ DB가 비어 있습니다. python -m app.main 을 먼저 실행하세요.")
        return

    retriever = ParentChildRetriever(vector_db=vector_db, k=3)

    test_queries = [
        "화이트바이오산업 전문인력 양성 정책의 신청자격이 어떻게 돼?",
        "항공정비 청년 지원 프로그램의 지원내용과 신청 방법을 알려줘",
        "K-MOOC 수강 대상과 혜택이 뭐야?",
    ]

    for query in test_queries:
        run_rag_test(query, retriever)
        print()


if __name__ == "__main__":
    main()
