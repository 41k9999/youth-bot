# app/services/retriever.py
import logging
from typing import Optional
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class ParentChildRetriever:
    """Child 요약으로 벡터 검색 후 Parent 마크다운을 반환하는 RAG 검색기.

    Args:
        vector_db: Chroma 벡터 DB 인스턴스.
        k: 반환할 최대 문서 수.
    """

    def __init__(self, vector_db: Chroma, k: int = 4) -> None:
        self.vector_db = vector_db
        self.k = k

    def retrieve(
        self,
        query: str,
        category: Optional[str] = None,
        location: Optional[str] = None,
    ) -> list[dict]:
        """쿼리로 유사도 검색을 수행하고 Parent 컨텍스트를 포함한 결과를 반환합니다.

        Args:
            query: 사용자 질의 문자열.
            category: 필터링할 카테고리 (장학공지, 청년정책 등). None이면 전체 검색.
            location: 필터링할 지역 (서울 등). None이면 전체 검색.

        Returns:
            child_summary, parent_context, title, url, category 를 담은 dict 리스트.
        """
        where_clause = self._build_filter(category, location)
        logger.info(f"🔍 검색 시작 | 쿼리: '{query}' | 메타데이터 필터: {where_clause}")

        docs = self.vector_db.similarity_search(
            query=query,
            k=self.k,
            filter=where_clause,
        )

        if not docs:
            logger.warning("⚠️ 검색 결과 없음.")
            return []

        results = []
        for doc in docs:
            meta = doc.metadata
            logger.info(
                f"  ✅ 검색 결과: [{meta.get('category', '-')}] "
                f"'{meta.get('title', '-')}' | 위치: {meta.get('location', '-')}"
            )
            results.append({
                "child_summary":  doc.page_content,
                "parent_context": meta.get("parent_context", ""),
                "title":          meta.get("title", ""),
                "url":            meta.get("url", ""),
                "category":       meta.get("category", ""),
                "location":       meta.get("location", ""),
            })

        return results

    def _build_filter(
        self,
        category: Optional[str],
        location: Optional[str],
    ) -> Optional[dict]:
        """메타데이터 필터 조건을 Chroma where-clause 형식으로 변환합니다.

        Args:
            category: 카테고리 필터 값.
            location: 지역 필터 값.

        Returns:
            Chroma where-clause dict. 조건 없으면 None.
        """
        conditions: list[dict] = []

        if category:
            conditions.append({"category": {"$eq": category}})
        if location:
            conditions.append({"location": {"$eq": location}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
