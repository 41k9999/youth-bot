# app/main.py
import os
import logging
from dotenv import load_dotenv

from app.services.scraper import SeoulTechScraper
from app.services.processor import DocumentAIPipeline
from app.services.youth_api import YouthApiIngestor
from app.schema import ProcessedNotice

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

_SCHOOL_PAGES_PER_BOARD = 10   # 카테고리당 최대 10페이지
_YOUTH_PER_REGION       = 200  # 지역당 목표 수집 건수
_YOUTH_DISPLAY_COUNT    = 100  # API 1회 요청 건수


def _url_exists(vector_db: Chroma, url: str) -> bool:
    """ChromaDB에 동일한 URL의 문서가 이미 있는지 확인합니다."""
    result = vector_db.get(where={"url": {"$eq": url}})
    return len(result["ids"]) > 0


def _youth_policy_exists(vector_db: Chroma, biz_id: str, url: str) -> bool:
    """청년정책 중복 여부를 bizId 우선으로 확인합니다.

    URL이 없거나 홈 URL인 경우 bizId(plcyNo)로 체크하여 정책 고유성을 보장합니다.
    """
    home_urls = {"https://www.youthcenter.go.kr/", "https://www.youthcenter.go.kr"}
    if biz_id:
        result = vector_db.get(where={"bizId": {"$eq": biz_id}})
        return len(result["ids"]) > 0
    if url not in home_urls:
        return _url_exists(vector_db, url)
    return False


def _clear_school_notices(vector_db: Chroma) -> None:
    """ChromaDB에서 학교 공지사항(카테고리별)을 모두 삭제하고 스크레이핑 히스토리를 초기화합니다."""
    school_categories = ["대학공지", "학사공지", "장학공지", "대학원공지", "취업공지"]
    total_deleted = 0
    for cat in school_categories:
        result = vector_db.get(where={"category": {"$eq": cat}})
        ids = result.get("ids", [])
        if ids:
            vector_db.delete(ids=ids)
            total_deleted += len(ids)
            logger.info("  🗑️  [%s] %d건 삭제", cat, len(ids))

    history_file = "data/scrape_history.txt"
    if os.path.exists(history_file):
        os.remove(history_file)
        logger.info("  🗑️  스크레이핑 히스토리 초기화 완료")

    logger.info("🗑️  학교 공지 총 %d건 삭제 완료 (강제 갱신)", total_deleted)


def _clear_youth_policies(vector_db: Chroma) -> None:
    """ChromaDB에서 category='청년정책'인 문서를 모두 삭제합니다.

    FORCE_REFRESH_YOUTH=true 시 호출하여 스키마 변경 후 재적재를 보장합니다.
    """
    result = vector_db.get(where={"category": {"$eq": "청년정책"}})
    ids = result.get("ids", [])
    if ids:
        vector_db.delete(ids=ids)
        logger.info("🗑️  기존 청년정책 문서 %d건 삭제 완료 (강제 갱신)", len(ids))
    else:
        logger.info("ℹ️  삭제할 기존 청년정책 문서 없음")


def main() -> None:
    openai_key = os.getenv("OPENAI_API_KEY")
    youth_key = os.getenv("YOUTH_API_KEY")

    if not openai_key:
        logger.error("❌ OPENAI_API_KEY가 존재하지 않습니다. .env를 확인해 주세요.")
        return

    scraper = SeoulTechScraper()
    pipeline = DocumentAIPipeline()
    youth_ingestor = YouthApiIngestor(api_key=youth_key)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_db = Chroma(
        collection_name="seoultech_v1",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    # ================= [파트 A: 서울과기대 공지사항 수집 및 적재] =================
    if os.getenv("FORCE_REFRESH_SCHOOL", "").lower() == "true":
        logger.info("🔄 FORCE_REFRESH_SCHOOL=true — 기존 학교 공지 데이터 초기화 중...")
        _clear_school_notices(vector_db)

    logger.info("📡 서울과기대 공지사항 수집 파이프라인 가동 (게시판당 최대 %d페이지)...",
                _SCHOOL_PAGES_PER_BOARD)
    new_notices = scraper.capture_new_notices(pages_per_board=_SCHOOL_PAGES_PER_BOARD)

    notice_count = 0
    if new_notices:
        for item in new_notices:
            if _url_exists(vector_db, item['url']):
                continue

            try:
                processed_data = pipeline.process(item)

                if processed_data:
                    doc = Document(
                        page_content=processed_data.child_summary,
                        metadata={
                            "id":             processed_data.doc_id,
                            "title":          processed_data.title,
                            "category":       item['category'],
                            "parent_context": processed_data.parent_markdown,
                            "url":            item['url'],
                        }
                    )
                    vector_db.add_documents([doc])
                    notice_count += 1
                    logger.info("📥 학교 공지 적재: [%s] %s", item['category'], processed_data.title[:30])
            except Exception as e:
                logger.error("❌ 공지 처리 에러 (%s): %s", item.get('title', '')[:20], e)

    logger.info("✅ 학교 공지 신규 적재 %d건 완료", notice_count)

    # ================= [파트 B: 온통청년 정책 API 수집 및 적재] =================
    if os.getenv("SKIP_YOUTH", "").lower() == "true":
        logger.info("⏭️  SKIP_YOUTH=true — 청년정책 수집 블록 전체 건너뜀")
        youth_count = 0
    else:
        logger.info("📡 온통청년 API 정책 지역별 수집 시작 (목표: %d건/지역 × 3지역)...",
                    _YOUTH_PER_REGION)

        if os.getenv("FORCE_REFRESH_YOUTH", "").lower() == "true":
            logger.info("🔄 FORCE_REFRESH_YOUTH=true — 기존 청년정책 데이터 초기화 중...")
            _clear_youth_policies(vector_db)

        # 기존 DB의 bizId를 미리 로드 → fetch_by_regions에서 GPT 호출 없이 skip
        existing_biz_ids: set[str] = set()
        try:
            existing = vector_db.get(where={"category": {"$eq": "청년정책"}})
            for meta in existing.get("metadatas", []):
                if meta and meta.get("bizId"):
                    existing_biz_ids.add(meta["bizId"])
            logger.info("  기존 청년정책 bizId %d건 로드 (GPT 중복 호출 방지)", len(existing_biz_ids))
        except Exception as e:
            logger.warning("  기존 bizId 로드 실패 (계속 진행): %s", e)

        try:
            youth_policies_all = youth_ingestor.fetch_by_regions(
                target_per_region=_YOUTH_PER_REGION,
                display_count=_YOUTH_DISPLAY_COUNT,
                skip_ids=existing_biz_ids,
            )
        except Exception as e:
            logger.error("❌ 청년정책 지역별 수집 실패: %s", e)
            youth_policies_all = []

        youth_count = 0
        for policy in youth_policies_all:
            if _youth_policy_exists(vector_db, policy.get("bizId", ""), policy["url"]):
                continue

            try:
                processed_policy = ProcessedNotice(
                    title=policy['title'],
                    source_url=policy['url'],
                    category="청년정책",
                    parent_markdown=policy['parent_markdown'],
                    child_summary=policy['child_summary'],
                    keywords=["청년정책", policy['category'], "지원금"],
                    target_conditions=[]
                )

                doc = Document(
                    page_content=processed_policy.child_summary,
                    metadata={
                        "id":                processed_policy.doc_id,
                        "title":             processed_policy.title,
                        "category":          "청년정책",
                        "location":          policy.get("location", "-"),
                        "bizId":             policy.get("bizId", ""),
                        "parent_context":    processed_policy.parent_markdown,
                        "url":               processed_policy.source_url,
                        "policy_main_cat":   policy.get("policy_main_cat", "-"),
                        "policy_sub_cat":    policy.get("policy_sub_cat", "-"),
                        "policy_keyword":    policy.get("policy_keyword", "-"),
                        "provider_group":    policy.get("provider_group", "-"),
                        "apply_period_type": policy.get("apply_period_type", "-"),
                    }
                )
                vector_db.add_documents([doc])
                youth_count += 1
                logger.info("📥 청년정책 적재: %s", processed_policy.title[:30])
            except Exception as e:
                logger.error("❌ 정책 처리 에러 (%s): %s", policy.get('title', '')[:20], e)

        logger.info("✅ 청년정책 신규 적재 %d건 완료", youth_count)

    # ── 카테고리별 최종 현황 보고 ─────────────────────────────────────────────
    total_result = vector_db.get()
    total_docs = len(total_result["ids"])
    cat_counts: dict[str, int] = {}
    for meta in total_result.get("metadatas", []):
        if meta:
            cat = meta.get("category", "미분류")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

    logger.info("─" * 60)
    logger.info("📊 ChromaDB 카테고리별 최종 현황:")
    for cat, cnt in sorted(cat_counts.items(), key=lambda x: -x[1]):
        logger.info("  %-15s : %4d건", cat, cnt)
    logger.info("─" * 60)
    logger.info("🎯 총 %d건 | 이번 실행 신규: 학교공지 %d건 + 청년정책 %d건",
                total_docs, notice_count, youth_count)
    logger.info("✨ 모든 데이터 수집 및 DB 동기화가 완료되었습니다.")


if __name__ == "__main__":
    main()
