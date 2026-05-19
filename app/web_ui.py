# app/web_ui.py
import os
import re
import logging
from datetime import date
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

logger = logging.getLogger(__name__)

# Streamlit Cloud는 st.secrets를 os.environ에 자동 주입하지 않으므로 수동 동기화
try:
    for _k in ("OPENAI_API_KEY", "GROQ_API_KEY", "MODEL_NAME"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

st.set_page_config(
    page_title="SeoulTech 학사 어드바이저",
    page_icon="🎓",
    layout="wide",
)

# resolve()로 절대 경로 보장 — Streamlit Cloud에서 __file__이 상대 경로일 때도 안전
_CHROMA_DIR = str(Path(__file__).resolve().parent.parent / "chroma_db")
_SCHOOL_CATS = {"대학공지", "학사공지", "장학공지", "대학원공지", "취업공지"}
_HOME_URLS = {"https://www.youthcenter.go.kr/", "https://www.youthcenter.go.kr"}
_YOUTH_DETAIL_URL = "https://www.youthcenter.go.kr/youngPlcyUnif/youngPlcyUnifDtl.do?bizId={}"
_MAX_CONTEXT_CHARS = 8_000
_MAX_HISTORY_MSGS = 2
_MAX_DOCS = 4
_MAX_DOC_CHARS = 1_500
_SCORE_THRESHOLD = 0.15  # 이 점수 미만인 문서는 관련 없는 것으로 판단해 제외


@st.cache_resource(show_spinner="ChromaDB 로드 중...")
def load_vector_db() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name="seoultech_v1",
        embedding_function=embeddings,
        persist_directory=_CHROMA_DIR,
    )


def get_db_stats(vector_db: Chroma) -> dict:
    all_docs = vector_db.get(include=["metadatas"])
    metadatas = all_docs.get("metadatas") or []
    breakdown: dict[str, int] = {}
    for m in metadatas:
        cat = (m or {}).get("category", "기타")
        breakdown[cat] = breakdown.get(cat, 0) + 1
    school = sum(v for k, v in breakdown.items() if k in _SCHOOL_CATS)
    policy = breakdown.get("청년정책", 0)
    return {"total": len(metadatas), "school": school, "policy": policy, "breakdown": breakdown}


# ── 만료 판단 ─────────────────────────────────────────────────────────────────
_EXPIRED_KEYWORDS = [
    "(마감)", "(선발완료)", "(종료)", "(접수마감)", "(모집완료)", "마감됨",
    "(지급완료)", "지급완료",
]

# 종료일 전용 패턴 — "까지/마감" 컨텍스트가 있는 날짜만 인식 (시작일·일반 날짜 제외)
# (?:\.?\([^)]*\))? 는 (월)/(화) 등 요일 표기를 선택적으로 처리
# YYYY-MM-DD(요일)까지 / YYYY.MM.DD마감
_END_FULL  = re.compile(
    r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})\.?일?(?:\.?\([^)]*\))?\s*(?:까지|마감)'
)
# 두 날짜가 모두 연도 포함한 범위: YYYY.M.D(요일) ~ YYYY.M.D
_END_RANGE_FULL = re.compile(
    r'\d{4}[.\-년]\s*\d{1,2}[.\-월]\s*\d{1,2}\.?일?(?:\.?\([^)]*\))?\s*~\s*'
    r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})'
)
# 앞만 연도 있는 범위: YYYY.M.D(요일) ~ M.D  (연도를 앞에서 상속)
_END_RANGE_MIXED = re.compile(
    r'(\d{4})[.\-년]\s*\d{1,2}[.\-월]\s*\d{1,2}\.?일?(?:\.?\([^)]*\))?\s*~\s*'
    r'(\d{1,2})[.\-월]\s*(\d{1,2})'
)
# M/D(요일)까지 or ~M/D까지
_END_SHORT = re.compile(
    r'(?:~\s*)?(\d{1,2})[/월]\s*(\d{1,2})일?(?:\.?\([^)]*\))?\s*까지'
)
# 행사·특강 일시 패턴 — 행사 자체가 이미 종료된 경우도 만료로 처리
# "일 시: 2026.5.7." / "일시: 5.7.(목)"
_EVENT_DATE_FULL = re.compile(
    r'일\s*시[^\d\n]{0,10}(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})'
)
_EVENT_DATE_SHORT = re.compile(
    r'일\s*시[^\d\n]{0,10}(\d{1,2})[.\-월]\s*(\d{1,2})'
)


def _doc_is_expired(title: str, child_summary: str) -> bool:
    """제목·요약에서 '종료일' 맥락의 날짜만 추출해 만료 여부를 반환합니다.

    단순 날짜 언급(시작일, 공지일 등)은 무시하고, '까지/마감' 또는
    날짜 범위의 끝 날짜만 종료일로 인식합니다.
    """
    today = date.today()
    if any(kw in title for kw in _EXPIRED_KEYWORDS):
        return True

    text = title + " " + child_summary

    for y, m, d_ in _END_FULL.findall(text):
        try:
            if date(int(y), int(m), int(d_)) < today: return True
        except ValueError: pass

    for y, m, d_ in _END_RANGE_FULL.findall(text):
        try:
            if date(int(y), int(m), int(d_)) < today: return True
        except ValueError: pass

    # YYYY.M.D ~ M.D 형태: 종료 연도를 시작 연도에서 상속
    for y_str, m, d_ in _END_RANGE_MIXED.findall(text):
        try:
            if date(int(y_str), int(m), int(d_)) < today: return True
        except ValueError: pass

    for m, d_ in _END_SHORT.findall(text):
        try:
            if date(today.year, int(m), int(d_)) < today: return True
        except ValueError: pass

    for y, m, d_ in _EVENT_DATE_FULL.findall(text):
        try:
            if date(int(y), int(m), int(d_)) < today: return True
        except ValueError: pass

    for m, d_ in _EVENT_DATE_SHORT.findall(text):
        try:
            if date(today.year, int(m), int(d_)) < today: return True
        except ValueError: pass

    return False


# 제목 키워드 매칭 시 제외할 일반 단어 (너무 많은 문서에 포함됨)
_KEYWORD_STOP_WORDS = {
    "관련", "알려줘", "알려", "정보", "주세요", "알고", "싶어",
    "공지", "공고", "정책", "안내", "내용", "대해", "대학",
    "있나요", "있어", "어떤", "어떻게", "재학생", "학생",
    "서울", "과기대", "뭔지", "무엇", "지원", "모집", "신청",
    "프로그램", "사업", "운영", "교육", "연구", "활용",
    "교수",
    "SeoulTech", "seoultech",
    "알려줘", "설명해줘", "어떻게", "뭐야", "뭔지", "있어", "해줘", "해봐",
}


_FALLBACK_MAX = 6  # 키워드 fallback 최대 반환 문서 수


def _clean_parent(text: str) -> str:
    """LLM 전달 전 컨텍스트 정제: 청년정책 메타데이터, 첨부파일 섹션, 빈 항목 제거."""
    # 청년정책 구조 메타데이터 줄 통째로 제거 (사용자에게 유의미하지 않은 분류 정보)
    # 학력조건/추가자격조건은 "없음" 등 유의미한 값을 가질 수 있으므로 제외
    _meta_keys = (
        r'정책\s*분류|정책분류|지원\s*유형(?:별)?|지원유형(?:별)?|주관\s*기관|주관기관'
        r'|신청\s*방식|신청방식|신청\s*URL|신청URL'
    )
    text = re.sub(rf'(?m)^[^\n]*?(?:{_meta_keys})[^\n]*\n?', '', text)
    # 값이 '-'뿐인 항목 행 제거 (학력조건: -, 추가자격조건: - 등)
    text = re.sub(r'(?m)^[^\n]+:\s*-+\s*$\n?', '', text)
    # 첨부파일 섹션 제거
    text = re.sub(r'#{1,3}\s*첨부파일.*?(?=#{1,3}|\Z)', '', text, flags=re.DOTALL)
    # "신청 URL" 테이블 행 제거
    text = re.sub(r'\|[^\|]*신청\s*URL[^\n]*\n?', '', text)
    # 3줄 이상 연속 빈줄 → 1줄로
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def _rewrite_query_for_search(query: str) -> str:
    """구어체/간접 쿼리를 벡터 검색에 적합한 핵심 명사 키워드로 변환합니다.

    조사·어미가 붙은 구어체 질문이 벡터 유사도 임계값을 통과하지 못할 때
    호출됩니다. llama-3.1-8b-instant(Groq)로 빠르게 처리합니다.

    Args:
        query: 원본 사용자 질의.

    Returns:
        검색에 적합하게 재작성된 쿼리 문자열. 실패 시 원본 반환.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    system = (
        "사용자 질문에서 핵심 검색 키워드만 추출하세요. "
        "조사(이/가/은/는/을/를/에/의/로/도/들 등) 및 어미를 제거하고 "
        "명사 위주 3~5단어로 요약하세요. 설명 없이 키워드만 반환하세요.\n"
        "예시:\n"
        "입력: 대학원생이 참고하면 좋을 정보들 가져와 줘\n"
        "출력: 대학원생 공지 지원 정책\n"
        "입력: 지금 신청가능한 장학금 있어?\n"
        "출력: 장학금 신청 모집\n"
        "입력: 취업 관련해서 요즘 뭐 있어?\n"
        "출력: 취업 채용 공고"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{q}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"q": query}).strip()
        logger.info("🔄 쿼리 재작성: '%s' → '%s'", query, result)
        return result if result else query
    except Exception as e:
        logger.warning("⚠️ 쿼리 재작성 실패 (원본 사용): %s", e)
        return query


def _title_keyword_entries(query: str, vector_db: Chroma) -> list[tuple]:
    """쿼리 키워드를 문서 제목에서 직접 매칭하는 보조 검색.

    child_summary가 부정확하게 생성돼 벡터 검색에서 누락된 문서를 보완한다.
    2글자 이상 고유 키워드(불용어 제외)가 제목에 포함된 경우에만 반환한다.
    """
    raw_words = re.split(r'[\s,·\-\(\)\[\]]+', query)
    # 조사 제거 후 키워드 추출 (예: "가족장학에" → "가족장학")
    _particles = re.compile(r'(에서|으로|한테서|에게서|에게|한테|에서|에|의|을|를|이|가|은|는|도|만|과|와)$')
    words = [_particles.sub('', w) for w in raw_words if len(w) >= 2 and w not in _KEYWORD_STOP_WORDS]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return []

    all_data = vector_db.get(include=["metadatas", "documents"])
    scored: list[tuple] = []   # (match_count, title, url, parent, category, child)
    seen_titles: set[str] = set()

    for meta, content in zip(all_data["metadatas"], all_data["documents"]):
        title = meta.get("title", "")
        if not title or title in seen_titles:
            continue
        match_count = sum(1 for w in words if w in title)
        # 키워드 2개 이상이면 전부 포함된 제목만 (AND), 1개면 포함 여부 (OR)
        required = len(words) if len(words) >= 2 else 1
        if match_count >= required:
            seen_titles.add(title)
            scored.append((
                match_count,
                title,
                meta.get("url", ""),
                meta.get("parent_context", "") or content,
                meta.get("category", ""),
                content,
            ))

    # 매칭 키워드 많은 순 정렬 → 가장 관련성 높은 문서 우선
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        (t, u, p, c, ch)
        for _, t, u, p, c, ch in scored[:_FALLBACK_MAX]
    ]


def retrieve_with_parent(query: str, vector_db: Chroma, k: int = 5) -> tuple[str, list[dict]]:
    """child_summary 유사 검색 → parent_context 추출 (Parent-Child RAG).

    [설계 원칙]
    1. 학교 공지 / 청년정책 각각 검색 → 대량 청년정책에 학교 공지가 묻히는 현상 방지
    2. 유사도 점수 임계값(_SCORE_THRESHOLD) 미만 문서는 무관한 문서로 판단해 제외
       → 참고출처에 엉뚱한 문서가 포함되는 문제 해소
    3. 만료 문서는 먼저 필터링, 유효 문서가 0건이면 만료 문서를 fallback으로 제공
    """
    school_filter = {"category": {"$in": list(_SCHOOL_CATS)}}
    youth_filter  = {"category": {"$eq": "청년정책"}}

    # 점수 포함 검색으로 관련 없는 문서 제거
    school_results = vector_db.similarity_search_with_relevance_scores(
        query, k=k + 3, filter=school_filter
    )
    youth_results = vector_db.similarity_search_with_relevance_scores(
        query, k=k + 3, filter=youth_filter
    )

    school_docs = [doc for doc, score in school_results if score >= _SCORE_THRESHOLD]
    youth_docs  = [doc for doc, score in youth_results  if score >= _SCORE_THRESHOLD]

    # 중복 제거 후 유효/만료 분류
    seen_urls: set[str]   = set()
    seen_titles: set[str] = set()
    valid_pool:   list[tuple] = []
    expired_pool: list[tuple] = []

    for doc in school_docs + youth_docs:
        meta  = doc.metadata or {}
        url   = meta.get("url", "")
        title = meta.get("title", "")
        child = doc.page_content

        if (url and url in seen_urls) or (title and title in seen_titles):
            continue
        if url:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)

        raw_parent = meta.get("parent_context") or child
        # PDF 페이지 목록 오염 데이터 제거 ("N페이지\nN - 문서제목" 패턴)
        raw_parent = re.sub(r'\d+페이지\n\d+ - [^\n]+\n?', '', raw_parent)
        parent   = _clean_parent(raw_parent)[:_MAX_DOC_CHARS]
        category = meta.get("category", "")
        entry    = (title, url, parent, category)

        if _doc_is_expired(title, child):
            expired_pool.append(entry)
        else:
            valid_pool.append(entry)

    # 키워드 직접 매칭 — 항상 실행해 벡터 검색 누락 문서(만료 포함) 보완
    for title, url, parent, category, child in _title_keyword_entries(query, vector_db):
            if (url and url in seen_urls) or (title and title in seen_titles):
                continue
            if url:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
            entry = (title, url, parent, category)
            if _doc_is_expired(title, child):
                expired_pool.insert(0, entry)  # 키워드 매칭 만료 문서 우선
            else:
                valid_pool.append(entry)

    # 구어체 쿼리 재작성 후 재검색 (여전히 결과 없을 때)
    if len(valid_pool) + len(expired_pool) == 0:
            rewritten = _rewrite_query_for_search(query)
            if rewritten and rewritten != query:
                rw_school = vector_db.similarity_search_with_relevance_scores(
                    rewritten, k=k + 3, filter=school_filter
                )
                rw_youth = vector_db.similarity_search_with_relevance_scores(
                    rewritten, k=k + 3, filter=youth_filter
                )
                rw_docs = (
                    [doc for doc, score in rw_school if score >= _SCORE_THRESHOLD]
                    + [doc for doc, score in rw_youth  if score >= _SCORE_THRESHOLD]
                )
                for doc in rw_docs:
                    meta  = doc.metadata or {}
                    url   = meta.get("url", "")
                    title = meta.get("title", "")
                    child = doc.page_content
                    if (url and url in seen_urls) or (title and title in seen_titles):
                        continue
                    if url:
                        seen_urls.add(url)
                    if title:
                        seen_titles.add(title)
                    _rp = meta.get("parent_context") or child
                    _rp = re.sub(r'\d+페이지\n\d+ - [^\n]+\n?', '', _rp)
                    parent   = _clean_parent(_rp)[:_MAX_DOC_CHARS]
                    category = meta.get("category", "")
                    entry    = (title, url, parent, category)
                    if _doc_is_expired(title, child):
                        expired_pool.append(entry)
                    else:
                        valid_pool.append(entry)

    only_expired = len(valid_pool) == 0 and len(expired_pool) > 0
    # 만료 문서가 있으면 valid 문서 수를 줄여 LLM이 만료 문서에 집중하게 함
    valid_cap = max(1, _MAX_DOCS - len(expired_pool[:2])) if expired_pool else _MAX_DOCS
    valid_limited   = valid_pool[:valid_cap]
    expired_limited = expired_pool[:max(1, _MAX_DOCS - len(valid_limited))]

    context_parts: list[str] = []
    sources: list[dict]      = []

    if only_expired:
        context_parts.append(
            "⚠️ [안내] 현재 검색된 문서는 모두 신청 기간이 지난 항목입니다. "
            "참고용으로 제공하오니, 현재 신청 가능 여부를 반드시 원본 링크에서 확인하세요."
        )

    for title, url, parent, category in expired_limited:
        tagged = f"{title} [기간 종료]"
        context_parts.append(f"### [{category}] {tagged}\n\n{parent}")
        sources.append({"title": tagged, "url": url, "category": category})

    for title, url, parent, category in valid_limited:
        context_parts.append(f"### [{category}] {title}\n\n{parent}")
        sources.append({"title": title, "url": url, "category": category})

    return "\n\n---\n\n".join(context_parts), sources


def _filter_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """답변에 실제 언급된 출처만 반환합니다."""
    _GENERIC = {
        '안내', '모집', '신청', '공고', '운영', '지원', '학기', '학년도', '참여', '대상',
        'SeoulTech', 'KIST', '서울과학기술', '프로그램', '교육과정', '학생', '학교',
        '현장실습', '동계계절', '동계', '하계', '계절학기', '참여학생', '모집안내',
    }

    def _extract_keywords(title: str) -> list[str]:
        clean = re.sub(r'\s*\[기간 종료\]', '', title)
        core = re.sub(r'^(\[.*?\]\s*)+', '', clean).strip()     # 카테고리 태그 제거
        core = re.sub(r'^\([^)]*\)\s*', '', core).strip()       # 괄호 접두어 제거
        core = re.sub(r'^[\d\s.\-~]+', '', core).strip()        # 숫자 접두어 제거
        core = re.sub(r'[「」『』【】〔〕《》\*★☆◆■●]', ' ', core)
        tokens = re.split(r'[\s\(\)\[\],·\-_/]+', core)
        return [t for t in tokens if len(t) >= 5 and t not in _GENERIC]

    def _is_cited(title: str) -> bool:
        keywords = _extract_keywords(title)
        return any(kw in answer for kw in keywords)

    cited = [s for s in sources if _is_cited(s.get("title", ""))]
    return cited if cited else sources


def _extract_count_constraint(query: str) -> int | None:
    """쿼리에서 개수 제한(예: '5개만', '3가지') 추출."""
    m = re.search(r'(\d+)\s*(?:개|가지|건|항목)', query)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 20 else None
    return None


def build_answer(query: str, context: str, history: list[dict]) -> str:
    _DETAIL_KEYWORDS = {"자세히", "자세하게", "구체적으로", "상세히", "상세하게"}
    is_detail = any(kw in query for kw in _DETAIL_KEYWORDS)
    today = date.today().strftime("%Y년 %m월 %d일")
    count_limit = _extract_count_constraint(query)

    context = context[:_MAX_CONTEXT_CHARS]
    recent_history = history[-_MAX_HISTORY_MSGS:]
    history_text = ""
    for msg in recent_history:
        role = "사용자" if msg["role"] == "user" else "어드바이저"
        history_text += f"{role}: {msg['content']}\n"

    doc_titles = [
        line.replace("### ", "").strip()
        for line in context.split("\n")
        if line.startswith("### ")
    ]
    titles_list = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(doc_titles))
    has_expired_fallback = "⚠️ [안내]" in context

    # 문서가 있음을 LLM에 명시적으로 전달 — "찾을 수 없다"는 판단을 LLM에 맡기지 않음
    _FORBIDDEN = (
        "다음 표현은 절대 사용 금지: "
        "'현재 제공된 문서에 대한 정보는 없습니다', "
        "'해당 정보를 찾을 수 없습니다', "
        "'관련 정보가 없습니다', "
        "'제공된 문서가 없습니다'. "
        "문서가 제공된 이상 반드시 그 내용을 답변에 활용하라."
    )

    if has_expired_fallback:
        doc_instruction = (
            f"아래 {len(doc_titles)}개 문서가 제공되었습니다(모두 신청 기간 종료). "
            "먼저 '현재 신청 기간이 종료된 정보입니다'라고 안내한 뒤 내용을 참고용으로 소개하세요. "
            "'유사한 공고가 다시 열릴 수 있으니 학교 홈페이지를 확인하세요.'라고 덧붙이세요."
        )
    else:
        count_instruction = (
            f"사용자가 정확히 {count_limit}개를 요청했습니다. "
            f"답변에 항목을 정확히 {count_limit}개만 포함할 것. 더 많거나 더 적게 나열하지 말 것.\n"
            if count_limit else ""
        )
        length_guide = (
            "아래 형식으로 간결하게 답변하세요:\n"
            "- [공고/프로그램명]: [핵심 정보(마감일·자격·금액 등)]\n"
            "제공된 문서에 없는 일반 직무 설명·배경지식·중복 문장은 절대 포함하지 말 것. "
            "공고 내용만 요약할 것."
            if not is_detail else
            "질문에 대해 날짜·자격·방법·서류 등 관련 정보를 항목별로 구체적으로 답변하세요. "
            "제공된 문서에 없는 일반 지식은 포함하지 말 것."
        )
        doc_instruction = (
            f"아래 {len(doc_titles)}개 문서가 제공되었습니다. "
            "반드시 이 문서들의 내용을 바탕으로 답변하세요. "
            f"{count_instruction}"
            f"{length_guide} "
            "동일한 출처(같은 문서)의 내용은 하나의 항목으로만 정리할 것. "
            "같은 문서를 '취업', '현장실습' 등 다른 측면으로 나눠 두 번 언급하지 말 것.\n"
            f"오늘은 {today}입니다. "
            "각 문서의 제목 끝에 '[기간 종료]' 태그가 있으면 그 문서만 마감됨으로 안내하고, "
            "태그가 없는 문서는 내용의 날짜를 확인해 신청 가능/예정으로 안내할 것. "
            "여러 문서를 다룰 때 각 문서를 개별적으로 판단할 것 — 일부가 마감이어도 나머지는 독립적으로 안내. "
            "절대로 '[기간 종료]' 문서를 근거로 '해당 정보가 없습니다'라고 답하지 말 것."
        )

    system_prompt = (
        f"당신은 서울과학기술대학교 학생들을 위한 전문 학사 어드바이저입니다.\n"
        f"오늘은 {today}입니다.\n\n"
        "━━━ 답변 규칙 ━━━\n"
        f"1. {doc_instruction}\n"
        f"2. {_FORBIDDEN}\n"
        "3. [참고 자료]에 없는 내용은 절대 추측하거나 지어내지 말 것.\n"
        "4. [참고 자료]에 없는 정책·기관·프로그램은 언급하지 말 것.\n"
        "4-1. 답변에 http://, https:// URL을 직접 쓰지 말 것. 링크가 필요하면 '참고 출처를 확인하세요'로 대체할 것.\n"
        "5. 각 공지사항 항목의 제목은 [제공된 문서 목록]에 명시된 원본 제목을 그대로 사용할 것. "
        "임의로 축약하거나 다른 이름으로 변경하지 말 것.\n"
        "6. [참고 자료] 내에 '[판독불가]'로 표시된 항목은 해당 내용을 언급하지 말고, "
        "원본 출처에서 직접 확인할 것을 안내할 것.\n\n"
        f"[제공된 문서 목록]\n{titles_list}\n\n"
        "[참고 자료]\n{context}\n\n"
        "[이전 대화]\n{history}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model=os.getenv("MODEL_NAME", "llama-3.1-8b-instant"),
        temperature=0.1,
        max_tokens=700,
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "history": history_text, "question": query})

    # 만료 문서 후처리: 모든 문서가 만료된 경우에만 적용 (has_expired_fallback)
    # — 유효+만료 혼재 시엔 LLM 판단을 신뢰해 유효 문서 텍스트를 손상하지 않음
    if has_expired_fallback:
        answer = re.sub(r'현재 신청[이가]?\s*가능합니다', '신청 기간이 마감되었습니다', answer)
        answer = re.sub(r'신청할 수 있습니다', '신청 기간이 마감되었습니다', answer)
        answer = re.sub(r'현재 신청\s*가능', '신청 기간 종료', answer)

    # URL 후처리: 마크다운 링크 [텍스트](URL) → 텍스트만 남김 (먼저 처리)
    answer = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', answer)
    # 괄호 안 단독 URL: (https://...) → 괄호째 제거 (문장 구조 보존)
    answer = re.sub(r'\s*\(\s*https?://\S+?\s*\)', '', answer)
    # 나머지 bare https?:// URL
    answer = re.sub(r'https?://\S+', '참고 출처를 확인하세요', answer)
    # www. 로 시작하는 주소도 제거
    answer = re.sub(r'\bwww\.\S+', '참고 출처를 확인하세요', answer)

    return answer


# ─── Streamlit UI ─────────────────────────────────────────────────────────────

def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()

    vector_db = load_vector_db()
    stats = get_db_stats(vector_db)

    # ── 사이드바 ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎓 학사 어드바이저")
        st.write(f"**학교 공지** {stats['school']}건 · **청년 정책** {stats['policy']}건")
        st.caption(f"📅 {date.today().strftime('%Y-%m-%d')} 기준")

        st.divider()

        with st.expander("시스템 관리자 정보"):
            st.metric("총 문서 수", f"{stats['total']:,}건")
            st.subheader("카테고리별 현황")
            for cat, cnt in sorted(stats["breakdown"].items(), key=lambda x: -x[1]):
                st.write(f"- **{cat}**: {cnt}건")

        st.divider()

        if st.button("대화 초기화", use_container_width=True):
            st.session_state.messages = []
            st.session_state.sources = {}
            st.rerun()

    # ── 메인 영역 ──────────────────────────────────────────────────────────────
    st.title("🎓 SeoulTech 학사 어드바이저")
    st.caption("서울과학기술대학교 공지사항 및 청년 지원 정책을 질의할 수 있습니다.")

    if stats["total"] == 0:
        st.warning("ChromaDB에 데이터가 없습니다. `python -m app.main`을 먼저 실행하세요.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "sources" not in st.session_state:
        st.session_state.sources = {}

    # 이전 대화 렌더링
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                srcs = st.session_state.sources.get(i, [])
                if srcs:
                    with st.expander("📎 참고 출처"):
                        for src in srcs:
                            if src["url"] and src["url"] not in _HOME_URLS:
                                st.markdown(f"- **[{src['category']}]** [{src['title']}]({src['url']})")
                            else:
                                st.markdown(f"- **[{src['category']}]** {src['title']}")

    # 사용자 입력
    if user_input := st.chat_input("질문을 입력하세요 (예: 서울, 경기권 대학생 지원 정책 알려줘)"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("관련 자료를 검색하는 중..."):
                context, sources = retrieve_with_parent(user_input, vector_db)

            if not sources:
                # 관련 문서가 없으면 LLM 호출 없이 바로 안내
                answer = (
                    "현재 질문과 관련된 정보는 탐색되지 않았습니다.\n\n"
                    "학교 공지사항은 https://www.seoultech.ac.kr 에서, "
                    "청년 정책은 https://www.youthcenter.go.kr 에서 직접 확인해 주세요."
                )
            else:
                with st.spinner("답변을 생성하는 중..."):
                    answer = build_answer(user_input, context, st.session_state.messages[:-1])

            st.markdown(answer)

            # 만료 문서 알림 — 대표 마감일 1회만 표시 (중복 방지)
            expired_sources = [s for s in sources if "[기간 종료]" in s.get("title", "")]
            if expired_sources:
                def _extract_deadline(raw: str) -> str | None:
                    # YYYY.M.D~YYYY.M.D 범위 → 끝 날짜
                    m = re.search(
                        r'\d{4}[.\-년]\s*\d{1,2}[.\-월]\s*\d{1,2}\.?\s*~\s*'
                        r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})', raw)
                    if m:
                        return f"{m.group(1)}년 {int(m.group(2))}월 {int(m.group(3))}일"
                    # YYYY.M.D~M.D 범위 → 끝 날짜 (연도 앞에서 상속)
                    m2 = re.search(
                        r'(\d{4})[.\-년]\s*\d{1,2}[.\-월]\s*\d{1,2}\.?\s*~\s*'
                        r'(\d{1,2})[/\.월]\s*(\d{1,2})', raw)
                    if m2:
                        return f"{m2.group(1)}년 {int(m2.group(2))}월 {int(m2.group(3))}일"
                    # 단일 YYYY.M.D 또는 ~M/D 형식
                    m3 = re.search(r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})', raw)
                    if m3:
                        return f"{m3.group(1)}년 {int(m3.group(2))}월 {int(m3.group(3))}일"
                    m4 = re.search(r'~\s*(\d{1,2})[/\.월]\s*(\d{1,2})', raw)
                    if m4:
                        return f"{date.today().year}년 {int(m4.group(1))}월 {int(m4.group(2))}일"
                    return None

                deadline = None
                for exp_src in expired_sources:
                    deadline = _extract_deadline(exp_src.get("title", ""))
                    if deadline:
                        break

                if deadline:
                    st.info(f"📅 이 공고는 **{deadline}**에 마감된 공고입니다. 유사한 공고가 재개될 수 있으니 [학교 홈페이지](https://www.seoultech.ac.kr)를 확인해 주세요.")
                else:
                    st.info("📅 이 공고는 신청 기간이 종료된 공고입니다. 유사한 공고가 재개될 수 있으니 [학교 홈페이지](https://www.seoultech.ac.kr)를 확인해 주세요.")

            # 청년정책 출처 알림
            has_youth = any(s.get("category") == "청년정책" for s in sources)
            if has_youth:
                st.info(
                    "ℹ️ 청년정책 정보는 개별 공고 페이지로 직접 연결이 어렵습니다. "
                    "[온통청년 포털](https://www.youthcenter.go.kr)에서 정책명을 검색해 주세요.",
                    icon="🔍"
                )

            # 답변에 실제 언급된 출처만 필터링해서 표시
            display_sources = _filter_cited_sources(answer, sources) if sources else []
            if display_sources:
                with st.expander("📎 참고 출처"):
                    for src in display_sources:
                        if src["url"] and src["url"] not in _HOME_URLS:
                            st.markdown(f"- **[{src['category']}]** [{src['title']}]({src['url']})")
                        else:
                            st.markdown(f"- **[{src['category']}]** {src['title']}")

        assistant_idx = len(st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.sources[assistant_idx] = display_sources


if __name__ == "__main__":
    main()
