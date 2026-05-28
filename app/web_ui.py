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
    page_title="ChaTech",
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
_SCORE_THRESHOLD = 0.20  # 이 점수 미만인 문서는 관련 없는 것으로 판단해 제외
# ↑ 0.25 → 0.20 완화 이유:
#   포괄적 질문("대학원생이 받을 수 있는 정보")은 여러 문서에 두루 유사해 개별 score가 낮아짐
#   → threshold 0.25면 유효 문서가 통째로 탈락하고 만료 문서만 남는 역설 발생
#   _kw_overlap 필터가 2차로 노이즈를 걸러주므로 threshold를 낮춰도 품질 유지 가능
_WEAK_VALID_THRESHOLD = 0.30   # 유효 문서 최대 점수가 이 값 미만이면 '약유효'로 판단
_STRONG_EXPIRED_GAP   = 0.10   # 만료 문서가 유효 문서보다 이 값 이상 높으면 only_expired 전환


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
    "(마감)", "_마감", "(선발완료)", "(종료)", "(접수마감)", "(모집완료)", "마감됨",
    "(지급완료)", "지급완료",
    # 선발 완료 / 합격자 발표 — 이미 진행이 끝난 공고
    "최종합격자", "합격자 발표", "합격자발표", "선발완료", "결과발표",
    # 과거 학기·연도가 제목에 명시된 경우 만료로 처리
    "2025학년도", "2025년도", "2025-1학기", "2025-2학기",
    "25-1학기", "25-2학기", "2025년 하반기", "2025년 상반기",
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
# M/D(요일)까지 or ~M/D까지 or M. D.까지 (점 구분자 포함)
_END_SHORT = re.compile(
    r'(?:~\s*)?(\d{1,2})[/\.월]\s*(\d{1,2})(?:일|\.)?(?:\.?\([^)]*\))?\s*까지'
)
# 행사·특강 일시 패턴 — 행사 자체가 이미 종료된 경우도 만료로 처리
# "일 시: 2026.5.7." / "일시: 5.7.(목)"
_EVENT_DATE_FULL = re.compile(
    r'일\s*시[^\d\n]{0,10}(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})'
)
_EVENT_DATE_SHORT = re.compile(
    r'일\s*시[^\d\n]{0,10}(\d{1,2})[.\-/월]\s*(\d{1,2})'
)
# YYYY년 MM월 DD일(요일) HH시 ~ MM월 DD일 형태 (시각 포함 신청 기간)
_END_RANGE_WITH_TIME = re.compile(
    r'(\d{4})[년\-]\s*(\d{1,2})[월\-]\s*(\d{1,2})일?(?:\.?\([^)]*\))?\s*\d{1,2}시\s*~\s*'
    r'(\d{1,2})[월\-]\s*(\d{1,2})일?'
)
# 채용설명회·행사 날짜 패턴 — "YYYY-MM-DD(요일) HH시~" or "YYYY년 MM월 DD일 HH:MM-" 형태
# 행사 날짜 자체가 과거이면 만료로 처리
# (~\s*\d{1,2}[월\-]) 는 "HH시 ~ MM월" 형태의 신청 범위와 구별하기 위한 negative lookahead
_EVENT_DATETIME_FULL = re.compile(
    r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*(\d{1,2})일?(?:\.?\([^)]*\))?\s*'
    r'\d{1,2}[:시](?!\s*~\s*\d{1,2}[월\-])'
)
# YYYY.M.D(요일) ~ D.(요일) HH:MM까지 — 일자만 있는 종료일 + 시간 포함 (월은 시작일에서 상속)
# 예: "2026. 3. 3.(수) ~ 9.(월) 18:00까지" → 종료일 2026-03-09
# _END_RANGE_MIXED는 "M.D" 형태를 기대하지만, 이 패턴은 "D. HH:MM까지" 형태를 별도 처리
_END_RANGE_DAY_TIME = re.compile(
    r'(\d{4})[.\-년]\s*(\d{1,2})[.\-월]\s*\d{1,2}\.?일?(?:\.?\([^)]*\))?\s*~\s*'
    r'(\d{1,2})\.(?:\s*\([^)]*\))?\s*\d{1,2}:\d{2}\s*까지'
)


def _doc_is_expired(title: str, child_summary: str, parent: str = "") -> bool:
    """제목·요약에서 '종료일' 맥락의 날짜만 추출해 만료 여부를 반환합니다.

    단순 날짜 언급(시작일, 공지일 등)은 무시하고, '까지/마감' 또는
    날짜 범위의 끝 날짜만 종료일로 인식합니다.

    Args:
        title: 문서 제목.
        child_summary: child 요약 텍스트.
        parent: parent_context 원문 (청년정책 API '신청방식: 마감' 감지용).
    """
    today = date.today()
    if any(kw in title for kw in _EXPIRED_KEYWORDS):
        return True

    # 청년정책 API: 신청방식이 "마감"인 경우 (parent_context에만 존재)
    # child_summary에는 "상시 모집"으로 잘못 기재되는 케이스가 있어 parent를 우선 체크
    if parent and re.search(r'신청\s*방식\s*:\s*마감', parent):
        return True

    # title + child_summary 전체에서 검사 (child_summary에 연도가 명시된 경우 포함)
    text = title + " " + child_summary

    # 과거 연도 + 월 패턴 — title뿐 아니라 child_summary 전체 대상
    _past_year_month = re.compile(
        rf'{today.year - 1}년\s*\d{{1,2}}[~～\-]?\d{{0,2}}월'
    )
    if _past_year_month.search(text):
        return True

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

    # 시각 포함 신청 기간: "2025년 11월 3일(월) 9시 ~ 11월 23일(일)" 형태
    for y, _sm, _sd, em, ed in _END_RANGE_WITH_TIME.findall(text):
        try:
            if date(int(y), int(em), int(ed)) < today: return True
        except ValueError: pass

    # 채용설명회·행사 날짜 자체가 과거 — "2026-03-20(금) 10시~" / "2026년 3월 12일 10:00-" 형태
    # "HH시 ~ MM월 DD일" (범위 종료일 있는 경우)는 negative lookahead로 제외
    for y, m, d_ in _EVENT_DATETIME_FULL.findall(text):
        try:
            if date(int(y), int(m), int(d_)) < today: return True
        except ValueError: pass

    # "YYYY.M.D(요일) ~ D.(요일) HH:MM까지" 형태 — 일자만 있는 종료일 + 시간 포함
    # 예: "2026. 3. 3.(수) ~ 9.(월) 18:00까지" → date(2026, 3, 9)
    # 학점교류 공지처럼 신청 기간이 "시작일 ~ 종료일 HH:MM까지" 형태인 경우 처리
    for y, m, d_ in _END_RANGE_DAY_TIME.findall(text):
        try:
            if date(int(y), int(m), int(d_)) < today: return True
        except ValueError: pass

    # 학기 근로/인턴 공고 휴리스틱: VLM이 마감일을 추출하지 못한("기간: 명시 없음") 경우
    # 1학기 근로/인턴 모집은 통상 2~3월 마감 → 4월 이후 반입은 만료로 처리
    # 2학기 근로/인턴 모집은 통상 8~9월 마감 → 10월 이후 반입은 만료로 처리
    # 대상 키워드: 실제로 학기 단위 모집이 이뤄지는 교내근로 유형만 적용
    _SEMESTER_RECRUIT_KW = {"ST근로", "ST인턴", "ST튜터", "교내근로장학생", "교내근로"}
    if any(kw in title for kw in _SEMESTER_RECRUIT_KW) and "기간: 명시 없음" in child_summary:
        if "1학기" in title and today.month >= 4:
            return True
        if "2학기" in title and today.month >= 10:
            return True

    return False


# 제목 키워드 매칭 시 제외할 일반 단어 (너무 많은 문서에 포함됨)
_KEYWORD_STOP_WORDS = {
    "관련", "알려줘", "알려", "정보", "주세요", "알고", "싶어",
    "공지", "공고", "정책", "안내", "내용", "대해", "대학",
    "있나요", "있어", "있는", "있나", "있습니까", "어떤", "어떻게", "재학생", "학생",
    "서울", "과기대", "뭔지", "무엇", "지원", "모집", "신청",
    "프로그램", "사업", "운영", "교육", "연구", "활용",
    "교수",
    "SeoulTech", "seoultech",
    "알려줘", "설명해줘", "어떻게", "뭐야", "뭔지", "있어", "해줘", "해봐",
    "자세하게", "자세히", "상세하게", "상세히", "구체적으로",
    # 학사 쿼리에서 도메인 키워드와 함께 쓰이지만 제목에는 없는 일반 수식어
    "일정", "방법", "기간",
    # 비교·수식 표현: "월세 지원 같은 정책들" → "같은", "종류" 등은 검색 키워드가 아님
    "같은", "같이", "같은거", "같은것", "처럼", "비슷한", "유사한", "종류",
    "있어요", "있습니다", "있습니다만", "혹시", "혹시나",
    # 시간 수식어: "현재 받을 수 있는", "지금 신청 가능한" → "현재", "지금"은 문서 키워드가 아님
    # (만료 감지용 _AVAIL_RE는 query 전체 텍스트를 보므로 stop word 처리해도 오작동 없음)
    "현재", "지금", "요즘", "오늘", "이번",
    # 동사/수식어: "도움되는", "제공받을", "알아볼" 등 포괄적 표현어
    "도움", "도움되", "유용", "필요", "좋은",
    "제공", "제공받", "받을", "받아",
    "알아", "알아봐", "알아볼", "찾아", "찾아봐",
    # 가능성/존재 표현: "신청 가능한", "받을 수 있는" 등
    "가능", "가능한", "가능하",
    # 이용·활용: "이용할 수 있는 프로그램" → "이용"은 검색 키워드 역할을 못 함
    "이용", "활용", "사용",
}


_FALLBACK_MAX = 6  # 키워드 fallback 최대 반환 문서 수


def _clean_parent(text: str) -> str:
    """LLM 전달 전 컨텍스트 정제: 청년정책 메타데이터, 첨부파일 섹션, 빈 항목 제거."""
    # 청년정책 구조 메타데이터 줄 통째로 제거 (사용자에게 유의미하지 않은 분류 정보)
    # 학력조건/추가자격조건은 "없음" 등 유의미한 값을 가질 수 있으므로 제외
    _meta_keys = (
        r'정책\s*분류|정책분류|지원\s*유형(?:별)?|지원유형(?:별)?|주관\s*기관|주관기관'
        r'|신청\s*방식|신청방식|신청\s*URL|신청URL|지역'
    )
    text = re.sub(rf'(?m)^[^\n]*?(?:{_meta_keys})[^\n]*\n?', '', text)
    # 값이 '-'뿐인 항목 행 제거 (학력조건: -, 추가자격조건: - 등)
    text = re.sub(r'(?m)^[^\n]+:\s*-+\s*$\n?', '', text)
    # 첨부파일 섹션 제거
    text = re.sub(r'#{1,3}\s*첨부파일.*?(?=#{1,3}|\Z)', '', text, flags=re.DOTALL)
    # "신청 URL" 테이블 행 제거
    text = re.sub(r'\|[^\|]*신청\s*URL[^\n]*\n?', '', text)
    # 빈 날짜 테이블 행 제거 ("신청 기간 | - - ~ - -" 형태)
    text = re.sub(r'\|[^\|]*신청\s*기간[^\|]*\|\s*[-\s~]+\s*\|', '', text)
    # "변경 및 취소 신청" 기간 행 제거
    # 이유: 학점교류 공지에서 "변경 및 취소 신청: 3.3 ~ 3.9" 항목이
    #       LLM에게 주요 학점교류 신청 기간으로 오독되는 현상 방지
    #       실제 신청기간은 "교류대학별 상이"이고, 변경·취소는 이미 지난 부가 안내임
    text = re.sub(r'(?m)^[^\n]*(?:변경\s*[및·&]\s*취소|취소\s*[및·&]\s*변경)\s*신청[^\n]*\n?', '', text)
    # 게시글 하단 다른 공지 제목 목록 제거 (학교 게시판 페이지 오염)
    # 패턴 1: "---" 구분선 뒤 카테고리 괄호 공지 헤더 나오면 제거 (삼성에스원 형태)
    text = re.sub(r'\n---\s*\n+#{1,3}\s*\[[^\]]*(?:캠퍼스|리크루팅|대학|학사|장학|취업|대학원)[^\]]*\].*', '', text, flags=re.DOTALL)
    # 패턴 2: 본문 중 ## 레벨 카테고리 공지 헤더 (ADTechnology 형태)
    text = re.sub(r'\n##\s+\[(?:캠퍼스|대학|학사|장학|취업|대학원).*', '', text, flags=re.DOTALL)
    # 연속된 --- 구분선 중복 제거
    # 이유: 청년정책 parent_markdown에서 메타데이터 행 제거 후 --- 구분선이 연달아 남는 경우
    #       ex) "신청 방식: 마감" 행 제거 → "---\n---\n---" → "---\n"
    text = re.sub(r'(?m)^---[ \t]*\n(?:---[ \t]*\n)+', '---\n', text)
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
    raw_words = [re.sub(r'[?!.]+$', '', w) for w in raw_words]  # 문장 부호 제거("돼?" → "돼")
    # 조사·접속조사 제거 후 키워드 추출
    # "복수전공이나" → "복수전공", "가족장학에" → "가족장학"
    _particles = re.compile(
        r'(이나|이랑|하고|하며|에서|으로|한테서|에게서|에게|한테|에서|에|의|을|를|이|가|은|는|도|만|과|와)$'
    )
    words = [_particles.sub('', w) for w in raw_words if len(w) >= 2 and w not in _KEYWORD_STOP_WORDS]
    words = [w for w in words if len(w) >= 2]
    if not words:
        return []

    all_data = vector_db.get(include=["metadatas", "documents"])
    scored: list[tuple] = []   # (match_count, title, url, parent, category, child)
    seen_titles: set[str] = set()

    # 한국어 어절 경계 체크: 키워드가 다른 한글 문자에 붙어 있으면 미매칭
    # 예: "수강신청" in "의무수강신청" → 앞에 "무(한글)" → 매칭 제외
    # 예: "수강신청" in "수강신청 확인" → 앞에 공백/없음 → 매칭 포함
    # 예외(서브스트링 허용): "연구실"·"인턴" 등 의미 어근은 복합어 안에 포함돼도 매칭
    # 예: "연구실" in "뇌인공지능연구실", "인턴" in "청년인턴십 공고"
    _SUBSTRING_OK_KEYWORDS = {"연구실", "인턴십", "장학금", "장학", "인턴", "실험실"}

    def _standalone_match(keyword: str, text: str) -> bool:
        # 어절 경계 기준 독립 매칭 (앞뒤에 한글이 없어야 함)
        standalone = r'(?<![가-힣])' + re.escape(keyword) + r'(?![가-힣])'
        if re.search(standalone, text):
            return True
        # 의미 어근 키워드는 복합어 안에 포함돼도 허용 (단순 in 체크)
        if keyword in _SUBSTRING_OK_KEYWORDS:
            return keyword in text
        return False

    for meta, content in zip(all_data["metadatas"], all_data["documents"]):
        title = meta.get("title", "")
        if not title or title in seen_titles:
            continue
        match_count = sum(1 for w in words if _standalone_match(w, title))
        # 키워드 1~2개 → OR 매칭 (1개 이상)
        # 키워드 3개 이상 → 절반 이상 매칭
        if len(words) <= 2:
            required = 1
        else:
            required = max(2, len(words) // 2)
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


def _generate_hyde_query(query: str) -> str:
    """HyDE: 쿼리에 대한 가상의 child_summary를 생성해 벡터 검색 정확도를 향상합니다.

    사용자 쿼리를 직접 임베딩하는 대신, 실제 DB에 저장된 child_summary 형식과
    유사한 가상 문서를 생성한 뒤 그것을 임베딩합니다.
    구어체 쿼리와 DB 문서 간의 벡터 공간 거리를 줄여 관련 문서 검색률을 높입니다.

    Args:
        query: 원본 사용자 질의.

    Returns:
        가상 child_summary 문자열. 생성 실패 또는 결과가 너무 짧으면 원본 쿼리 반환.
    """
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0.1,
    )
    system = (
        "당신은 서울과학기술대학교 공지사항 데이터베이스의 문서 요약 전문가입니다.\n"
        "사용자 질문에 가장 적합한 공지사항이 있다면, 그 문서의 핵심 요약이 어떤 형태일지 "
        "아래 형식으로 4줄 이내로 작성하세요. 실제 존재하는 공고처럼 구체적으로 작성하세요.\n\n"
        "형식:\n"
        "카테고리: [대학공지/학사공지/장학공지/대학원공지/취업공지/청년정책 중 하나]\n"
        "제목: [공고·정책명]\n"
        "기간: [신청 또는 행사 기간]\n"
        "내용: [핵심 조건·대상·금액·일정 등 1~2줄]\n\n"
        "설명 없이 형식에 맞는 내용만 출력하세요."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "{q}"),
    ])
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"q": query}).strip()
        logger.info("🔍 HyDE 생성: '%s' → %d자", query[:30], len(result))
        return result if len(result) >= 20 else query
    except Exception as e:
        logger.warning("⚠️ HyDE 생성 실패 (원본 사용): %s", e)
        return query


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

    # 지역명 감지: 쿼리에 서울/경기 지역명이 명시된 경우 youth 검색에 location 필터 추가
    # → HyDE가 지역 외 방향으로 편향되더라도 해당 지역 정책을 추가 확보
    _SEOUL_TERMS    = {"서울", "서울시", "서울특별시", "서울권"}
    _GYEONGGI_TERMS = {"경기", "경기도", "경기권"}
    _query_tokens = set(re.sub(r'[?!.]+$', '', w) for w in re.split(r'[\s,·\-\(\)\[\]]+', query))
    _has_seoul    = bool(_query_tokens & _SEOUL_TERMS)
    _has_gyeonggi = bool(_query_tokens & _GYEONGGI_TERMS)
    _location_filter: dict | None = None
    if _has_seoul and _has_gyeonggi:
        # 서울+경기 동시 → OR 필터
        _location_filter = {"$and": [
            {"category": {"$eq": "청년정책"}},
            {"$or": [{"location": {"$eq": "서울특별시"}},
                     {"location": {"$eq": "경기도"}}]},
        ]}
    elif _has_seoul:
        _location_filter = {"$and": [{"category": {"$eq": "청년정책"}},
                                     {"location": {"$eq": "서울특별시"}}]}
    elif _has_gyeonggi:
        _location_filter = {"$and": [{"category": {"$eq": "청년정책"}},
                                     {"location": {"$eq": "경기도"}}]}

    # HyDE: 가상 child_summary 생성 → 구어체 쿼리와 DB 문서 간 벡터 거리 단축
    hyde_query = _generate_hyde_query(query)

    # HyDE + 원본 쿼리 병합 검색
    # HyDE만 쓰면 특정 방향으로 편향 → 지역/주제 다양성이 있는 DB에서 관련 문서 누락
    # 원본 쿼리를 병합해 HyDE 편향을 보완하되, 동일 문서는 높은 score로 deduplicate
    def _merge_results(a: list, b: list) -> list:
        seen: dict[str, tuple] = {}
        for doc, score in a + b:
            # title을 우선 uid로 사용. url은 온통청년처럼 여러 문서가 공유하는 경우
            # 동일 url을 가진 모든 문서가 1건으로 합쳐지는 문제를 방지.
            uid = doc.metadata.get("title") or doc.metadata.get("url") or doc.page_content[:60]
            if uid not in seen or score > seen[uid][1]:
                seen[uid] = (doc, score)
        return list(seen.values())

    hyde_school = vector_db.similarity_search_with_relevance_scores(
        hyde_query, k=k * 4, filter=school_filter
    )
    hyde_youth = vector_db.similarity_search_with_relevance_scores(
        hyde_query, k=k * 4, filter=youth_filter
    )

    # HyDE와 원본 쿼리가 다를 때만 원본 병합 (같으면 불필요한 호출 방지)
    if hyde_query != query:
        orig_school = vector_db.similarity_search_with_relevance_scores(
            query, k=k * 4, filter=school_filter
        )
        orig_youth = vector_db.similarity_search_with_relevance_scores(
            query, k=k * 4, filter=youth_filter
        )
        school_results = _merge_results(hyde_school, orig_school)
        youth_results  = _merge_results(hyde_youth,  orig_youth)
    else:
        school_results = hyde_school
        youth_results  = hyde_youth

    # 지역 필터 추가 검색: 서울/경기 지역명이 감지된 경우 해당 location 문서를 추가 확보
    # HyDE가 다른 방향으로 편향됐을 때도 지역 정책을 놓치지 않기 위함
    # _loc_filter_titles: ChromaDB location 메타데이터로 지역 검증된 문서 title 집합
    # → _kw_overlap 필터 & _query_geos 지오 필터에서 이 문서들을 면제 (중복 필터링 방지)
    _loc_filter_titles: set[str] = set()
    if _location_filter:
        loc_results = vector_db.similarity_search_with_relevance_scores(
            query, k=k * 4, filter=_location_filter
        )
        _loc_filter_titles = {doc.metadata.get("title", "") for doc, _ in loc_results}
        youth_results = _merge_results(youth_results, loc_results)

    school_docs_scored = [(doc, score) for doc, score in school_results if score >= _SCORE_THRESHOLD]
    youth_docs_scored  = [(doc, score) for doc, score in youth_results  if score >= _SCORE_THRESHOLD]

    # 중복 제거 후 유효/만료 분류
    # 온통청년 정책은 모두 동일한 랜딩페이지 URL을 공유하므로 URL 중복 제거에서 제외
    # (제외하지 않으면 첫 번째 청년정책만 남고 나머지 86건이 모두 건너뜀)
    _SHARED_LANDING_URLS: set[str] = {
        "https://www.youthcenter.go.kr/",
        "https://www.youthcenter.go.kr",
    }
    seen_urls: set[str]   = set()
    seen_titles: set[str] = set()
    seen_base_titles: set[str] = set()  # "(수정)", "(마감)" 등 접두어 제거한 기본 제목 dedup
    valid_pool:   list[tuple] = []
    expired_pool: list[tuple] = []
    score_map: dict[str, float] = {}  # title → relevance score (약유효 감지용)

    def _base_title(t: str) -> str:
        """제목 앞의 (날짜/상태) 접두어 제거 → 동일 공고 수정판 dedup에 사용."""
        return re.sub(r'^\s*\([^)]*\)\s*', '', t).strip()

    for doc, _rel_score in school_docs_scored + youth_docs_scored:
        meta  = doc.metadata or {}
        url   = meta.get("url", "")
        title = meta.get("title", "")
        child = doc.page_content

        if (url and url in seen_urls) or (title and title in seen_titles):
            continue
        # 같은 공고의 "(수정)", "(마감)" 등 수정판 중복 제거
        _bt = _base_title(title)
        if _bt and _bt in seen_base_titles:
            continue
        # 공유 랜딩 URL(온통청년 등)은 seen_urls에 추가하지 않아 동일 URL 문서 간 중복 제거 방지
        if url and url not in _SHARED_LANDING_URLS:
            seen_urls.add(url)
        if title:
            seen_titles.add(title)
        if _bt:
            seen_base_titles.add(_bt)

        raw_parent = meta.get("parent_context") or child
        # PDF 페이지 목록 오염 데이터 제거 ("N페이지\nN - 문서제목" 패턴)
        # PDF 목차 오염 데이터 제거
        raw_parent = re.sub(r'(?m)^#{0,6}\s*\d+페이지\s*$\n?', '', raw_parent)  # "#### 2페이지"
        raw_parent = re.sub(r'(?m)^-\s*\d+\s*-\s*$\n?', '', raw_parent)         # "- 2 -"
        raw_parent = re.sub(r'(?m)^\d+\s*-\s*[^\n]*$\n?', '', raw_parent)        # "2 - 텍스트"
        raw_parent = re.sub(r'\d+페이지\n\d+ - [^\n]+\n?', '', raw_parent)       # 혼합 형식
        parent   = _clean_parent(raw_parent)[:_MAX_DOC_CHARS]
        category = meta.get("category", "")
        entry    = (title, url, parent, category)
        if title:
            score_map[title] = _rel_score  # 벡터 검색 점수 기록

        if _doc_is_expired(title, child, parent=raw_parent):
            expired_pool.append(entry)
        else:
            valid_pool.append(entry)

    # 키워드 직접 매칭 — 항상 실행해 벡터 검색 누락 문서(만료 포함) 보완
    for title, url, kw_parent, category, child in _title_keyword_entries(query, vector_db):
            if (url and url in seen_urls) or (title and title in seen_titles):
                continue
            if url and url not in _SHARED_LANDING_URLS:
                seen_urls.add(url)
            if title:
                seen_titles.add(title)
            # 키워드 fallback 경로에도 PDF 오염 데이터 제거 적용
            kw_parent = re.sub(r'(?m)^#{0,6}\s*\d+페이지\s*$\n?', '', kw_parent)
            kw_parent = re.sub(r'(?m)^-\s*\d+\s*-\s*$\n?', '', kw_parent)
            kw_parent = re.sub(r'(?m)^\d+\s*-\s*[^\n]*$\n?', '', kw_parent)
            kw_parent = re.sub(r'\d+페이지\n\d+ - [^\n]+\n?', '', kw_parent)
            kw_parent = _clean_parent(kw_parent)[:_MAX_DOC_CHARS]
            # 정제 후 내용이 너무 짧으면 child_summary로 대체 (PDF 미리보기 오염 케이스)
            if len(kw_parent.strip()) < 80:
                kw_parent = child
            entry = (title, url, kw_parent, category)
            if _doc_is_expired(title, child, parent=kw_parent):
                expired_pool.insert(0, entry)  # 키워드 매칭 만료 문서 우선
            else:
                valid_pool.append(entry)

    # 구어체 쿼리 재작성 후 재검색 (여전히 결과 없을 때)
    if len(valid_pool) + len(expired_pool) == 0:
            rewritten = _rewrite_query_for_search(query)
            if rewritten and rewritten != query and rewritten != hyde_query:
                rw_school = vector_db.similarity_search_with_relevance_scores(
                    rewritten, k=k * 4, filter=school_filter
                )
                rw_youth = vector_db.similarity_search_with_relevance_scores(
                    rewritten, k=k * 4, filter=youth_filter
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
                    _rp = re.sub(r'(?m)^#{0,6}\s*\d+페이지\s*$\n?', '', _rp)
                    _rp = re.sub(r'(?m)^-\s*\d+\s*-\s*$\n?', '', _rp)
                    _rp = re.sub(r'(?m)^\d+\s*-\s*[^\n]*$\n?', '', _rp)
                    _rp = re.sub(r'\d+페이지\n\d+ - [^\n]+\n?', '', _rp)
                    parent   = _clean_parent(_rp)[:_MAX_DOC_CHARS]
                    category = meta.get("category", "")
                    entry    = (title, url, parent, category)
                    if _doc_is_expired(title, child, parent=_rp):
                        expired_pool.append(entry)
                    else:
                        valid_pool.append(entry)

    # 쿼리 키워드 추출 후 각 pool을 키워드 overlap 순으로 재정렬
    # ⚠️ 조사 제거를 먼저 해야 "정보를" → "정보" → stop word 필터에서 제거됨
    # 문장 부호 제거: "돼?" → "돼", "있어!" → "있어" (부호가 키워드로 인식되어 _min_overlap 오동작 방지)
    _raw_words = [re.sub(r'[?!.]+$', '', w) for w in re.split(r'[\s,·\-\(\)\[\]]+', query)]
    # "들": 복수형 어미 제거 ("정책들"→"정책", "학생들"→"학생") — 이후 stop word 필터가 처리
    _ptcl = re.compile(r'(이나|이랑|하고|하며|에서|으로|한테서|에게서|에게|한테|에|의|을|를|이|가|은|는|도|만|과|와|들)$')
    _q_kws = [_ptcl.sub('', w) for w in _raw_words if len(w) >= 2]  # 조사 먼저 제거
    _q_kws = [w for w in _q_kws if len(w) >= 2 and w not in _KEYWORD_STOP_WORDS]  # 그 다음 필터

    def _kw_overlap(entry: tuple) -> int:
        """제목 + 본문 앞 500자에서 쿼리 키워드 매칭 수를 반환.

        제목만 보던 기존 방식의 한계:
        - "서울시 청년 월세" → 제목에 "서울시"+"청년"만 있어도 overlap=2로 무관 문서 통과
        - "KIST 현장실습" → 제목에 없고 본문에만 있으면 miss
        본문 앞부분까지 함께 검사하면 두 문제 모두 해소.
        """
        title, _, parent, _ = entry[0], entry[1], entry[2], entry[3]
        search_text = title + " " + parent[:500]
        return sum(1 for w in _q_kws if w in search_text)

    def _title_overlap(entry: tuple) -> int:
        """제목에서만 키워드 매칭 수를 반환 (정렬용 — 제목 매칭이 더 강한 신호)."""
        return sum(1 for w in _q_kws if w in entry[0])

    if _q_kws:
        # 정렬은 제목 매칭 우선, 동점이면 본문 포함 전체 overlap으로 결정
        valid_pool   = sorted(valid_pool,   key=lambda e: (_title_overlap(e), _kw_overlap(e)), reverse=True)
        expired_pool = sorted(expired_pool, key=lambda e: (_title_overlap(e), _kw_overlap(e)), reverse=True)

        # 제목+본문 기준 overlap 필터 — 제목만 볼 때보다 정밀
        # 키워드 5개 이상 → 최소 2개 이상 매칭 필요 (기존 3개 기준보다 완화)
        # 이유: 포괄적 질문("대학원생이 현재 도움되는 정보 알려줘")은 stop word 처리 후에도
        #       수식어가 남아 _q_kws 수가 늘어나는데, 실제 핵심 키워드는 1~2개뿐임
        #       → _min_overlap=2 때문에 유효 문서가 걸러지는 문제 해소
        _min_overlap = 2 if len(_q_kws) >= 5 else 1
        # ChromaDB location 필터로 지역 검증된 문서는 _kw_overlap 임계값 면제
        # 이유: "경기권" 쿼리 → 평택시 문서 본문엔 "경기도"는 있어도 "경기권"이 없어
        #       overlap < 2 로 탈락하지만, location="경기도" 메타데이터로 이미 지역 검증됨
        _kw_matched_valid = [
            e for e in valid_pool
            if _kw_overlap(e) >= _min_overlap or e[0] in _loc_filter_titles
        ]
        if _kw_matched_valid:
            valid_pool = _kw_matched_valid
        elif _q_kws:
            valid_pool = []

        # expired_pool도 동일한 overlap 기준으로 필터
        # 이유: "월세 지원 같은 정책 있어?" → 관련 없는 만료 문서(학자금대출이자, 장학금 등)가
        #       only_expired 케이스에서 그대로 답변에 포함되는 문제 해소
        # fallback 분기:
        #   - 유효 문서가 있을 때: 필터 통과 0개 → expired 숨김 (관련 없는 만료 표시 방지)
        #   - 유효 문서가 없을 때: 필터 통과 0개 → 원본 유지 (lexical mismatch 방어)
        #     예: "복수전공" → DB 표기 "복수·부전공"으로 substring 불일치 → fallback
        _kw_matched_expired = [e for e in expired_pool if _kw_overlap(e) >= _min_overlap]
        if _kw_matched_expired:
            expired_pool = _kw_matched_expired
        elif valid_pool:
            expired_pool = []  # 유효 있고 관련 만료 없음 → 만료 숨김
        else:
            # 유효도 없고 엄격 필터(_min_overlap)도 0개인 경우:
            # overlap >= 1 로 완화하여 최소한의 관련 문서만 fallback
            # 이유: "월세/생활비" 쿼리 → "창업자를 찾습니다"처럼 완전 무관(overlap=0) 문서 제거
            #       "복수전공" 쿼리 → "복수·부전공" 문서는 본문에 "복수전공" 있어 overlap≥1 통과
            _loose_expired = [e for e in expired_pool if _kw_overlap(e) >= 1]
            expired_pool = _loose_expired  # 비어도 OK (무관 문서보다 빈 결과가 나음)

    # 지역 특정 쿼리: 청년정책 문서를 쿼리 지역명으로 필터링 (다른 지역 혼입 방지)
    # "노원구 청년 지원" → 용인시·안산시 등 다른 지역 청년정책 문서 제거
    _GEO_PATTERN = re.compile(r'[가-힣]{1,5}(?:시|구|군|도|권)$')
    _query_geos: set[str] = {w for w in _q_kws if _GEO_PATTERN.search(w)}
    # "서울", "경기" 같이 GEO_PATTERN에 안 걸리는 광역명도 _has_seoul/_has_gyeonggi로 감지됨
    # → 이 경우에도 지오 필터를 강제 적용해야 경기도 문서가 서울 쿼리에 혼입되지 않음
    _apply_geo_filter = bool(_query_geos) or _has_seoul or _has_gyeonggi
    if _apply_geo_filter:
        # 광역 지역어 → 검색 접두어 매핑 (루프 밖에서 한 번만 정의)
        _GEO_PREFIX_MAP = {
            "경기권": "경기", "경기도": "경기",
            "서울권": "서울", "서울시": "서울", "서울특별시": "서울",
        }
        _school_valid = [e for e in valid_pool if e[3] != "청년정책"]
        _youth_valid_geo: list[tuple] = []
        for e in valid_pool:
            if e[3] != "청년정책":
                continue
            # 1) ChromaDB location 필터로 지역 검증된 문서 → 제목 매칭 생략 (이미 지역 확인됨)
            #    "경기권" 쿼리 → 평택시·경기청년 등 제목에 "경기권" 없어도 location="경기도" 확인됨
            if e[0] in _loc_filter_titles:
                _youth_valid_geo.append(e)
                continue
            # 2) 제목에 쿼리 지역명이 직접 포함 ("노원구 지원" → 제목에 "노원구")
            if any(g in e[0] for g in _query_geos):
                _youth_valid_geo.append(e)
                continue
            # 3) 광역 지역어 폴백: 제목+본문 앞 300자에서 지역 접두어 매칭
            #    "경기권" → "경기"로 축약 → 본문에 "경기도", "경기청년" 등이 있으면 통과
            if any(
                _GEO_PREFIX_MAP.get(g, g[:2]) in (e[0] + " " + e[2][:300])
                for g in _query_geos
            ):
                _youth_valid_geo.append(e)
        # 지오 필터 이후 score_map 기반 재정렬
        # 이유: _school_valid + _youth_valid_geo 순서로 단순 연결하면
        #       학교 공지가 앞에 오고 경기도/서울 청년 정책이 _MAX_DOCS 바깥으로 밀림
        #       → score가 높은 경기도 청년 정책(0.38~0.44)이 학교 장학 공지보다 앞에 와야 함
        valid_pool = sorted(
            _school_valid + _youth_valid_geo,
            key=lambda e: score_map.get(e[0], 0.0),
            reverse=True,
        )

        # 만료 pool에도 동일한 지역 필터 적용
        # 이유: "서울권" 쿼리에 통영시(경남) 등 완전히 다른 지역의 만료 청년정책이 혼입되면
        #       _add_to_context=True 처리되어 "⚠️ [안내]" 마커가 삽입됨
        #       → build_answer의 has_expired_fallback이 오발동 → "과거 자료입니다" 잘못 표시
        _school_expired_g = [e for e in expired_pool if e[3] != "청년정책"]
        _youth_expired_geo: list[tuple] = []
        for e in expired_pool:
            if e[3] != "청년정책":
                continue
            if e[0] in _loc_filter_titles:
                _youth_expired_geo.append(e)
                continue
            if any(g in e[0] for g in _query_geos):
                _youth_expired_geo.append(e)
                continue
            if any(
                _GEO_PREFIX_MAP.get(g, g[:2]) in (e[0] + " " + e[2][:300])
                for g in _query_geos
            ):
                _youth_expired_geo.append(e)
        expired_pool = _school_expired_g + _youth_expired_geo

    # 약유효/강만료 감지
    # valid 문서의 최고 관련성 점수가 낮고, expired 문서가 훨씬 관련성 높을 때 → only_expired 전환
    # 조건 추가: valid 문서 중 쿼리 키워드가 제목에 포함된 것이 없어야 함
    # → valid 문서가 쿼리와 키워드 관련성도 없어야 진짜 '무관 문서'로 판단
    # 예: KIST 현장실습(0.258, '휴학' 키워드 없음)이 valid이고 휴학/복학(0.445)이 expired → override
    # 반례: 창업보육센터(0.260, '창업' 키워드 있음)는 override 안 함
    _only_by_count = len(valid_pool) == 0 and len(expired_pool) > 0
    _weak_valid_override = False
    if valid_pool and expired_pool:
        _max_vs = max((score_map.get(t, 0.0) for t, *_ in valid_pool), default=0.0)
        _max_es = max((score_map.get(t, 0.0) for t, *_ in expired_pool), default=0.0)
        # 지역 검증된 유효 문서(loc_filter_titles)는 keyword overlap 없어도 관련성 있음으로 처리
        # 이유: 서울/경기 쿼리에서 지역 정책들은 "서울권", "거주" 등의 exact keyword가
        #       body 앞 500자에 없어도 location 메타데이터로 이미 지역 검증됐으므로
        #       weak_valid_override 오발동 억제 필요
        _valid_has_kw = any(_kw_overlap(e) > 0 or e[0] in _loc_filter_titles for e in valid_pool)
        if _max_vs < _WEAK_VALID_THRESHOLD and (_max_es - _max_vs) >= _STRONG_EXPIRED_GAP and not _valid_has_kw:
            _weak_valid_override = True
            logger.info("약유효/강만료 감지: valid_max=%.3f expired_max=%.3f kw_match=%s → only_expired 전환",
                        _max_vs, _max_es, _valid_has_kw)
    only_expired = _only_by_count or _weak_valid_override

    valid_limited   = valid_pool[:_MAX_DOCS]
    expired_limited = expired_pool[:max(1, _MAX_DOCS - len(valid_limited))]

    context_parts: list[str] = []
    sources: list[dict]      = []

    if only_expired:
        context_parts.append(
            "⚠️ [안내] 현재 검색된 문서는 모두 신청 기간이 지난 항목입니다. "
            "참고용으로 제공하오니, 현재 신청 가능 여부를 반드시 원본 링크에서 확인하세요."
        )

    # 유효 문서만 LLM 컨텍스트에 포함
    _YOUTH_LEADING_TITLE = re.compile(r'^##\s+[^\n]+\n+(?:---\s*\n+)?')

    def _extract_parent_title(raw_parent: str) -> str:
        """parent_context 첫 번째 # 헤딩에서 카테고리 태그 제거 후 반환.
        OCR 오류로 stored title과 parent 제목이 다를 때 _filter_cited_sources 보조용."""
        m = re.match(r'#+\s*(.+)', raw_parent.strip())
        if not m:
            return ""
        return re.sub(r'^(\[.*?\]\s*)+', '', m.group(1)).strip()

    for title, url, parent, category in valid_limited:
        _pctx = parent
        if category == "청년정책":
            # 청년정책 parent_markdown 첫 줄 "## 정책명\n---\n" 중복 제목 제거
            # context header "### [청년정책] 정책명"과 내용이 동일해 LLM 답변에 제목이 두 번 나타나는 현상 방지
            _pctx = _YOUTH_LEADING_TITLE.sub('', _pctx).strip()
        context_parts.append(f"### [{category}] {title}\n\n{_pctx}")
        sources.append({"title": title, "url": url, "category": category, "in_context": True,
                        "parent_title": _extract_parent_title(parent)})

    # 유효 문서들의 제목 키워드 overlap 최대값 — 만료 doc 직접 질문 오발동 방지에 사용
    # ※ _kw_overlap(제목+본문) 대신 _title_overlap(제목만)을 기준으로 비교
    #   이유: "가족 장학금" 쿼리에서 "선원가족 장학사업"(valid)의 본문에 "장학금"이 등장해
    #         _kw_overlap=2가 되면 _has_enough_valid=True로 오판 → "SeoulTech 가족 장학금"(expired)
    #         이 컨텍스트에서 제외되는 버그. 제목 기준으로만 비교하면 "선원가족 장학사업"의
    #         title_overlap=1("가족"만 일치) < expired title_overlap=2("가족"+"장학금") → 올바르게 표시.
    _max_valid_title_overlap = max((_title_overlap(e) for e in valid_limited), default=0)

    # 만료 문서 처리:
    # - only_expired: 전체 내용 LLM 컨텍스트에 포함
    # - 유효 문서와 혼재 시:
    #   - 만료 문서 제목에 쿼리 키워드가 포함된 경우(사용자가 바로 그 문서를 직접 질문)
    #     → 기간 종료 안내와 함께 LLM 컨텍스트에 포함 (출처 표시도)
    #   - 무관한 만료 문서: in_context=False로 sources에만 추가 (출처 표시 제외)
    for title, url, parent, category in expired_limited:
        tagged = f"{title} [기간 종료]"
        # 쿼리 키워드가 만료 문서 제목에 포함 = 사용자가 해당 문서를 직접 물어본 케이스
        # "장학금", "공고" 같은 범용 단어는 _expired_kw_hit 판단에서 제외
        # → "지금 신청할 수 있는 장학금" 같은 일반 쿼리에서 false hit 방지
        _EXPIRED_KW_STOP = {"장학금", "공고", "모집", "안내", "지원", "신청", "정책"}
        _expired_kw_hit = bool(_q_kws) and any(
            kw in title for kw in _q_kws if kw not in _EXPIRED_KW_STOP
        )
        # 이 만료 문서의 제목 기준 overlap — 유효 문서 최대 title_overlap과 비교
        # 예: "가족 장학금" → 유효(선원가족 장학사업 title_overlap=1) < 만료(가족장학금 title_overlap=2) → 만료 표시 ✓
        # 예: "해외 인턴십 있어?" → 유효(인턴십 공고 title_overlap=2) ≥ 만료(해외인턴십 title_overlap=2) → 표시 안 함 ✓
        _this_expired_title_overlap = _title_overlap((title, url, parent, category))
        # 지역 검증된 유효 문서가 있으면 다른 지역 만료 문서를 context에 포함시킬 필요 없음
        # 예: "서울권 거주 대학생" → 서울 유효 정책 4건이 있는데 통영시(경남) 만료 문서를 context에 넣으면
        #     has_direct_expired=True로 인해 LLM이 "신청 기간 종료" 첫 문장 강제 출력 → 혼란 유발
        _valid_has_geo_coverage = any(e[0] in _loc_filter_titles for e in valid_limited)
        _has_enough_valid = (
            (len(valid_limited) >= 1 and _max_valid_title_overlap >= _this_expired_title_overlap)
            or _valid_has_geo_coverage
        )
        _add_to_context = only_expired or (_expired_kw_hit and not _has_enough_valid)
        if only_expired:
            # 전체 만료 케이스: 글로벌 마커("⚠️ [안내] 현재 검색된 문서는 모두")로 표시
            _pctx_exp = parent
            if category == "청년정책":
                _pctx_exp = _YOUTH_LEADING_TITLE.sub('', _pctx_exp).strip()
            context_parts.append(f"### [{category}] {tagged}\n\n{_pctx_exp}")
        elif _expired_kw_hit and not _has_enough_valid:
            # 유효 문서와 혼재하지만 사용자가 직접 질문한 만료 문서:
            # "⚠️ [직접 질문 만료]" 마커로 구분 → build_answer의 has_expired_fallback에 영향 없음
            # ※ LLM이 첫 문장에 기간 종료 안내를 반드시 포함하도록 컨텍스트 앞에 지시문을 박아 강제
            _pctx_exp = parent
            if category == "청년정책":
                _pctx_exp = _YOUTH_LEADING_TITLE.sub('', _pctx_exp).strip()
            context_parts.append(
                "⚠️ [직접 질문 만료] 아래 문서는 신청 기간이 종료된 정보입니다. "
                "답변 첫 문장에 반드시 '이 장학금/공고의 신청 기간은 종료된 정보입니다'라고 안내하세요.\n\n"
                f"### [{category}] {tagged}\n\n"
                f"{_pctx_exp}"
            )
        # in_context 여부를 sources에 기록 — _filter_cited_sources가 컨텍스트 외 만료 문서를
        # 출처로 잘못 노출하는 문제 방지 (같은 키워드로 공통 매칭되는 케이스)
        sources.append({"title": tagged, "url": url, "category": category,
                        "in_context": _add_to_context,
                        "parent_title": _extract_parent_title(parent)})

    return "\n\n---\n\n".join(context_parts), sources


def _filter_cited_sources(answer: str, sources: list[dict]) -> list[dict]:
    """답변에 실제 언급된 출처만 반환합니다."""
    _GENERIC = {
        '안내', '모집', '신청', '공고', '운영', '지원', '학기', '학년도', '참여', '대상',
        'SeoulTech', 'KIST', '서울과학기술', '프로그램', '교육과정', '학생', '학교',
        '현장실습', '동계계절', '동계', '하계', '계절학기', '참여학생', '모집안내',
    }

    def _get_core(title: str) -> str:
        """공통 전처리: [기간 종료] 태그·카테고리 접두어·특수문자 제거."""
        clean = re.sub(r'\s*\[기간 종료\]', '', title)
        core = re.sub(r'^(\[.*?\]\s*)+', '', clean).strip()     # 카테고리 태그 제거
        core = re.sub(r'^\([^)]*\)\s*', '', core).strip()       # 괄호 접두어 제거
        core = re.sub(r'^[\d\s.\-~]+', '', core).strip()        # 숫자 접두어 제거
        core = re.sub(r'[「」『』【】〔〕《》\*★☆◆■●]', ' ', core)
        return core.strip()

    def _extract_keywords(title: str) -> list[str]:
        core = _get_core(title)
        tokens = re.split(r'[\s\(\)\[\],·\-_/]+', core)
        return [t for t in tokens if len(t) >= 5 and t not in _GENERIC]

    def _is_cited(title: str, parent_title: str = "") -> bool:
        ans_nospace = re.sub(r'(?<=[가-힣]) +(?=[가-힣])', '', answer)
        keywords = _extract_keywords(title)
        if keywords:
            # 한국어 공백 정규화: "가족 장학금"(답변) ↔ "가족장학금"(제목 키워드) 매칭
            # LLM이 사용자 질문 표기를 그대로 따라 공백을 넣는 경우 발생
            if any(kw in answer or kw in ans_nospace for kw in keywords):
                return True
        # parent_title 키워드로 재시도
        # OCR 오류로 stored title이 LLM이 사용하는 parent_context 제목과 다른 경우 대비
        # 예: stored="뇌인공지능연구실"(OCR 오류), parent="뇌인지기능연구실"(정확) → parent_title로 매칭
        if parent_title:
            pt_keywords = _extract_keywords(parent_title)
            if pt_keywords and any(kw in answer or kw in ans_nospace for kw in pt_keywords):
                return True
        # 모든 토큰이 5자 미만인 경우 (짧은 한국어 정책명 등):
        # 정제된 전체 제목이 답변에 포함되어야만 cited로 인정.
        # 이렇게 하면 짧은 청년정책 제목도 정확히 매칭되고, 엉뚱한 fallback 방지.
        core = _get_core(title)
        return bool(core) and (core in answer or core in ans_nospace)

    # in_context=True 문서만 citation 대상으로 한정
    # → 컨텍스트에 없는 만료 문서가 공통 키워드("데이터사이언스학과" 등)로 잘못 매칭되는 문제 방지
    context_sources = [s for s in sources if s.get("in_context", True)]
    cited = [s for s in context_sources if _is_cited(s.get("title", ""), s.get("parent_title", ""))]
    _no_info_patterns = (
        "정보가 없습니다", "찾을 수 없습니다", "자세한 정보는 없습니다",
        "관련 정보가 없습니다", "데이터베이스에 없습니다",
        "제공된 문서 목록에 없습니다", "제공된 문서에는 없습니다",
        "목록에 없습니다", "해당 정보는 없습니다", "제공된 문서에 없습니다",
    )
    if not cited:
        if any(p in answer for p in _no_info_patterns):
            return []
        # 정확 매칭 실패 시: 4자 이상 단어 교집합으로 약한 출처 후보 추출
        # → 전혀 무관한 소스가 폴백으로 표시되는 문제 방지
        # 예: 데이터사이언스학과 연구실 답변에 청년정책(평택시 인턴)이 출처로 뜨는 케이스
        _ans_tokens = {
            w for w in re.split(r'[\s\(\)\[\],/·\-_\*#|]+', answer)
            if len(w) >= 4 and w not in _GENERIC and not w.isdigit()
        }
        partial = []
        for s in context_sources:
            _t_all = s.get("title", "") + " " + s.get("parent_title", "")
            _t_tokens = {
                w for w in re.split(r'[\s\(\)\[\],/·\-_\*#|]+', _t_all)
                if len(w) >= 4 and w not in _GENERIC and not w.isdigit()
            }
            if _t_tokens & _ans_tokens:  # 교집합이 있으면 약한 매칭으로 출처 인정
                partial.append(s)
        # 약한 매칭도 없으면 빈 목록 반환 — 잘못된 출처 노출보다 미표시가 낫다
        return partial if partial else []
    return cited


def _extract_count_constraint(query: str) -> int | None:
    """쿼리에서 개수 제한(예: '5개만', '3가지') 추출."""
    m = re.search(r'(\d+)\s*(?:개|가지|건|항목)', query)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 20 else None
    return None


def build_answer(query: str, context: str, history: list[dict]) -> str:
    _ELIGIBILITY_KEYWORDS = {
        "지원할 수 있", "신청할 수 있", "자격이 되", "해당되나", "해당되는지",
        "가능할까", "가능한가", "가능한지", "됩니까", "할 수 있나",
    }
    _DETAIL_KEYWORDS = {
        "자세히", "자세하게", "구체적으로", "상세히", "상세하게",
        "지원 가능", "신청 가능", "될까요",
    } | _ELIGIBILITY_KEYWORDS
    is_detail = any(kw in query for kw in _DETAIL_KEYWORDS)
    is_eligibility = any(kw in query for kw in _ELIGIBILITY_KEYWORDS)
    today = date.today().strftime("%Y년 %m월 %d일")
    count_limit = _extract_count_constraint(query)

    # "지금/현재 신청 가능한 X 있어?" 패턴 감지
    # only_expired 상태에서 이 쿼리가 오면 만료 목록 나열 대신 "없습니다" 안내
    _AVAIL_RE = re.compile(
        r'(?:지금|현재|요즘|이번에?)\s*(?:신청|지원|접수)'
        r'|(?:신청|지원|접수)\s*(?:가능한|할 수 있는|되는|가능하)'
        r'|(?:지금|현재|요즘)\s*.{0,10}(?:있어|있나|있나요|있어요|있습니까)'
    )
    _is_availability_query = bool(_AVAIL_RE.search(query))

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
    # "⚠️ [안내] 현재 검색된 문서는 모두" = only_expired 케이스 (전체 만료)
    # "⚠️ [직접 질문 만료]" = 유효 문서와 혼재하지만 사용자가 직접 물어본 만료 문서
    has_expired_fallback = "⚠️ [안내] 현재 검색된 문서는 모두" in context
    has_direct_expired   = "⚠️ [직접 질문 만료]" in context

    _FORBIDDEN = (
        "절대 금지 사항:\n"
        "① '일반적으로', '보통', '대개' 등 일반 지식을 추가하는 표현 금지.\n"
        "② [참고 자료]에 없는 내용을 만들어 쓰는 것 금지.\n"
        "③ 지역명이 명시된 경우(예: '노원구', '성북구', '서울시' 등 시·구 단위 지역명): "
        "제공된 문서에 해당 지역 정보가 없으면 "
        "'현재 해당 지역 관련 정보가 데이터베이스에 없습니다'라고 솔직하게 안내할 것. "
        "유사한 다른 지역 정보를 마치 질문한 지역의 정보인 것처럼 제시하지 말 것.\n"
        "③-2. 질문의 핵심 주제가 제공된 문서들과 명백히 다를 경우 "
        "(예: '연구장학금·인건비' 물었는데 등록금 대출 문서만 있는 경우, "
        "'지원 가능한 연구실' 물었는데 장학금·행사 문서만 있는 경우), "
        "억지로 연결 짓지 말고 '해당 정보가 데이터베이스에 없습니다'라고 안내할 것. "
        "규칙 7(유효 문서 소개)은 문서가 질문 주제와 실제로 관련 있을 때만 적용됨.\n"
        "④ '저소득층', '휴학생', '편입생', '외국인' 등 자격·대상 조건은 문서 내용에 "
        "암묵적으로 포함될 수 있음. 제목에 해당 단어가 없어도 문서 내용을 꼼꼼히 확인해 "
        "조건 충족 여부를 판단한 뒤 답변할 것. 내용에 근거가 있으면 반드시 활용할 것.\n"
        "⑤ 사용자가 특정 장학금·공고 이름을 언급한 경우: "
        "그 이름이 [제공된 문서 목록]에 없으면 일부 단어가 겹치더라도 "
        "다른 장학금·공고를 마치 같은 것인 양 연결 짓거나 대체해서 설명하지 말 것. "
        "예: '가족장학금'을 물었는데 '선원가족 장학사업'만 있는 경우 → 완전히 다른 장학금이므로 대체 불가. "
        "'가족'처럼 단어가 부분 일치해도 제도·기관이 다르면 별개의 장학금임. "
        "'해당 장학금 정보가 현재 데이터베이스에 없습니다'라고 솔직하게 안내할 것.\n"
        "⑥ 질문과 직접 관련 없는 다른 장학금·공고·정책의 이름을 답변에서 언급하거나 비교 예시로 쓰지 말 것. "
        "예: 가족장학금을 설명할 때 '국가장학금', 'Alliance 장학금' 등을 비교·언급 금지. "
        "질문에 직접 답변하기 위해 필요한 문서만 참조할 것.\n"
        "⑦ 사용자가 '지원받을 수 있는 정보', '혜택', '프로그램'을 물었을 때: "
        "예비군 비고·신고, 병적 제출, 수강신청 변경, 학적 처리 등 학교 생활 행정 의무 절차는 "
        "'지원·혜택'이 아니므로 답변에서 제외할 것.\n"
        f"⑧ 문서에 날짜가 명시된 경우, 오늘({today}) 이전의 날짜는 이미 지난 기간임. "
        "지난 날짜를 '신청할 수 있습니다' 등 현재 가능한 것처럼 제시하지 말 것. "
        "여러 날짜가 있으면 오늘 기준으로 유효한 날짜만 '신청 가능' 표현에 사용할 것."
    )

    # 전체 만료(only_expired) 처리
    # ① 현재 가용 여부 쿼리("지금/현재 신청 가능한 X") → LLM 없이 "없습니다" + 과거 제목 간략 언급
    # ② 절차·방법·정보 쿼리("복수전공 방법", "장학금 자격") → 만료 명시 후 LLM으로 내용 제공
    if has_expired_fallback and _is_availability_query:
        _past_titles: list[str] = []
        for t in doc_titles:
            _clean = re.sub(r'\s*\[기간 종료\]', '', t)
            _clean = re.sub(r'^\[.*?\]\s*', '', _clean).strip()
            if _clean:
                _past_titles.append(_clean)
        _titles_preview = ", ".join(_past_titles[:3])
        if len(_past_titles) > 3:
            _titles_preview += " 등"
        _past_hint = (
            f"\n\n이전에는 **{_titles_preview}** 등의 공고가 있었습니다."
            if _past_titles else ""
        )
        return (
            f"현재 신청 가능한 정보가 없습니다.{_past_hint}\n\n"
            "유사한 공고가 다시 열릴 수 있으니 "
            "[학교 홈페이지](https://www.seoultech.ac.kr) 또는 "
            "[온통청년 포털](https://www.youthcenter.go.kr)을 확인해 주세요."
        )

    # 이하는 ① 유효 문서가 있는 정상 케이스 또는 ② 절차/정보 쿼리 + only_expired 케이스
    if has_expired_fallback:
        # 절차·방법·정보 쿼리 + only_expired: 만료 명시 후 LLM으로 내용 제공
        doc_instruction = (
            f"현재 유효한 정보가 없어 과거 관련 문서 {len(doc_titles)}건을 참고용으로 제공합니다. "
            "반드시 첫 문장을 '지금 시점에 정확히 일치하는 공고는 찾지 못했어요. "
            "아래는 참고가 될 수 있는 과거 자료입니다 😊'로 시작하세요. "
            "이후 질문에 해당하는 내용(방법·자격·절차 등)을 항목별로 간결하게 안내하세요. "
            "마지막에 '더 정확한 최신 정보는 학교 홈페이지나 온통청년 포털에서 확인하시길 권장해요.'로 마무리하세요. "
            "【절대 금지】현재 신청 가능한 것처럼 표현하거나, 관련 없는 내용을 억지로 연결 짓는 것.\n"
            "【포맷 규칙】마크다운 헤더(## 또는 ###)를 사용하지 말 것. 항목은 불릿(-)으로만 나열할 것."
        )
    else:
        count_instruction = (
            f"사용자가 정확히 {count_limit}개를 요청했습니다. "
            f"답변에 항목을 정확히 {count_limit}개만 포함할 것. 더 많거나 더 적게 나열하지 말 것.\n"
            if count_limit else ""
        )
        length_guide = (
            "아래 형식으로 각 항목을 나열하세요:\n"
            "- **공고명** (마감 X.X): 대상과 핵심 내용을 1~2문장으로 간단히 설명.\n"
            "• 상시 모집이면 '(상시)'로 표기할 것.\n"
            "좋은 예: '- **서울청년정책네트워크 하반기 모집** (마감 5.29): 서울 거주 19~39세 청년을 대상으로 청년 정책 참여 활동을 함께할 분들을 모집합니다.'\n"
            "좋은 예(상시): '- **국민취업지원제도** (상시): 취업 취약계층 청년에게 구직 활동을 지원하는 제도입니다.'\n"
            "나쁜 예(절대 금지): 공고명만 나열하고 설명 없음, URL 노출, 마감일 없는 항목에 임의 날짜 생성.\n"
            "제공된 모든 유효 문서를 각각 항목으로 나열할 것 (1~2개만 선별하지 말 것). "
            "자세한 내용은 참고 출처 확인 안내로 대체할 것.\n"
            "【포맷 규칙】마크다운 헤더(## 또는 ###)를 절대 사용하지 말 것. 항목 사이 빈 줄은 최대 1줄."
            if not is_detail else
            (
                "사용자가 지원 자격·가능 여부를 묻고 있습니다. 반드시 아래 순서로 답변하세요:\n"
                "1. '해당 공고의 지원 자격은 [문서에 적힌 내용]입니다.' 형태로 자격 조건을 먼저 인용할 것\n"
                "2. 사용자 조건(학과·학년 등)과 대조해 지원 가능 여부를 판단할 것\n"
                "3. 마감일, 실습 기간, 신청 방법 등 핵심 정보를 추가할 것\n"
                "4. 문서에 자격 조건이 없으면 '자격 조건은 참고 출처에서 확인하세요.'로 안내할 것\n"
                "【절대 금지】'네, 가능합니다.' 한 줄만 쓰는 것. 반드시 근거(자격 조건)를 제시할 것."
                if is_eligibility else
                "질문에 대해 날짜·자격·방법·서류 등 관련 정보를 항목별로 구체적으로 답변하세요. "
                "각 문서의 원본 제목을 그대로 사용하고 다른 문서의 이름으로 대체하지 말 것. "
                "제공된 문서에 없는 일반 지식은 포함하지 말 것."
            )
        )
        expired_rule = (
            "제목 끝에 '[기간 종료]' 태그가 있는 문서 처리 규칙:\n"
            "  - 일반 [기간 종료] 문서: '마감되었습니다'로만 안내. 날짜·내용을 현재 가능한 정보로 제시 금지.\n"
            "  - '⚠️ [직접 질문 만료]' 표시 문서(사용자가 직접 물어본 장학금·공고): "
            "첫 문장에서 반드시 신청 기간이 종료됐음을 밝힌 뒤, "
            "장학금의 성격·신청자격·금액·조건을 참고용으로 설명할 것. "
            "현재 신청 가능한 것처럼 표현하지 말 것.\n"
            "태그 없는 문서만 신청 가능/예정으로 안내할 것."
        )
        doc_instruction = (
            f"아래 {len(doc_titles)}개 문서가 제공되었습니다. "
            "반드시 이 문서들의 내용을 바탕으로 답변하세요. "
            f"{count_instruction}"
            f"{length_guide} "
            "동일한 출처(같은 문서)의 내용은 하나의 항목으로만 정리할 것. "
            "같은 문서를 다른 측면으로 나눠 두 번 언급하지 말 것.\n"
            "같은 제도·프로그램에 대한 여러 공지(예: 대학별 학점교류 안내)가 있으면 "
            "각각 따로 나열하지 말고 하나의 항목으로 통합하여 안내할 것. "
            "대학별·기관별로 기간이 다를 경우 '대학마다 기간이 상이하니 참고 출처를 확인하세요'로 대체할 것.\n"
            f"오늘은 {today}입니다.\n"
            f"{expired_rule}"
        )

    # 직접 질문 만료 문서가 있을 때 첫 문장 강제 지시 (system_prompt 최하단에 배치해 우선순위 확보)
    direct_expired_override = (
        "\n\n[최우선 지시 — 반드시 따를 것] [참고 자료]에 '⚠️ [직접 질문 만료]' 표시 문서가 있습니다. "
        "이 공고/장학금의 신청 기간은 이미 종료되었습니다. "
        "답변의 첫 번째 문장은 반드시 다음 경고문을 그대로 출력하세요: "
        "'⚠️ 현재 이 공고/장학금의 신청 기간은 종료된 정보입니다.' "
        "경고문 다음에 줄바꿈 후, 참고용으로 자격·금액·조건 등을 설명하세요. "
        "경고 없이 바로 내용을 설명하는 것은 절대 금지합니다."
        if has_direct_expired else ""
    )

    system_prompt = (
        f"당신은 서울과학기술대학교 학생들을 위한 전문 학사 어드바이저입니다.\n"
        f"오늘은 {today}입니다.\n\n"
        "━━━ 답변 규칙 ━━━\n"
        f"1. {doc_instruction}\n"
        f"2. {_FORBIDDEN}\n"
        "3. [참고 자료]에 없는 내용은 절대 추측하거나 지어내지 말 것.\n"
        "4. [제공된 문서 목록]에 없는 기업명·기관명·공고명을 답변에 절대 생성하지 말 것.\n"
        "4-1. 답변에 URL·도메인 주소를 직접 쓰지 말 것. 링크가 필요하면 '참고 출처를 확인하세요'로 대체할 것.\n"
        "5. 각 공지사항의 제목은 [제공된 문서 목록]에 명시된 원본 제목을 그대로 사용할 것. "
        "임의로 축약하거나 다른 이름으로 변경하지 말 것.\n"
        "6. [참고 자료] 내에 '[판독불가]'로 표시된 항목은 해당 내용을 언급하지 말 것.\n"
        "7. 제목에 '[기간 종료]' 태그가 없는 유효 문서가 있고 질문 주제와 관련이 있으면 반드시 그 내용을 소개할 것. "
        "자격·대상이 일부 학생에게만 해당되더라도 '(대상: ...)' 조건을 명시하고 안내하세요. "
        "단, 문서가 질문 주제와 명백히 무관하면 절대 금지 사항 ③-2에 따라 '없습니다'로 안내할 것.\n\n"
        f"[제공된 문서 목록]\n{titles_list}\n\n"
        "[참고 자료]\n{context}\n\n"
        "[이전 대화]\n{history}"
        f"{direct_expired_override}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    model_name = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")
    # GROQ_API_KEY, GROQ_API_KEY2~4 순서로 로테이션
    _groq_keys = [
        k for k in [
            os.getenv("GROQ_API_KEY"),
            os.getenv("GROQ_API_KEY2"),
            os.getenv("GROQ_API_KEY3"),
            os.getenv("GROQ_API_KEY4"),
        ] if k
    ]

    answer = None
    last_err: Exception | None = None
    for _key_idx, _api_key in enumerate(_groq_keys):
        llm = ChatGroq(
            api_key=_api_key,
            model=model_name,
            temperature=0.1,
            max_tokens=700,
            frequency_penalty=0.3,
        )
        chain = prompt | llm | StrOutputParser()
        try:
            answer = chain.invoke({"context": context, "history": history_text, "question": query})
            if _key_idx > 0:
                logger.info("Groq key rotation: key #%d 사용", _key_idx + 1)
            break
        except Exception as e:
            err_str = str(e).lower()
            if "rate_limit" in err_str or "429" in err_str or "rate limit" in err_str:
                logger.warning("Groq key #%d rate limit → 다음 키로 전환 (%s)", _key_idx + 1, e)
                last_err = e
                continue
            logger.error("LLM invoke error (key #%d): %s", _key_idx + 1, e, exc_info=True)
            return "⚠️ 답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

    if answer is None:
        logger.warning("모든 Groq API 키 rate limit 소진: %s", last_err)
        return (
            "⚠️ 현재 모든 API 키의 요청량 한도에 도달했습니다.\n"
            "잠시 후 다시 시도해 주세요. "
            "문제가 지속되면 학교 홈페이지(seoultech.ac.kr) 또는 온통청년 포털(youthcenter.go.kr)을 직접 확인해 주세요."
        )

    # has_expired_fallback 후처리: 비가용 쿼리에서 LLM이 "과거 자료" 안내를 무시한 경우 강제 교정
    # 예: "해외파견 정보 찾아줘" → LLM이 "현재 지원 가능한..." 으로 시작하는 경우
    if has_expired_fallback and not _is_availability_query:
        _correct_kws = ["찾지 못했어요", "찾지 못했습니다", "과거 자료", "지금 시점", "정확히 일치"]
        if not any(kw in answer[:120] for kw in _correct_kws):
            answer = (
                "지금 시점에 정확히 일치하는 공고는 찾지 못했어요. "
                "아래는 참고가 될 수 있는 과거 자료입니다 😊\n\n"
            ) + answer

    # 직접 질문 만료 후처리: 항상 ⚠️ 배너를 맨 앞에 보장
    if has_direct_expired:
        _expire_banner = "⚠️ **현재 이 공고/장학금의 신청 기간은 종료된 정보입니다.**"
        # LLM이 이미 경고를 첫 줄에 넣었으면 중복 방지, 아니면 강제 삽입
        _first_line = answer.split('\n')[0]
        _already_warned = any(kw in _first_line for kw in ["종료", "마감", "신청 불가", "기간이 지", "⚠️"])
        if not _already_warned:
            answer = _expire_banner + "\n\n" + answer
        else:
            # 첫 줄에 ⚠️ 이모지가 없으면 배너 아이콘만 보강
            if "⚠️" not in _first_line:
                answer = "⚠️ " + answer

    # 마크다운 헤더(## / ###) → 볼드체 변환 (LLM이 포맷 규칙을 무시한 경우 방어)
    # ### 제목 → **제목**, ## 제목 → **제목**
    answer = re.sub(r'(?m)^#{2,3}\s+(.+)$', r'**\1**', answer)
    # 카테고리 대괄호 태그 제거
    answer = re.sub(r'\[(?:대학|학사|장학|대학원|취업)공지\]\s*', '', answer)
    answer = re.sub(r'\[캠퍼스\s*리쿠르팅\]\s*', '', answer)
    answer = re.sub(r'\[전체대학원\]\s*', '', answer)
    answer = re.sub(r'\[[^\]]{2,10}\]\s*(?=\d{4}|[가-힣])', '', answer)
    # 이메일 주소 제거 — ac.kr, co.kr 같은 다단계 TLD 포함
    answer = re.sub(r'\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b', '참고 출처를 확인하세요', answer)
    # 이메일 regex 부분 매칭 후 남은 ".kr", ".com" 등 고아 TLD 정리
    answer = re.sub(r'(참고 출처를 확인하세요)\.[a-zA-Z]{2,4}\b', r'\1', answer)

    # 청년정책 메타데이터 줄 답변에서 제거 (컨텍스트에서 복사된 경우 방지)
    answer = re.sub(
        r'(?m)^[^\n]*?(?:정책\s*분류|지원\s*유형(?:별)?|주관\s*기관|신청\s*방식|신청\s*URL|지역)[^\n]*\n?',
        '', answer
    )
    # 값이 '-'뿐인 항목 행 제거 (예: 학력 조건: -, 추가 자격 조건: -)
    answer = re.sub(r'(?m)^[^\n]+:\s*-+\s*$\n?', '', answer)
    answer = re.sub(r'\n{3,}', '\n\n', answer)  # 빈 줄 정리

    # URL 후처리: 마크다운 링크 [텍스트](URL) → 텍스트만 남김 (먼저 처리)
    answer = re.sub(r'\[([^\]]+)\]\(https?://[^\)]+\)', r'\1', answer)
    # 괄호 안 단독 URL: (https://...) → 괄호째 제거 (문장 구조 보존)
    answer = re.sub(r'\s*\(\s*https?://\S+?\s*\)', '', answer)
    # 나머지 bare https?:// URL
    answer = re.sub(r'https?://\S+', '참고 출처를 확인하세요', answer)
    # www. 로 시작하는 주소도 제거
    answer = re.sub(r'\bwww\.\S+', '참고 출처를 확인하세요', answer)
    # bare 도메인 URL 제거 (예: job.seoultech.ac.kr/... , portal.seoultech.ac.kr/...)
    answer = re.sub(r'\b[a-zA-Z0-9][\w\-]*\.(?:ac\.kr|co\.kr|go\.kr|or\.kr|com|net|org)/\S*', '참고 출처를 확인하세요', answer)
    # 전화번호 제거 (예: 02-970-6128, 010-1234-5678)
    answer = re.sub(r'\b0\d{1,2}[-.\s]\d{3,4}[-.\s]\d{4}\b', '참고 출처를 확인하세요', answer)

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
    st.title("🎓 ChaTech")
    st.caption("서울과학기술대학교 공지사항 및 청년 지원 정책을 질의할 수 있습니다.")

    with st.expander("💡 이렇게 질문해보세요", expanded=False):
        st.markdown(
            "**📌 현재 신청 가능한 공고 조회**\n"
            "> 지금 신청할 수 있는 장학금 알려줘\n\n"
            "> 현재 모집 중인 현장실습 있어?\n\n"
            "**📌 특정 제도·정책 정보**\n"
            "> 국가장학금 신청 자격이 어떻게 돼?\n\n"
            "> 복수전공 신청 방법 알려줘\n\n"
            "**📌 자격 확인**\n"
            "> 나 컴공 4학년인데 KIST 현장실습 지원할 수 있어?\n\n"
            "**📌 청년 정책**\n"
            "> 서울시 청년 월세 지원 받을 수 있어?\n\n"
            "⚠️ **현재 DB 기준 날짜로 마감된 공고는 '없음'으로 안내됩니다.** "
            "최신 공고는 [학교 홈페이지](https://www.seoultech.ac.kr) 또는 "
            "[온통청년 포털](https://www.youthcenter.go.kr)에서 직접 확인하세요."
        )

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
                    "죄송해요, 해당 질문과 관련된 정보를 찾지 못했어요 😅\n\n"
                    "더 구체적인 키워드로 다시 질문해 주시거나, "
                    "학교 공지사항은 **서울과기대 홈페이지**, "
                    "청년 정책은 **온통청년 포털**에서 직접 확인해 보세요!"
                )
            else:
                with st.spinner("답변을 생성하는 중..."):
                    answer = build_answer(user_input, context, st.session_state.messages[:-1])

            st.markdown(answer)

            # 답변에 실제 언급된 출처만 필터링 (이후 모든 판단의 기준)
            display_sources = _filter_cited_sources(answer, sources) if sources else []

            # 만료 문서 알림
            expired_sources = [s for s in display_sources if "[기간 종료]" in s.get("title", "")]
            valid_sources   = [s for s in display_sources if "[기간 종료]" not in s.get("title", "")]
            if expired_sources:
                if valid_sources:
                    # 만료+유효 혼재 — 사용자가 각 항목의 [기간 종료] 태그를 직접 확인하도록 안내
                    st.warning(
                        "⚠️ 위 답변에는 **신청 가능한 공고**와 **기간이 종료된 공고**가 함께 포함되어 있습니다. "
                        "참고 출처의 각 항목 옆 **[기간 종료]** 표시를 확인해 주세요."
                    )
                else:
                    # 전체 만료 — 대표 마감일 추출 후 안내
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

            # 청년정책 출처 알림 — 답변에 실제 인용된 출처(display_sources) 기준
            has_youth = any(s.get("category") == "청년정책" for s in display_sources)
            if has_youth:
                st.info(
                    "ℹ️ 청년정책 정보는 개별 공고 페이지로 직접 연결이 어렵습니다. "
                    "[온통청년 포털](https://www.youthcenter.go.kr)에서 정책명을 검색해 주세요.",
                    icon="🔍"
                )

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
