# ChaTech — 서울과학기술대학교 지능형 학사 어드바이저

서울과기대 공식 공지사항(학사·장학·취업 등 5개 게시판)과 온통청년 청년정책을 통합하여 학생 질문에 실시간으로 답변하는 RAG 기반 챗봇입니다.

---

## 핵심 설계

### Visual ETL
학교 홈페이지의 HTML 구조가 동적·불규칙하여 전통적인 파싱 대신 **Playwright로 공지 본문을 이미지로 캡처 → GPT-4o-mini(VLM)로 마크다운 변환** 방식을 채택했습니다.

### Parent-Child RAG
검색 정확도와 답변 품질을 동시에 확보하기 위해 문서를 두 레이어로 분리합니다.

| 레이어 | 내용 | 역할 |
|--------|------|------|
| **Child** | 핵심 요약 3줄 | 벡터 검색 인덱스 |
| **Parent** | 마크다운 원문 전체 | LLM 답변 생성 컨텍스트 |

Child로 관련 문서를 찾고, Parent의 전체 정보로 답변을 생성하여 검색 노이즈를 줄이고 정보 손실을 방지합니다.

---

## 아키텍처

```
[데이터 수집 / app/main.py]
    scraper.py    →  Playwright로 공지 이미지 캡처 (중복 체크)
    processor.py  →  GPT-4o-mini VLM으로 마크다운 변환
    youth_api.py  →  온통청년 API 청년정책 수집

[ChromaDB]  ←  child_summary(검색용) + parent_markdown(답변용)

[Streamlit 챗봇 / app/web_ui.py]
    retrieve_with_parent()  →  카테고리 분리 검색 → 만료 필터 → 2단계 폴백
    build_answer()          →  Groq(llama-3.1-8b) 기반 답변 생성
```

---

## 주요 가드레일

- **만료 문서 필터링** : 공고 마감일 파싱 후 현재 날짜 기준 자동 제외
- **카테고리 분리 검색** : 학교공지·청년정책 별도 검색 후 합산 (특정 카테고리 묻힘 방지)
- **유사도 임계값** : score threshold 미만 문서 제거
- **2단계 폴백** : 키워드 매칭 → 쿼리 재작성 후 재검색

---

## 평가 결과 (RAGAS)

동일한 골든셋(`eval/golden_set_v2.json`, 50개 Q&A) 기준으로 이전 버전(prototype2)과 비교했습니다.

| 지표 | prototype2 | **ChaTech** |
|------|:----------:|:-----------:|
| Faithfulness | 0.9180 | 0.7573 |
| **Answer Relevancy** | 0.4852 | **0.4883 ↑** |
| Context Precision | 0.9244 | 0.8356 |
| Context Recall | 0.9200 | 0.7900 |

> prototype2는 child 요약 기반으로 컨텍스트 점수가 높지만, 실제 질문 의도에 부합하는 답변 생성(Answer Relevancy)에서는 Parent-Child RAG를 적용한 ChaTech이 앞섭니다.

평가 스크립트 및 결과 : [`eval/`](./eval)

---

## 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.11 |
| Scraping | Playwright |
| VLM | GPT-4o-mini |
| LLM | llama-3.1-8b-instant (Groq) |
| Embedding | text-embedding-3-small (OpenAI) |
| Vector DB | ChromaDB |
| UI | Streamlit |
| Evaluation | RAGAS |

---

## 프로젝트 구조

```
app/
├── web_ui.py           # RAG 파이프라인 + 챗봇 UI (메인)
├── main.py             # 데이터 수집·적재 파이프라인
├── schema.py           # Pydantic 데이터 모델
└── services/
    ├── scraper.py      # Playwright 기반 공지 수집
    ├── processor.py    # VLM 기반 Visual ETL
    └── youth_api.py    # 온통청년 API 수집
eval/
├── run_ragas.py                        # RAGAS 평가 스크립트
├── golden_set_v2.json                  # 평가용 Q&A 50개
├── ragas_results_8b_v2golden.json      # ChaTech 평가 결과
└── ragas_results_proto2_v2golden.json  # prototype2 비교 결과
prototype2/
└── streamlit_proto2.py  # 비교 대상 이전 버전
```
