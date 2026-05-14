# Project: chatech (지능형 학사 어드바이저)


## Critical Rules (절대 규칙)
- **마크다운 일관성**: VLM 추출 시 모든 공지는 정의된 마크다운 구조(Header, Table, List)를 엄격히 준수해야 하며, 데이터 파싱 효율을 위해 형식을 임의로 변경하는 것을 금지함.
- **시크릿 관리**: `.env` 파일의 API Key들은 절대 커밋하거나 하드코딩하지 않으며 `os.getenv`로만 관리함.
- **데이터 보존**: 날짜, 금액, 신청 자격 등 핵심 정보는 `parent_markdown`에 원문 형태(표, 리스트 등)로 누락 없이 보존해야 함.
- **에러 격리**: 개별 아이템 처리 중 발생하는 에러가 전체 파이프라인의 중단으로 이어지지 않도록 `try-except` 블록으로 보호함.


## Architecture (아키텍처)
dags/              -> Airflow DAG 정의 (워크플로우 스케줄링 및 의존성 관리)
app/
  services/
    scraper.py     -> Playwright 기반 공지사항 이미지 캡처 및 중복 체크
    processor.py   -> VLM(GPT-4o-mini) 기반 Visual ETL (이미지 -> 마크다운)
    youth_api.py   -> 청년정책 API 수집 및 처리
  main.py          -> 파이프라인 통합 제어 및 로컬 테스트용 스크립트
  schema.py        -> Pydantic 기반 데이터 모델 정의
data/              -> 수집 이력(scrape_history.txt) 관리
images/            -> 카테고리별 공지사항 캡처본 저장
chroma_db/         -> Chroma 벡터 데이터베이스 저장소
docker-compose.yml -> Airflow 환경 구동을 위한 컨테이너 설정


## Tech Stack (기술 스택)
Language: Python 3.11

Orchestration: Apache Airflow (Docker 기반)

Scraping: Playwright (Chromium)

AI/LLM: GPT-4o-mini (Vision Analysis), OpenAI Embeddings

Vector DB: Chroma

Library: LangChain, Pydantic, python-dotenv


## Build & Test Commands (빌드/테스트)
```bash
docker-compose up -d
conda activate capstone             # 가상환경 활성화
pip install playwright openai python-dotenv pydantic langchain langchain-openai langchain-chroma
playwright install chromium         # 브라우저 엔진 설치
python app/main.py                  # 파이프라인 전체 실행
```


## Domain Context (도메인 컨텍스트)
- **Target Source**: 서울과학기술대학교(SeoulTech) 공식 홈페이지의 5대 주요 공지 게시판(대학, 학사, 장학, 대학원, 취업)을 주요 데이터 소스로 함.
- **Visual ETL Rationale**: 학교 홈페이지의 HTML 구조가 불규칙하고 동적으로 변하는 경우가 많아, 전통적인 파싱 대신 본문 영역을 이미지로 캡처하여 VLM(GPT-4o-mini)으로 분석하는 방식을 채택함.
- **Parent-Child Split Strategy**: 
    - **Child (Summary)**: 검색 효율을 높이기 위해 핵심 정보를 3줄 내외로 요약한 데이터. 벡터 DB(`chroma_db`)의 인덱스로 활용됨.
    - **Parent (Full Markdown)**: 실제 답변 생성 시 정확한 정보를 제공하기 위해 마크다운으로 변환된 본문 전체 데이터. 메타데이터로 저장됨.
- **Data Integration**: 학교 공지사항 외에도 온통청년 API를 통해 제공되는 정책 데이터를 통합하여 대학생에게 필요한 정보를 원스톱으로 제공하는 것을 목표로 함.


## Coding Convention (코딩 컨벤션)
- **Naming**: 변수와 함수명은 `snake_case`, 클래스명은 `PascalCase`를 사용함
- **Type Hinting**: 모든 함수의 인자와 반환값에 파이썬 Type Hint를 반드시 적용함 (예: `def func(url: str) -> bool:`)
- **Docstrings**: 클래스와 복잡한 함수에는 Google Style Docstrings를 사용하여 설명을 추가함
- **Task Atomicity**: Airflow Task는 하나의 명확한 작업(수집, 분석, 저장 등)만 수행하도록 원자성을 유지함
- **Logging**: `print()` 사용을 금지하며, `logging` 라이브러리를 통해 단계별 성공/실패 로그를 남김
- **JSON Enforcement**: VLM(GPT-4o-mini)의 응답은 항상 `schema.py`에 정의된 JSON 구조를 유지하도록 프롬프트에서 강제함



## Key Patterns (핵심 패턴)
- **Visual ETL Flow**: 공지사항 수집 시 HTML 파싱 대신 [Playwright 캡처 -> VLM 분석 -> 마크다운 변환] 순서를 엄격히 따름
- **Parent-Child RAG**: 검색 성능 향상을 위해 `child_summary`를 임베딩 인덱스로 쓰고, `parent_markdown`을 답변 생성용 메타데이터로 매핑함
- **Deduplication First**: 모든 스크래핑 작업 시작 전 `data/scrape_history.txt`를 대조하여 중복 수집과 토큰 낭비를 원천 차단함
- **Graceful Failure**: 개별 공지사항 처리 실패가 전체 파이프라인 중단으로 이어지지 않도록 `try-except` 블록으로 각 아이템을 보호함
- **Schema-Driven**: 모든 데이터는 `schema.py`의 Pydantic 모델을 통과해야만 저장하며, 검증 실패 시 해당 데이터는 폐기하고 로그를 남김
- **Orchestration First**: 모든 파이프라인은 Airflow DAG를 통해 작업 간 의존성(Dependency)과 순서를 관리함
- **Idempotency (멱등성)**: 동일한 날짜의 작업을 여러 번 실행해도 중복 데이터가 쌓이지 않도록 수집 이력을 엄격히 체크함
- **Failure Recovery**: 수집 실패 시 Airflow의 자동 재시도(Retry) 기능을 활용하여 파이프라인 가용성을 보장함