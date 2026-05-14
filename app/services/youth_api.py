# app/services/youth_api.py
import os
import json
import urllib.parse
import logging
import requests
import pandas as pd
import xml.etree.ElementTree as ET
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_XLSX_PATH    = os.path.join(os.path.dirname(__file__), "..", "..", "API코드정보.xlsx")
_PRIMARY_URL  = "https://www.youthcenter.go.kr/opi/youthPlcyList.do"
_FALLBACK_URL = "https://www.youthcenter.go.kr/wrk/yrm/plcyInfo/selectPlcy"
_HOME_URL     = "https://www.youthcenter.go.kr/"

# 지역별 쿼터제 수집 대상 (검색어, 표시명)
# polyRgnSecd 코드 방식은 서버에서 무시됨 → searchWord 텍스트 검색으로 대체
_REGION_QUERIES: list[tuple[str, str]] = [
    ("서울", "서울"),
    ("경기", "경기"),
    ("",     "전국"),  # 빈 검색어 = 전체 최신순
]


def _load_code_mappings(xlsx_path: str) -> Dict[str, Dict[str, str]]:
    """xlsx 파일에서 코드→명칭 매핑 딕셔너리를 빌드합니다.

    Returns:
        {
          'code_info': {'0054001': '중앙부처', ...},
          'lclsf':     {'1': '일자리', ...},
          'mclsf':     {'1': '취업', ...},
          'keyword':   {'1': '대출', ...},
        }
    """
    maps: Dict[str, Dict[str, str]] = {
        "code_info": {},
        "lclsf":     {},
        "mclsf":     {},
        "keyword":   {},
    }
    try:
        wb = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl", header=None)

        # ① 코드정보: 5열 (분류, -, -, 코드, 코드내용)
        df_code = wb.get("코드정보", pd.DataFrame())
        for _, row in df_code.iterrows():
            code = str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else ""
            name = str(row.iloc[4]).strip() if pd.notna(row.iloc[4]) else ""
            if code and name and code != "코드":
                maps["code_info"][code] = name

        # ② 정책대분류: 2열 (번호, 명칭)
        df_lclsf = wb.get("정책대분류", pd.DataFrame())
        for _, row in df_lclsf.iterrows():
            num  = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            if num.isdigit() and name:
                maps["lclsf"][num] = name

        # ③ 정책중분류: 2열 (번호, 명칭)
        df_mclsf = wb.get("정책중분류", pd.DataFrame())
        for _, row in df_mclsf.iterrows():
            num  = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            if num.isdigit() and name:
                maps["mclsf"][num] = name

        # ④ 정책키워드: 2열 (번호, 명칭)
        df_kw = wb.get("정책키워드", pd.DataFrame())
        for _, row in df_kw.iterrows():
            num  = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            name = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
            if num.isdigit() and name:
                maps["keyword"][num] = name

        logger.info(
            "✅ 코드 매핑 로드: 코드정보=%d건, 대분류=%d건, 중분류=%d건, 키워드=%d건",
            len(maps["code_info"]), len(maps["lclsf"]),
            len(maps["mclsf"]),     len(maps["keyword"]),
        )
    except Exception as e:
        logger.error("❌ xlsx 코드 매핑 로드 실패: %s", e)

    return maps


class YouthApiIngestor:
    """온통청년 정책 API 수집 및 ChromaDB 적재용 데이터 변환 클래스."""

    def __init__(self, api_key: str | None, xlsx_path: str = _XLSX_PATH) -> None:
        self.api_key = urllib.parse.unquote(api_key) if api_key else None
        self.client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model   = "gpt-4o-mini"
        self._maps   = _load_code_mappings(xlsx_path)

    # ── 코드 → 명칭 변환 ─────────────────────────────────────────────────────
    def _decode(self, value: Any, map_key: str) -> str:
        """코드 또는 번호를 사람이 읽을 수 있는 명칭으로 변환합니다."""
        if value is None or str(value).strip() in ("", "None"):
            return "-"
        mapping = self._maps.get(map_key, {})
        str_val = str(value).strip()
        return mapping.get(str_val, str_val)

    # ── 세션 초기화 (CSRF 토큰 획득) ─────────────────────────────────────────
    @staticmethod
    def _build_session() -> requests.Session:
        """온통청년 홈을 방문해 XSRF 토큰과 JWT 쿠키를 획득한 세션을 반환합니다."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Referer": _HOME_URL,
        })
        try:
            session.get(_HOME_URL, timeout=15)
            xsrf = session.cookies.get("XSRF-TOKEN", "")
            session.headers["X-XSRF-TOKEN"] = xsrf
            session.headers["user_id"] = "guest"
            logger.info("  세션 초기화 완료 | XSRF-TOKEN: %s...", xsrf[:10])
        except Exception as e:
            logger.warning("  세션 초기화 실패 (계속 진행): %s", e)
        return session

    # ── Parent 마크다운 템플릿 ────────────────────────────────────────────────
    def _build_parent_markdown(self, p: dict, decoded: dict) -> str:
        """API JSON → 구조화된 마크다운.

        정책 설명, 상세 지원내용, 신청 자격, 신청 방법을 모두 포함합니다.
        """
        title      = p.get("plcyNm", "제목 없음")
        content    = p.get("plcyExplnCn") or "내용 없음"
        support_cn = p.get("plcySprtCn") or ""
        apply_url  = p.get("refUrlAddr1") or p.get("aplyUrlAddr") or _HOME_URL
        location   = p.get("rgtrHghrkInstCdNm") or "-"

        bgng = p.get("aplyPrdBgngYmd", "")
        end_ = p.get("aplyPrdEndYmd",  "")
        if bgng and end_:
            deadline = f"{bgng[:4]}-{bgng[4:6]}-{bgng[6:]} ~ {end_[:4]}-{end_[4:6]}-{end_[6:]}"
        else:
            deadline = p.get("aplyPrdCn") or "상시 모집"

        # 대상 연령
        min_age = p.get("sprtTrgtMinAge")
        max_age = p.get("sprtTrgtMaxAge")
        if min_age and max_age:
            age_range = f"{min_age}세 ~ {max_age}세"
        elif min_age:
            age_range = f"{min_age}세 이상"
        elif max_age:
            age_range = f"{max_age}세 이하"
        else:
            age_range = "-"

        school        = p.get("schoolCdNm") or "-"
        qualification = p.get("addAplyQlfcCndCn") or "-"
        restriction   = p.get("ptcpPrpTrgtCn") or "-"
        apply_method  = p.get("plcyAplyMthdCn") or "-"

        main_cat   = decoded["main_cat"]
        sub_cat    = decoded["sub_cat"]
        keyword    = decoded["keyword"]
        provider   = decoded["provider"]
        apply_type = decoded["apply_type"]

        support_section = f"\n### 상세 지원내용\n{support_cn}\n" if support_cn and support_cn != "-" else ""
        restriction_line = f"- 참여 제한 대상: {restriction}\n" if restriction != "-" else ""
        apply_method_section = f"\n### 신청 방법\n{apply_method}\n" if apply_method != "-" else ""

        return (
            f"## {title}\n\n"
            f"---\n"
            f"- 정책분류: {main_cat} > {sub_cat}\n"
            f"- 지원유형: {keyword}\n"
            f"- 주관기관: {provider}\n"
            f"- 지역: {location}\n"
            f"- 신청방식: {apply_type}\n"
            f"---\n\n"
            f"### 정책 설명\n{content}\n"
            f"{support_section}\n"
            f"### 신청 자격\n"
            f"- 대상 연령: {age_range}\n"
            f"- 학력 조건: {school}\n"
            f"- 추가 자격 조건: {qualification}\n"
            f"{restriction_line}"
            f"{apply_method_section}\n"
            f"| 항목 | 내용 |\n"
            f"|------|------|\n"
            f"| 신청 기간 | {deadline} |\n"
            f"| 신청 URL  | {apply_url} |\n\n"
            f"---\n📌 상세 내용은 [온통청년 포털]({apply_url})에서 확인하세요"
        )

    # ── 만료 정책 필터 ────────────────────────────────────────────────────────
    @staticmethod
    def _is_expired(raw: dict) -> bool:
        """신청 종료일이 6개월 이상 지난 정책이면 True를 반환합니다."""
        end_ymd = raw.get("aplyPrdEndYmd", "")
        if not end_ymd or len(end_ymd) < 8:
            return False  # 상시 모집 등 날짜 없는 경우는 제외하지 않음
        try:
            from datetime import date, timedelta
            end_date = date(int(end_ymd[:4]), int(end_ymd[4:6]), int(end_ymd[6:8]))
            return end_date < (date.today() - timedelta(days=180))
        except Exception:
            return False

    # ── Child 요약 (GPT) ─────────────────────────────────────────────────────
    def _generate_child_summary(self, title: str, parent_md: str) -> str:
        """Parent 마크다운 → GPT-4o-mini → 3줄 요약 (지역명 강화)."""
        prompt = (
            f"아래는 청년 정책 '{title}'의 상세 내용입니다.\n"
            "검색 엔진이 문서를 찾을 수 있도록 아래 규칙을 엄수하여 3줄로 요약하세요.\n\n"
            "규칙:\n"
            "1. 첫 번째 줄 맨 앞: 수혜 지역을 '[서울]', '[경기]', '[전국]' 등 대괄호 태그로 "
            "반드시 표기. 예) '[서울] 서울시 청년 수당 - 만 19~34세 미취업 청년 대상'. "
            "지역 태그는 검색의 결정적 키워드이므로 절대 생략 불가.\n"
            "2. 두 번째 줄: 지원 대상 연령과 자격 조건을 간결하게 기술.\n"
            "3. 세 번째 줄: 핵심 혜택(금액, 횟수, 기간 등 구체 수치 포함)과 신청 기간을 기술.\n\n"
            f"내용:\n{parent_md[:2000]}\n\n"
            '반드시 JSON 형식으로만 응답하세요: {"child_summary": "..."}'
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(resp.choices[0].message.content).get("child_summary", "")
        except Exception as e:
            logger.error("Child 요약 생성 실패: %s", e)
            return f"[청년정책] {title}: 신청 정보는 온통청년 포털에서 확인하세요."

    # ── 데이터 변환 공통 로직 ─────────────────────────────────────────────────
    def _convert(self, raw: dict) -> dict | None:
        """API 원시 데이터 → ChromaDB 적재용 딕셔너리. 만료 정책은 None 반환."""
        if self._is_expired(raw):
            logger.debug("  ⏭️  만료 정책 스킵: %s (종료: %s)",
                         raw.get("plcyNm", "")[:30], raw.get("aplyPrdEndYmd", ""))
            return None

        # 대분류: API가 이미 이름을 반환하는 경우 (userLclsfNm)와
        # 코드 번호를 반환하는 경우 (lclsfNm) 모두 처리
        main_cat_raw = raw.get("userLclsfNm") or raw.get("lclsfNm")
        sub_cat_raw  = raw.get("userMclsfNm") or raw.get("mclsfNm")
        # API가 이름을 반환할 때는 decode가 pass-through (dict에 없으면 원값 반환)
        decoded = {
            "main_cat":   self._decode(main_cat_raw,                  "lclsf"),
            "sub_cat":    self._decode(sub_cat_raw,                   "mclsf"),
            "keyword":    raw.get("plcyKywdNm") or self._decode(raw.get("plcyKywdSn"), "keyword") or "-",
            "provider":   raw.get("rgtrInstCdNm") or self._decode(raw.get("pvsnInstGroupCd"), "code_info") or "-",
            "apply_type": self._decode(raw.get("aplyPrdSeCd"),        "code_info"),
        }

        parent_md     = self._build_parent_markdown(raw, decoded)
        child_summary = self._generate_child_summary(raw.get("plcyNm", ""), parent_md)

        return {
            "title":            raw.get("plcyNm", "제목 없음"),
            "url":              raw.get("refUrlAddr1") or raw.get("aplyUrlAddr") or _HOME_URL,
            "bizId":            raw.get("plcyNo", ""),
            "location":         raw.get("rgtrHghrkInstCdNm", "-"),
            "category":         raw.get("bscPlanPlcyWayNo", "-"),
            "parent_markdown":  parent_md,
            "child_summary":    child_summary,
            # ChromaDB 필터링용 메타데이터
            "policy_main_cat":   decoded["main_cat"],
            "policy_sub_cat":    decoded["sub_cat"],
            "policy_keyword":    decoded["keyword"],
            "provider_group":    decoded["provider"],
            "apply_period_type": decoded["apply_type"],
        }

    # ── 방법 1: 공식 Open API (openApiVlak 키 방식) ──────────────────────────
    def _fetch_via_openapi(
        self,
        query: str,
        display: int,
        page_num: int = 1,
        region_code: str | None = None,
    ) -> List[dict] | None:
        """공식 Open API 엔드포인트로 youthPolicyList를 수집합니다.

        1차: rtnType=json 요청 → result.youthPolicyList 추출
        2차: JSON 실패 시 XML 폴백
        """
        if not self.api_key:
            return None

        base_params: dict = {
            "openApiVlak": self.api_key,
            "display":     display,
            "pageIndex":   page_num,
            "srchPolyBizSecd": query,
        }
        if region_code:
            base_params["polyRgnSecd"] = region_code

        # 1차: JSON 응답 시도 (timeout=5로 단축 — 8080 포트 리다이렉트 시 빠르게 포기)
        try:
            params = {**base_params, "rtnType": "json"}
            logger.info("🌐 [1차-JSON] 호출: %s", _PRIMARY_URL)
            resp = requests.get(_PRIMARY_URL, params=params, timeout=5, allow_redirects=False)
            logger.info("  HTTP %s | Content-Type: %s", resp.status_code, resp.headers.get("Content-Type", "-"))
            if resp.is_redirect:
                logger.warning("  ⚠️ 리다이렉트 감지 → XML 폴백 시도 (Location: %s)", resp.headers.get("Location", "-"))
            else:
                resp.raise_for_status()
                data = resp.json()
                policy_list = data.get("result", {}).get("youthPolicyList", [])
                if policy_list:
                    logger.info("  ✅ JSON 수집 성공: %d건", len(policy_list))
                    return policy_list
                logger.warning("  ⚠️ youthPolicyList 비어 있음: %s", json.dumps(data, ensure_ascii=False)[:300])
        except (ValueError, requests.exceptions.HTTPError) as e:
            logger.warning("  ⚠️ JSON 방식 실패 (%s) → XML 폴백 시도", e)
        except Exception as e:
            logger.warning("  ❌ JSON 호출 예외: %s → XML 폴백 시도", e)

        # 2차: XML 폴백 (timeout=5, 리다이렉트 차단)
        try:
            logger.info("  🌐 [1차-XML] XML 폴백 호출")
            resp = requests.get(_PRIMARY_URL, params=base_params, timeout=5, allow_redirects=False)
            if not resp.is_redirect:
                resp.raise_for_status()
                root = ET.fromstring(resp.text)
                items = root.findall(".//youthPolicyList")
                if items:
                    policy_list = [{child.tag: child.text for child in item} for item in items]
                    logger.info("  ✅ XML 수집 성공: %d건", len(policy_list))
                    return policy_list
            logger.warning("  ⚠️ XML 응답에 youthPolicyList 없음 (또는 리다이렉트)")
        except Exception as e:
            logger.warning("  ❌ XML 폴백 실패: %s", e)

        return None

    # ── 방법 2: 세션 기반 내부 API (폴백) ────────────────────────────────────
    def _fetch_via_session(
        self,
        query: str,
        display: int,
        page_num: int = 1,
        region_code: str | None = None,
    ) -> List[dict] | None:
        """세션 쿠키(XSRF + JWT)를 이용해 내부 API로 정책을 수집합니다."""
        try:
            logger.info("🌐 [2차] 세션 기반 폴백 호출: %s (p.%d)", _FALLBACK_URL, page_num)
            session = self._build_session()
            plcy_req: dict = {"useYn": "Y", "plcyAprvSttsCd": "0044002", "searchWord": query}
            if region_code:
                plcy_req["polyRgnSecd"] = region_code
            body = {
                "plcyReq":   plcy_req,
                "paggingVO": {"pageNum": page_num, "pageSize": display},
            }
            resp = session.post(_FALLBACK_URL, json=body, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if data.get("resultCode") != 200:
                logger.warning("  ⚠️ API resultCode: %s", data.get("resultCode"))
                return None
            policy_list = data.get("result", {}).get("plcyList", [])
            logger.info("  ✅ 세션 기반 수집 성공: %d건", len(policy_list))
            return policy_list
        except Exception as e:
            logger.error("  ❌ 세션 기반 호출 실패: %s", e)
            return None

    # ── 퍼블릭 인터페이스 ─────────────────────────────────────────────────────
    def fetch_all(
        self,
        query: str = "청년",
        display_count: int = 10,
        page_num: int = 1,
        region_code: str | None = None,
    ) -> List[Dict[str, Any]]:
        """온통청년 API에서 정책 목록을 수집합니다.

        1차: openApiVlak 키 방식 (사용자 지정 URL)
        2차 폴백: 세션 기반 내부 API
        """
        policy_list = (
            self._fetch_via_openapi(query, display_count, page_num, region_code)
            or self._fetch_via_session(query, display_count, page_num, region_code)
        )

        if not policy_list:
            logger.warning("⚠️ 모든 API 방법 실패 → mock 데이터 반환")
            return self._get_mock_data()

        return [r for r in (self._convert(p) for p in policy_list) if r is not None]

    def fetch_by_regions(
        self,
        target_per_region: int = 200,
        display_count: int = 100,
        skip_ids: set[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """지역별 쿼터제로 청년 정책을 수집합니다.

        Args:
            target_per_region: 지역당 목표 신규 수집 건수 (기본 200).
            display_count: API 호출 1회당 요청 건수 (기본 100).
            skip_ids: 이미 DB에 존재하는 bizId 집합. 포함된 정책은
                      GPT 호출 없이 즉시 건너뜁니다.

        Returns:
            전 지역에서 수집한 신규 정책 목록 (전역 중복 제거 후).
        """
        skip_ids = skip_ids or set()
        all_results: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for region_query, region_name in _REGION_QUERIES:
            region_collected = 0
            page_num = 1
            logger.info("📍 [%s] 지역 수집 시작 (목표 신규: %d건)", region_name, target_per_region)

            while region_collected < target_per_region:
                policy_list = (
                    self._fetch_via_openapi(region_query, display_count, page_num)
                    or self._fetch_via_session(region_query, display_count, page_num)
                )

                if not policy_list:
                    logger.info("  [%s] p.%d 응답 없음 — 수집 종료", region_name, page_num)
                    break

                new_in_page = 0
                for p in policy_list:
                    if region_collected >= target_per_region:
                        break
                    policy_id = p.get("plcyNo", "")

                    # 이미 처리한 ID (이번 실행 내 중복)
                    if policy_id and policy_id in seen_ids:
                        continue
                    if policy_id:
                        seen_ids.add(policy_id)

                    # 이미 DB에 있는 ID → GPT 호출 없이 skip
                    if policy_id and policy_id in skip_ids:
                        logger.debug("  [%s] DB 기존 정책 스킵 (GPT 미호출): %s", region_name, policy_id)
                        continue

                    try:
                        converted = self._convert(p)
                        if converted is not None:
                            all_results.append(converted)
                            region_collected += 1
                            new_in_page += 1
                    except Exception as e:
                        logger.error("  [%s] 정책 변환 오류: %s", region_name, e)

                logger.info("  [%s] p.%d: +%d건 신규 (누계 %d건)", region_name, page_num, new_in_page, region_collected)

                if len(policy_list) < display_count:
                    logger.info("  [%s] 마지막 페이지 도달 (반환 %d건)", region_name, len(policy_list))
                    break

                page_num += 1

            logger.info("✅ [%s] 최종 신규 수집: %d건", region_name, region_collected)

        logger.info("🌏 지역별 수집 완료 — 신규 총 %d건 (전역 중복 제거 후)", len(all_results))
        return all_results

    def _get_mock_data(self) -> List[Dict[str, Any]]:
        raw_list = [
            {
                "plcyNm":              "[가짜 데이터] 서울시 청년 수당 지원 프로그램",
                "plcyExplnCn":         "서울시에 거주하는 미취업 청년들을 대상으로 활동 지원금을 지급하는 정책입니다.",
                "plcySprtCn":          "월 50만원 활동 지원금 지급 (최대 6개월)",
                "refUrlAddr1":         "https://www.youthcenter.go.kr",
                "rgtrHghrkInstCdNm":   "서울특별시",
                "plcyNo":              "MOCK001",
                "userLclsfNm":         "1",
                "userMclsfNm":         "1",
                "plcyKywdNm":          "2",
                "pvsnInstGroupCd":     "0054002",
                "aplyPrdSeCd":         "0057001",
                "aplyPrdBgngYmd":      "20260501",
                "aplyPrdEndYmd":       "20260531",
            }
        ]
        return [self._convert(r) for r in raw_list]
