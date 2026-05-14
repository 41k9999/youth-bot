# app/services/processor.py
import os
import io
import base64
import json
import logging
import requests
from openai import OpenAI
from app.schema import ProcessedNotice
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class DocumentAIPipeline:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model  = "gpt-4o-mini"

    # ── 유틸리티 ──────────────────────────────────────────────────────────────
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _excel_to_markdown(self, excel_path: str) -> str:
        """엑셀 파일 → 시트별 마크다운 표 변환."""
        try:
            import pandas as pd
            dfs = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
            parts = []
            for sheet_name, df in dfs.items():
                df = df.dropna(how="all").fillna("")
                if df.empty:
                    continue
                parts.append(f"### {sheet_name}\n\n{df.to_markdown(index=False)}")
            return "\n\n".join(parts)
        except Exception as e:
            logger.error(f"엑셀 마크다운 변환 실패: {e}")
            return ""

    def _extract_pdf_text(self, pdf_url: str, orig_name: str = "") -> str:
        """pypdf로 PDF 전 페이지 텍스트를 추출합니다.

        텍스트가 부족한 페이지는 렌더링 이미지→VLM으로 보완합니다.

        Args:
            pdf_url: PDF 다운로드 URL.
            orig_name: 로그용 파일명.

        Returns:
            마크다운 형식으로 구성된 전 페이지 텍스트.
        """
        try:
            resp = requests.get(pdf_url, timeout=30)
            resp.raise_for_status()
            pdf_bytes = resp.content
        except Exception as e:
            logger.error(f"  PDF 다운로드 실패 ({orig_name}): {e}")
            return ""

        pages_md: list[str] = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(pdf_bytes))
            total = len(reader.pages)
            logger.info(f"  📄 PDF 텍스트 추출 시작 ({orig_name}) — 전체 {total}페이지")

            for idx, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if len(text) >= 50:
                    pages_md.append(f"#### {idx}페이지\n{text}")
                    logger.info(f"    ✅ {idx}/{total}p 텍스트 추출 완료 ({len(text)}자)")
                else:
                    # 텍스트가 거의 없는 페이지 → VLM으로 보완
                    logger.info(f"    ⚠️  {idx}/{total}p 텍스트 부족({len(text)}자) → VLM 분석")
                    vlm_text = self._analyze_pdf_page_via_vlm(pdf_bytes, idx, total)
                    if vlm_text:
                        pages_md.append(f"#### {idx}페이지 (VLM)\n{vlm_text}")

        except Exception as e:
            logger.error(f"  pypdf 추출 실패 ({orig_name}): {e}")
            return ""

        combined = "\n\n".join(pages_md)
        logger.info(f"  📄 PDF 추출 완료: 총 {len(combined)}자")
        return combined

    def _analyze_pdf_page_via_vlm(self, pdf_bytes: bytes, page_num: int, total: int) -> str:
        """단일 PDF 페이지를 PNG 이미지로 렌더링한 뒤 GPT-4o-mini Vision으로 분석합니다.

        pdf2image(poppler) 로 렌더링하여 GPT Vision이 요구하는 이미지 MIME 타입을 충족합니다.
        pdf2image가 설치되지 않은 경우 빈 문자열을 반환합니다.
        """
        try:
            from pdf2image import convert_from_bytes
        except ImportError:
            logger.warning("  ⚠️ pdf2image 미설치 → p%d VLM 분석 스킵 (pip install pdf2image)", page_num)
            return ""

        try:
            images = convert_from_bytes(
                pdf_bytes,
                first_page=page_num,
                last_page=page_num,
                dpi=150,
            )
            if not images:
                logger.warning("  pdf2image: p%d 렌더링 결과 없음", page_num)
                return ""

            buf = io.BytesIO()
            images[0].save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            prompt = (
                f"아래는 첨부 PDF의 {page_num}/{total} 페이지입니다. "
                "이 페이지의 모든 내용을 빠짐없이 마크다운으로 변환해 주세요."
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64}"
                    }},
                ]}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("  VLM PDF 페이지 분석 실패 (p%d): %s", page_num, e)
            return ""

    def _generate_child_from_text(self, title: str, parent_text: str) -> str:
        """텍스트 기반 GPT 호출로 3줄 Child 요약을 생성합니다."""
        prompt = (
            f"아래는 서울과학기술대학교(과기대) '{title}' 공지사항의 상세 내용입니다.\n"
            "검색에 최적화된 3줄 핵심 요약을 작성하세요.\n\n"
            "규칙:\n"
            "1. 첫 번째 줄 맨 앞에 '[과기대]' 태그를 반드시 표기하세요. "
            "예) '[과기대] 2026-1학기 장학금 신청 - 재학생 대상'.\n"
            "2. 날짜·금액·신청 대상 등 핵심 정보를 반드시 포함하세요.\n\n"
            f"내용:\n{parent_text[:3000]}\n\n"
            '반드시 JSON 형식으로만 응답하세요: {"child_summary": "..."}'
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        return json.loads(resp.choices[0].message.content).get("child_summary", "")

    # ── 전략별 처리 ───────────────────────────────────────────────────────────
    def _process_vision(self, scrape_data: dict, append_link: bool = False) -> ProcessedNotice | None:
        """[full / body_only] 이미지 → GPT-4o-mini Vision → Parent + Child 생성."""
        if not os.path.exists(scrape_data["image_path"]):
            logger.error(f"이미지 없음: {scrape_data['image_path']}")
            return None

        img_b64 = self._encode_image(scrape_data["image_path"])
        prompt = (
            f"당신은 서울과학기술대학교(과기대) 공지사항 전문 분석가입니다.\n"
            f"제공된 이미지는 과기대 '{scrape_data['category']}' 게시판의 캡처본입니다.\n\n"
            "수행 과제:\n"
            "1. 이미지 내 모든 텍스트·표를 완벽한 Markdown으로 변환하여 'parent_markdown'으로 반환하세요. "
            "텍스트가 흐릿하거나 불분명한 경우 절대 추측하지 말고 반드시 '[판독불가]'로 표시하세요.\n"
            "2. 검색에 최적화된 3줄 핵심 요약을 'child_summary'로 작성하세요. "
            "첫 번째 줄 맨 앞에 '[과기대]' 태그를 반드시 표기하세요 "
            "(예: '[과기대] 2026-1학기 장학금 신청 - 재학생 대상 2026-03-10까지 접수'). "
            "날짜·금액·신청 대상을 반드시 포함하세요.\n"
            "3. 핵심 키워드 5개와 지원 자격(있을 경우)을 추출하세요.\n\n"
            "반드시 아래 JSON 형식으로만 응답하세요:\n"
            '{"parent_markdown": "...", "child_summary": "...", '
            '"keywords": ["..."], "target_conditions": ["..."]}'
        )
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "너는 데이터 엔지니어링을 위한 고성능 문서 구조화 도구이다."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        analysis = json.loads(resp.choices[0].message.content)
        parent_md = analysis["parent_markdown"]

        # 5페이지 이하 PDF 첨부파일 전 페이지 텍스트를 parent_markdown에 통합
        pdf_attach = scrape_data.get("pdf_attachment")
        if pdf_attach:
            pdf_text = self._extract_pdf_text(
                pdf_attach["url"], pdf_attach.get("orig_name", "")
            )
            if pdf_text:
                parent_md += (
                    f"\n\n---\n## 첨부파일 전문 ({pdf_attach.get('orig_name', 'PDF')})\n\n"
                    f"{pdf_text}"
                )
                logger.info(f"  📎 PDF 본문 통합 완료: {len(pdf_text)}자")

        if append_link:
            parent_md += (
                f"\n\n---\n📌 상세 내용은 [원본 링크]({scrape_data['url']})에서 확인하세요"
            )

        return ProcessedNotice(
            title=scrape_data["title"],
            source_url=scrape_data["url"],
            category=scrape_data["category"],
            parent_markdown=parent_md,
            child_summary=analysis["child_summary"],
            keywords=analysis.get("keywords", []),
            target_conditions=analysis.get("target_conditions", []),
        )

    def _process_excel(self, scrape_data: dict) -> ProcessedNotice | None:
        """[excel] 엑셀 다운로드 → Pandas 마크다운 → GPT Child 생성."""
        excel_path = scrape_data.get("excel_path")

        # 엑셀 파일이 없으면 Vision 처리로 폴백
        if not excel_path or not os.path.exists(excel_path):
            logger.warning("엑셀 파일 없음 → Vision 처리(body_only)로 폴백")
            return self._process_vision(scrape_data, append_link=True)

        excel_md = self._excel_to_markdown(excel_path)
        if not excel_md:
            logger.warning("엑셀 변환 실패 → Vision 처리(body_only)로 폴백")
            return self._process_vision(scrape_data, append_link=True)

        parent_md = (
            f"## {scrape_data['title']}\n\n"
            f"| 항목 | 내용 |\n|------|------|\n"
            f"| 카테고리 | {scrape_data['category']} |\n"
            f"| 원본 URL | {scrape_data['url']} |\n\n"
            f"{excel_md}\n\n"
            f"---\n📌 상세 내용은 [원본 링크]({scrape_data['url']})에서 확인하세요"
        )
        child_summary = self._generate_child_from_text(scrape_data["title"], parent_md)

        return ProcessedNotice(
            title=scrape_data["title"],
            source_url=scrape_data["url"],
            category=scrape_data["category"],
            parent_markdown=parent_md,
            child_summary=child_summary,
            keywords=[],
            target_conditions=[],
        )

    # ── 퍼블릭 인터페이스 ─────────────────────────────────────────────────────
    def process(self, scrape_data: dict) -> ProcessedNotice | None:
        """처리 전략에 따라 분기하여 ProcessedNotice를 반환합니다."""
        strategy = scrape_data.get("strategy", "full")
        logger.info(f"  ⚙️  strategy={strategy}")
        try:
            if strategy == "excel":
                return self._process_excel(scrape_data)
            elif strategy == "body_only":
                return self._process_vision(scrape_data, append_link=True)
            else:  # 'full'
                return self._process_vision(scrape_data, append_link=False)
        except Exception as e:
            logger.error(f"❌ 프로세싱 오류 ({scrape_data.get('title', 'Unknown')}): {e}")
            return None
