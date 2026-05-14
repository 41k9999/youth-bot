# app/services/scraper.py
import os
import re
import io
import logging
import requests
from playwright.sync_api import sync_playwright, Page

logger = logging.getLogger(__name__)

_RE_DOWNLOAD = re.compile(r"downloadfile\s*\(\s*'([^']+)'\s*,\s*'([^']+)'\s*,\s*'([^']+)'\s*\)")
_RE_PREVIEW  = re.compile(r'docPreview\s*\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)')


class SeoulTechScraper:
    BASE_URL = "https://www.seoultech.ac.kr"

    def __init__(self) -> None:
        self.target_boards = {
            "대학공지":  "/service/info/notice/",
            "학사공지":  "/service/info/matters/",
            "장학공지":  "/service/info/janghak/",
            "대학원공지": "/service/info/graduate/",
            "취업공지":  "/service/info/job/",
        }
        self.history_file = "data/scrape_history.txt"

    # ── 히스토리 관리 ─────────────────────────────────────────────────────────
    def _get_history(self) -> set:
        os.makedirs("data", exist_ok=True)
        if os.path.exists(self.history_file):
            with open(self.history_file, "r", encoding="utf-8") as f:
                return set(f.read().splitlines())
        return set()

    def _save_history(self, url: str) -> None:
        os.makedirs("data", exist_ok=True)
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(f"{url}\n")

    @staticmethod
    def _normalize_url(url: str) -> str:
        """nowpage 파라미터를 제거한 정규화 URL을 반환합니다.

        SeoulTech 공지 URL에는 nowpage=N 이 포함되어 페이지마다 달라지므로,
        bidx(게시글 ID) 기준의 정규 URL로 변환해 중복을 방지합니다.
        """
        import urllib.parse
        parsed = urllib.parse.urlparse(url)
        qs = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
        qs.pop("nowpage", None)
        new_query = urllib.parse.urlencode(qs, doseq=True)
        return urllib.parse.urlunparse(parsed._replace(query=new_query))

    # ── 첨부파일 분석 ─────────────────────────────────────────────────────────
    def _analyze_attachments(self, page: Page) -> dict:
        """웹페이지의 ul.list_attach 내 li 항목을 파싱해 첨부파일 목록과 처리 전략을 반환합니다."""
        attachments = []

        for li in page.locator("ul.list_attach li").all():
            a_els = li.locator("a").all()
            for a in a_els:
                href = a.get_attribute("href") or ""
                m = _RE_DOWNLOAD.search(href)
                if m:
                    dir_path, hash_name, orig_name = m.groups()
                    attachments.append(self._build_attachment(dir_path, hash_name, orig_name))

            span_els = li.locator("span[onclick]").all()
            for span in span_els:
                onclick = span.get_attribute("onclick") or ""
                m = _RE_PREVIEW.search(onclick)
                if m:
                    _, dir_path, hash_name, orig_name = m.groups()
                    attachments.append(self._build_attachment(dir_path, hash_name, orig_name))

        types = {a["file_type"] for a in attachments}
        primary_excel = next((a for a in attachments if a["file_type"] == "excel"), None)
        pdf_attachment = None

        if primary_excel:
            strategy = "excel"
        elif "zip" in types:
            strategy = "body_only"
        elif "pdf" in types:
            pdf = next(a for a in attachments if a["file_type"] == "pdf")
            page_count = self._get_pdf_page_count(pdf["url"])
            if page_count <= 5:
                strategy = "full"
                pdf_attachment = pdf
            else:
                strategy = "body_only"
            logger.debug("  PDF %dp → strategy=%s", page_count, strategy)
        else:
            strategy = "full"

        logger.debug("  첨부 %d건 | 전략: %s", len(attachments), strategy)
        return {
            "attachments":    attachments,
            "strategy":       strategy,
            "primary_excel":  primary_excel,
            "pdf_attachment": pdf_attachment,
        }

    @staticmethod
    def _build_attachment(dir_path: str, hash_name: str, orig_name: str) -> dict:
        name_lower = orig_name.lower()
        if any(name_lower.endswith(e) for e in (".xlsx", ".xls")):
            file_type = "excel"
        elif name_lower.endswith(".zip"):
            file_type = "zip"
        elif name_lower.endswith(".pdf"):
            file_type = "pdf"
        elif any(name_lower.endswith(e) for e in (".hwp", ".hwpx", ".doc", ".docx", ".ppt", ".pptx")):
            file_type = "doc"
        elif any(name_lower.endswith(e) for e in (".jpg", ".jpeg", ".png", ".gif")):
            file_type = "image"
        else:
            file_type = "other"

        url = f"https://www.seoultech.ac.kr{dir_path}/{hash_name}"
        return {"file_type": file_type, "url": url, "orig_name": orig_name}

    # ── PDF 페이지 수 확인 ────────────────────────────────────────────────────
    def _get_pdf_page_count(self, url: str) -> int:
        try:
            from pypdf import PdfReader
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            reader = PdfReader(io.BytesIO(resp.content))
            return len(reader.pages)
        except Exception as e:
            logger.warning("  PDF 페이지 수 확인 실패: %s → body_only 처리", e)
            return 999

    # ── 엑셀 다운로드 ─────────────────────────────────────────────────────────
    def _download_excel(self, url: str, category: str, orig_name: str) -> str | None:
        try:
            save_dir = f"data/excel/{category}"
            os.makedirs(save_dir, exist_ok=True)
            safe_name = re.sub(r"[^\w\-_.]", "_", orig_name)
            filepath = os.path.join(save_dir, safe_name)
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                f.write(resp.content)
            logger.info("  엑셀 다운로드: %s", filepath)
            return filepath
        except Exception as e:
            logger.error("  엑셀 다운로드 실패: %s", e)
            return None

    # ── 페이지 번호 클릭 ──────────────────────────────────────────────────────
    def _click_page_number(self, page: Page, page_num: int) -> bool:
        """div.dc-paging 내 특정 숫자 링크를 클릭합니다.

        SeoulTech 게시판은 href='javascript:dcPaging(N)' 방식을 사용하므로
        URL 추출이 불가능 — Playwright로 직접 클릭합니다.
        """
        try:
            paging = page.locator("div.dc-paging")
            if paging.count() == 0:
                logger.debug("  div.dc-paging 없음")
                return False
            links = paging.first.locator("a")
            for i in range(links.count()):
                link = links.nth(i)
                if (link.inner_text() or "").strip() == str(page_num):
                    link.click()
                    page.wait_for_load_state("networkidle")
                    return True
        except Exception as e:
            logger.debug("  p.%d 클릭 실패: %s", page_num, e)
        return False

    # ── 메인 수집 로직 ────────────────────────────────────────────────────────
    def capture_new_notices(self, pages_per_board: int = 10) -> list:
        """각 게시판을 최대 pages_per_board 페이지까지 순회하며 신규 공지를 수집합니다.

        Phase 1: dcPaging(N) 클릭으로 전 페이지 링크를 먼저 수집합니다.
        Phase 2: 수집된 신규 링크를 순차 방문하여 스크린샷을 찍습니다.
        """
        history = self._get_history()
        new_items = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            for category, sub_url in self.target_boards.items():
                board_url = self.BASE_URL + sub_url
                logger.info("📋 [%s] Phase 1: 링크 수집 시작 (최대 %d 페이지)", category, pages_per_board)

                # ── Phase 1: 전 페이지 링크 수집 ─────────────────────────────
                page.goto(board_url, wait_until="networkidle")
                all_links: list[dict] = []

                for page_num in range(1, pages_per_board + 1):
                    try:
                        rows = page.locator("table.tbl_list td.tit a")
                        count = rows.count()

                        if count == 0:
                            logger.info("  [%s] p.%d 항목 없음 — 수집 종료", category, page_num)
                            break

                        for i in range(count):
                            link = rows.nth(i)
                            all_links.append({
                                "title":    link.inner_text().strip(),
                                "href":     link.get_attribute("href"),
                                "page_num": page_num,
                            })

                        logger.info("  [%s] p.%d: %d건 링크 확인 (누적 %d건)", category, page_num, count, len(all_links))

                        if page_num < pages_per_board:
                            if not self._click_page_number(page, page_num + 1):
                                logger.info("  [%s] 마지막 페이지 도달 (p.%d)", category, page_num)
                                break
                    except Exception as e:
                        logger.error("❌ [%s] p.%d 목록 로드 에러: %s", category, page_num, e)
                        break

                # ── Phase 2: 신규 아이템 처리 ────────────────────────────────
                new_in_board = 0
                for link_data in all_links:
                    title    = link_data["title"]
                    href     = link_data["href"]
                    raw_url  = board_url + href
                    item_url = self._normalize_url(raw_url)

                    if item_url in history:
                        continue

                    try:
                        page.goto(raw_url, wait_until="networkidle")

                        attach_info = self._analyze_attachments(page)
                        strategy    = attach_info["strategy"]

                        safe_title = re.sub(r"[^\w\s]", "", title).strip()[:15]
                        img_dir    = f"images/{category}"
                        img_path   = f"{img_dir}/{safe_title}.png"
                        os.makedirs(img_dir, exist_ok=True)

                        content_area = page.locator("div.wrap_view")
                        if content_area.count() == 0:
                            logger.warning("  ⚠️ 본문 영역 없음: %s", title[:20])
                        else:
                            content_area.screenshot(path=img_path)

                            item = {
                                "title":          title,
                                "category":       category,
                                "url":            item_url,
                                "image_path":     img_path,
                                "strategy":       strategy,
                                "attachments":    attach_info["attachments"],
                                "excel_path":     None,
                                "pdf_attachment": attach_info["pdf_attachment"],
                            }

                            if strategy == "excel" and attach_info["primary_excel"]:
                                excel = attach_info["primary_excel"]
                                item["excel_path"] = self._download_excel(
                                    excel["url"], category, excel["orig_name"]
                                )

                            new_items.append(item)
                            self._save_history(item_url)
                            history.add(item_url)
                            new_in_board += 1
                            logger.info("  ✅ [%s] p.%d 수집: %s",
                                        category, link_data["page_num"], title[:30])

                    except Exception as e:
                        logger.error("  ❌ [%s] 아이템 에러 (%s): %s", category, title[:20], e)
                    finally:
                        try:
                            page.goto(board_url, wait_until="networkidle")
                        except Exception as nav_err:
                            logger.warning("  ⚠️ 페이지 복귀 실패: %s", nav_err)

                logger.info("  [%s] 완료: 신규 %d건 / 전체 링크 %d건", category, new_in_board, len(all_links))

            browser.close()

        return new_items
