from pydantic import BaseModel, Field
from typing import List
import uuid
from datetime import datetime

class ProcessedNotice(BaseModel):
    # 데이터 고유 ID (중복 방지 및 참조용)
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    
    # 기본 정보
    title: str = Field(..., description="공지사항 제목")
    source_url: str = Field(..., description="원본 URL")
    category: str = Field(..., description="대학공지/학사공지/장학공지/대학원공지/취업공지 중 하나")
    
    # Parent Data (RAG 컨텍스트용 원본)
    parent_markdown: str = Field(..., description="VLM이 추출한 원본 마크다운 텍스트")
    
    # Child Data (벡터 검색 최적화용)
    child_summary: str = Field(..., description="검색 성능을 위한 3줄 핵심 요약")
    keywords: List[str] = Field(default=[], description="검색 인덱싱용 핵심 키워드 리스트")
    
    # 메타데이터
    target_conditions: List[str] = Field(default=[], description="대상자 자격 조건 등")
    collected_at: str = Field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    class Config:
        # 데이터 유효성 검사 강화
        validate_assignment = True