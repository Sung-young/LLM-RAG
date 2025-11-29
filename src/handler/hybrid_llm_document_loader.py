from dotenv import load_dotenv
from langchain_aws import ChatBedrock
import os
import io
import tempfile
import fitz
import re
import base64
import pypdf
from typing import List, Union, Optional
import logging

import camelot
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from pdfplumber.utils import get_bbox_overlap, obj_to_bbox, extract_text

import sys
# Add parent directory to path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.loader_modules import to_documents, split_texts
except ModuleNotFoundError:
    from src.utils.loader_modules import to_documents, split_texts

# AWS Bedrock 임포트

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridLLMPdfLoader(BaseLoader):
    """
    테이블 개수에 따라 처리 방식을 결정하는 하이브리드 PDF 로더
    - 테이블 0~1개: pdfplumber로 처리 (기존 방식)
    - 테이블 2개 이상: AWS Bedrock LLM으로 전체 페이지 파싱
    """

    def __init__(
        self,
        file_path: str,
        file: Union[str, io.BytesIO],
        file_name: str,
        # llm_model: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # Cross-region Claude 3.5 Sonnet
        llm_model: str = "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region: str = "us-east-1",
    ):
        """
        file_name이 None일 경우를 대비하여 항상 문자열 값을 갖도록 보장
        """
        self.file_path = file_path
        self.file = file
        if file_name:
            self.file_name = file_name
        elif isinstance(file, str):
            self.file_name = file
        else:
            self.file_name = "in-memory.pdf"

        self.llm_model = llm_model
        self.aws_region = aws_region

        # AWS Bedrock LLM 초기화
        self.llm = ChatBedrock(
            model_id=self.llm_model,
            region_name=self.aws_region,
            model_kwargs={
                "temperature": 0.1,
                "max_tokens": 4096,
            },
        )
        logger.info(f"AWS Bedrock LLM 초기화 완료: {self.llm_model}")

    def load(self) -> List[Document]:
        """메인 로더 함수"""
        tmp_path = None
        # BytesIO 객체를 임시 파일로 저장하여 경로 기반으로 처리
        if isinstance(self.file, io.BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                self.file.seek(0)
                tmp.write(self.file.read())
                tmp_path = tmp.name
            file_path = tmp_path
        elif isinstance(self.file, str):
            if not os.path.exists(self.file):
                raise FileNotFoundError(f"Path does not exist: {self.file}")
            file_path = self.file
        else:
            raise TypeError(
                "file must be a filesystem path (str) or a io.BytesIO object")

        try:
            # 하이브리드 로직 실행
            all_docs = self._load_hybrid_with_llm(file_path)
            for doc in all_docs:
                doc.metadata["is_clause"] = False

        finally:
            # 임시 파일이 생성되었다면 삭제
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return all_docs

    def _extract_content(self, page_area):
        """[Plumber 전용] 주어진 영역에서 'table 영역 분리 후 재조립' 로직 실행"""
        chars = list(page_area.chars)
        tables = page_area.find_tables()
        table_bboxes = [table.bbox for table in tables]

        non_table_chars = [
            char
            for char in chars
            if not any(get_bbox_overlap(obj_to_bbox(char), bbox) for bbox in table_bboxes)
        ]

        processed_tables = []
        for table in tables:
            table_crop = page_area.crop(table.bbox)
            if not table_crop.chars:
                continue

            first_table_char = table_crop.chars[0]
            df = pd.DataFrame(table.extract())
            if not df.empty:
                df.columns = df.iloc[0]
                markdown = df.drop(0).to_markdown(index=False)
                processed_tables.append(first_table_char | {"text": markdown})

        final_chars = non_table_chars + processed_tables
        return extract_text(final_chars, layout=True)

    def _two_column(self, page):
        """[Plumber 전용] 2단 페이지를 처리"""
        page_content_items = []
        center_x = page.width / 2

        # 왼쪽 컬럼 처리
        left_bbox = (0, 0, center_x, page.height)
        left_text = self._extract_content(page.crop(left_bbox))
        if left_text and left_text.strip():
            page_content_items.append((0, "text", left_text))

        # 오른쪽 컬럼 처리
        right_bbox = (center_x, 0, page.width, page.height)
        right_text = self._extract_content(page.crop(right_bbox))
        if right_text and right_text.strip():
            page_content_items.append((1, "text", right_text))

        return page_content_items

    def _single_column(self, page, tables_on_page):
        """[Camelot + Plumber] 1단 페이지를 처리하는 하이브리드 메서드"""
        page_content_items = []
        table_bboxes = [t._bbox for t in tables_on_page]

        # 1. Camelot 테이블 콘텐츠 추가
        for table in tables_on_page:
            y_pos = page.height - table._bbox[3]
            page_content_items.append((y_pos, "table", table))

        # 2. Plumber로 테이블 외부 텍스트 추출
        all_words = page.extract_words()
        non_table_words = []
        for word in all_words:
            is_in_table = False
            for bbox in table_bboxes:
                table_bbox_plumber = (
                    bbox[0],
                    page.height - bbox[3],
                    bbox[2],
                    page.height - bbox[1],
                )
                word_center_x = (word["x0"] + word["x1"]) / 2
                word_center_y = (word["top"] + word["bottom"]) / 2
                if (
                    word_center_x >= table_bbox_plumber[0]
                    and word_center_x <= table_bbox_plumber[2]
                    and word_center_y >= table_bbox_plumber[1]
                    and word_center_y <= table_bbox_plumber[3]
                ):
                    is_in_table = True
                    break
            if not is_in_table:
                non_table_words.append(word)

        if non_table_words:
            full_text = " ".join(
                [w["text"] for w in sorted(
                    non_table_words, key=lambda w: (w["top"], w["x0"]))]
            )
            y_pos = non_table_words[0]["top"] if non_table_words else 0
            page_content_items.append((y_pos, "text", full_text))

        return page_content_items

    def _is_two_column(self, page):
        """페이지가 2단 컬럼인지 판단"""
        center_x = page.width / 2
        words = page.extract_words()
        if not words:
            return False

        left = any(w["x1"] < center_x for w in words)
        right = any(w["x0"] > center_x for w in words)
        crossing = sum(
            1 for w in words if w["x0"] < center_x and w["x1"] > center_x)
        ratio = crossing / len(words)

        return (ratio < 0.05) and left and right

    def _is_bbox_overlap(self, bbox1, bbox2, threshold=0.3):
        """
        두 bbox가 겹치는지 확인 (IoU 기반)
        threshold: IoU 값이 이 값보다 크면 겹친다고 판단
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # 교집합 영역 계산
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return False

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # 각 bbox 영역
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        # IoU (Intersection over Union)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        return iou > threshold

    def _split_pdf_by_pages(self, pdf_bytes: bytes) -> list[bytes]:
        """PDF를 페이지별로 분할하여 각 페이지를 개별 PDF 바이트로 반환"""
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            page_pdfs = []

            for page_num in range(len(pdf_reader.pages)):
                pdf_writer = pypdf.PdfWriter()
                pdf_writer.add_page(pdf_reader.pages[page_num])

                page_bytes = io.BytesIO()
                pdf_writer.write(page_bytes)
                page_pdfs.append(page_bytes.getvalue())

            logger.info(f"PDF를 {len(page_pdfs)}개 페이지로 분할 완료")
            return page_pdfs

        except Exception as e:
            logger.error(f"PDF 페이지 분할 실패: {str(e)}")
            return [pdf_bytes]

    def _llm_parse_page(self, page_pdf_bytes: bytes, page_num: int) -> str:
        """
        AWS Bedrock LLM을 사용하여 페이지 전체를 파싱
        복잡한 테이블 구조를 정확히 추출
        """
        try:
            base64_data = base64.b64encode(page_pdf_bytes).decode()

            prompt = """
            당신은 이미지에서 텍스트를 완벽하게 추출하는 OCR 전문가입니다. RAG 시스템의 지식 베이스 구축을 위해 이미지의 모든 텍스트 내용을 빠짐없이 추출해야 합니다.

        **핵심 목표: 완전한 텍스트 추출 (요약 금지)**
        - 이미지에 있는 모든 텍스트를 하나도 빠뜨리지 않고 추출
        - 원본 텍스트의 정확성과 완전성을 최우선으로 함
        - 반복되는 페이지 하단 문구(고정 안내문, 회사 홍보문구, 교육용 문구)는 자동으로 제외
        - 어떤 내용도 요약하거나 생략하지 않음

        **완전 추출 지침:**

        1. **전체 텍스트 추출:**
            - 제목, 부제목, 본문, 각주, 캡션 등 모든 텍스트
            - 헤더, 페이지 번호까지 포함
            - 작은 글씨나 흐릿한 텍스트도 최대한 정확히 추출
            - 광고, 연락처 등도 모두 포함

        2. **구조 정보 유지:**
            - **제목 계층:** 크기와 굵기에 따라 # ## ### 마크다운으로 표현
            - **단락 구분:** 원본의 단락과 줄바꿈 구조 그대로 유지
            - **목록:** 번호나 불릿포인트 형식 원본 그대로 재현
            - **표:** 모든 행과 열을 마크다운 테이블로 완전 변환

        3. **정확한 데이터 추출:**
            - **숫자 데이터:** 금액, 비율, 날짜, 전화번호 등 한 글자도 틀리지 않게
            - **고유명사:** 회사명, 상품명, 인명 등 정확한 표기
            - **특수문자:** 괄호, 하이픈, 슬래시 등 모든 기호 포함

        4. **시각적 요소 텍스트화:**
            - **표와 차트:** 모든 데이터를 표 형식으로 완전 변환
            - **이미지 속 텍스트:** 로고나 이미지 안의 텍스트도 모두 추출
            - **도식/다이어그램:** 텍스트 요소만 추출하고 `[도식: 설명]` 추가
            - **순수 이미지:** `[이미지: 간단한 설명]`로만 표시

        5. **OCR 품질 최적화:**
            - 불분명한 글자는 맥락을 고려해 추정하되, 확실하지 않으면 `[불분명: ?]` 표시
            - 명백한 OCR 오류(예: "보험험"→"보험")는 수정
            - 띄어쓰기와 문장부호를 자연스럽게 보정

        6. **출력 형식 - 순수 텍스트만:**
            - 메타데이터나 분석 내용 추가 금지
            - "다음은 추출한 내용입니다" 같은 서론 금지
            - 오직 이미지의 원본 텍스트만 출력
            - 마크다운 구조화는 허용 (가독성을 위해)
            
        7. **페이지 내 구분선·섹션 구조 자동 판단:**
            - LLM은 PDF 페이지를 분석하여 시각적·의미적 구분을 스스로 판단
            - 한 페이지 내에는 반드시 여러 <div> 가 존재
            - 이미지 내 <div> 형태를 만들 시 <div> 태그 바로 옆에 **해당 페이지 제목**을 첫 줄에 반드시 표기
             - **해당 페이지 제목 판단 기준 (우선순위)**:
                1. **가장 큰 폰트 + 중앙 또는 상단 중앙 정렬 + 강조색(진한색/빨강/굵은색)** → 페이지의 **주요 제목(main title)**
                2. **작은 배너나 왼쪽 상단, 회색/녹색 박스 형태 문구** (예: "이달의 이슈", "투자형 변액연금") → **부제(subtitle)** 로 간주 (제목으로 표기하지 말 것)
            - 페이지 내 내용이 주제별(예: ‘통합건강’, ‘3대(癌)치료’, ‘수술’, ‘입원’)로 명확히 구분되는 경우, 각 섹션을 <div> 태그로 감싸고, 해당 섹션의 제목을 <div>태그 내 두번째 줄에 포함
            - 페이지 내 보험회사, 보험명이 있는 경우, <div>태그 내 두번째 줄에 포함
            - <div> 태그 안에 표가 있는 경우 , <table> 태그는 div 태그 안에 위치시키기
        
            예시:
                <div> 해당 페이지 제목
                통합건강 [라이나생명(보험회사) 골라담는 건강보험(보험이름)]
                (통합건강에 해당하는 모든 내용)
                </div>

                <div> 해당 페이지 제목
                3대치료 [한화손해보험 00기초플랜]
                (3대 치료에 해당하는 모든 내용)
                <table> 3대치료
                | 항목 | 내용 |
                </table>
                </div>
            - 구분 판단 기준:
                - 폰트 크기나 굵기가 다르거나
                - 색상 블록이나 시각적 구획선이 존재하거나
                - 명확히 다른 주제(카테고리)로 나뉘는 경우 -> 별도 <div>로 분리
                - 만약 한 페이지에 2개 이상의 주요 구역이 존재하면, 각 구역을 별도 <div>로 감싸고 줄바꿈 처리
                - div 태그 안에 표가 포함될 경우, 표 태그는 div 태그 안에 위치시키기
            - <div>안에 내용은 절대 요약하거나 생략하지 말고, 모든 내용을 완전하게 포함

        8. **표 구조 자동 판단 및 태깅: **
            - 이미지 내 표 형태의 정보가 발견되면 반드시 <table> 태그로 감싼 후 <table> 태그 바로 옆에 '해당 페이지 제목'을 표기
                ** 단 table 태그가 div 태그 안에 표인 경우 <table>태그 옆에 div 태그 제목을 표기 **
            - 예시:
                <table> 펫보험 비교
                | 보험상품 | 보장금액 | 납입기간 | 보험료 |
                | A플랜 | 500만원 | 20년 | 3만원 |
                </table>
            - 표의 시작과 끝을 명확히 구분하고, 표가 끝난 뒤에는 줄바꿈 후 다음 내용을 계속 이어서 작성
            - 시각적으로 표처럼 보이지만 셀 경계가 불분명한 경우라도, 열 구조를 추론하여 가능한 한 정확하게 표 형태로 변환
            - 표안에 내용은 절대 요약하거나 생략하지 말고, 모든 내용을 완전하게 포함
            

        **절대 금지 사항:**
        - 내용 요약이나 압축
        - 중요도에 따른 선별적 추출
        - 개인적 해석이나 분석 추가
        - 원본에 없는 설명이나 주석

        **완전성 체크리스트:**
        □ 모든 페이지의 텍스트 추출 완료
        □ 표, 목록, 각주까지 모든 구조적 요소 포함
        □ 숫자, 날짜, 고유명사의 정확성 확인

        ---
        이제 주어진 이미지에서 모든 텍스트를 완전히 추출하세요. 한 글자도 놓치지 마세요.
            """

            logger.info(f"페이지 {page_num} LLM 파싱 시작...")

            # AWS Bedrock은 PDF 직접 전송을 지원하지 않으므로
            # PDF를 이미지로 변환하여 전송
            pdf_document = fitz.open(stream=page_pdf_bytes, filetype="pdf")
            try:
                page = pdf_document[0]

                # 고해상도로 이미지 변환 (DPI 높이기)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2배 확대
                img_bytes = pix.tobytes("png")
                img_base64 = base64.b64encode(img_bytes).decode()

                # Bedrock 메시지 형식으로 구성
                message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }

                # LLM 호출
                response = self.llm.invoke([message])

                # 응답에서 텍스트 추출
                if hasattr(response, "content"):
                    extracted_text = response.content
                else:
                    extracted_text = str(response)

                logger.info(
                    f"페이지 {page_num} LLM 파싱 완료 (추출된 텍스트 길이: {len(extracted_text)}자)")
                return extracted_text
            
            finally:
                # PDF 문서를 명시적으로 닫아서 메모리 누수 방지
                pdf_document.close()

        except Exception as e:
            logger.error(f"페이지 {page_num} LLM 파싱 실패: {str(e)}")
            return f"[LLM 파싱 실패: {str(e)[:100]}]"

    def _load_hybrid_with_llm(self, file: Union[str, io.BytesIO]) -> List[Document]:
        """
        하이브리드 처리 로직:
        - 테이블 0~1개: pdfplumber (기존 방식)
        - 테이블 2개 이상: LLM 전체 페이지 파싱
        """
        tmp_path = None
        if isinstance(file, io.BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                file.seek(0)
                tmp.write(file.read())
                tmp_path = tmp.name
            file_path = tmp_path
        else:
            file_path = os.path.abspath(file)

        all_docs: List[Document] = []
        doc_idx = 0

        # PDF 바이트 읽기 (LLM 처리용)
        with open(file_path, "rb") as f:
            pdf_bytes = f.read()
        page_pdf_list = self._split_pdf_by_pages(pdf_bytes)

        try:
            try:
                pdf = pdfplumber.open(file_path)
            except Exception as e:
                logger.error(f"pdfplumber.open() 실패 → 파일 전체 건너뜀: {e}")
                return []

            total_pages = len(pdf.pages)
            logger.info(f"총 {total_pages}페이지 처리 시작")

            for i in range(1, total_pages + 1):
                try:
                    page = pdf.pages[i - 1]
                except Exception as e:
                    logger.error(f"페이지 로딩 실패 (p{i}) → 건너뜀: {e}")
                    continue

                try:
                    # 1단계: Camelot으로 테이블 탐지 (lattice만 사용)
                    tables_on_page = []
                    lattice_count = 0

                    # lattice (선이 있는 표) - 공식 문서는 대부분 선이 있는 표 사용
                    try:
                        tables_lattice = camelot.read_pdf(
                            file_path,
                            pages=str(i),
                            flavor="lattice",
                            suppress_stdout=True,
                            # 파라미터 조정: 표를 과도하게 분리하지 않도록
                            line_scale=15,  # 선 감지 민감도 낮춤 (기본값 15) - 작은 선은 무시
                            copy_text=None,  # 텍스트만 있는 영역은 표로 간주하지 않음
                        )
                        if tables_lattice and len(tables_lattice) > 0:
                            # 유효한 표만 필터링
                            for t in tables_lattice:
                                df = t.df
                                # 조건 1: 최소 3행 이상
                                # 조건 2: 최소 2열 이상 (1열짜리는 표가 아님)
                                # 조건 3: 비어있지 않은 셀이 있어야 함
                                non_empty_cells = df.apply(lambda x: x.astype(
                                    str).str.strip().ne('')).sum().sum()

                                if (df.shape[0] >= 2 and
                                        df.shape[1] >= 2):
                                    tables_on_page.append(t)
                                    logger.debug(f"페이지 {i} - 유효한 표로 인정됨")
                                else:
                                    logger.debug(
                                        f"페이지 {i} - 표로 인정되지 않음 (필터링됨)")

                            lattice_count = len(tables_on_page)
                    except Exception as e:
                        logger.debug(f"페이지 {i}: lattice 방식 실패 - {str(e)}")

                    table_count = len(tables_on_page)
                    logger.info(
                        f"페이지 {i}: 테이블 {table_count}개 발견 (lattice: {lattice_count})")

                    # 2단계: 테이블 개수에 따라 분기 처리
                    if table_count >= 2:
                        # ⭐ 테이블 2개 이상 → LLM 처리
                        logger.info(f"페이지 {i}: 테이블 2개 이상 → LLM으로 처리")

                        page_pdf_bytes = page_pdf_list[i - 1]
                        extracted_text = self._llm_parse_page(
                            page_pdf_bytes, i)

                        if extracted_text and extracted_text.strip():
                            text_chunks = split_texts(extracted_text)

                            new_docs, doc_idx = to_documents(
                                file_path=self.file_name,
                                file_name=self.file_name,
                                text_chunks=text_chunks,
                                doc_idx=doc_idx,
                                page_num=i,
                                total_pages=total_pages,
                            )

                            for doc in new_docs:
                                doc.metadata.update({
                                    "extract_mode": "llm_complex_table",
                                    "table_count": table_count
                                })

                            all_docs.extend(new_docs)
                            logger.info(
                                f"페이지 {i} LLM 처리 완료 - 청크 수: {len(text_chunks)}")
                        else:
                            logger.warning(f"페이지 {i} LLM 추출 결과가 비어있음")

                    else:
                        # 테이블 0~1개 → pdfplumber 방식
                        logger.info(
                            f"페이지 {i}: 테이블 {table_count}개 → pdfplumber로 처리")

                        page_content_items = []
                        extract_mode_suffix = ""

                        if table_count == 0:
                            # 테이블 없음
                            try:
                                if self._is_two_column(page):
                                    page_content_items = self._two_column(page)
                                else:
                                    page_content_items = [
                                        (0, "text", page.extract_text())]
                                extract_mode_suffix = "plumber_only"
                            except Exception as e:
                                logger.error(
                                    f"plumber fallback 실패 (p{i}) → 건너뜀: {e}")
                                continue

                        else:
                            # 테이블 1개
                            try:
                                is_two_col = self._is_two_column(page)
                                if is_two_col:
                                    page_content_items = self._two_column(page)
                                    extract_mode_suffix = "2col"
                                else:
                                    page_content_items = self._single_column(
                                        page, tables_on_page)
                                    extract_mode_suffix = "1col"
                            except Exception as e:
                                logger.error(
                                    f"table/text hybrid 처리 실패 (p{i}) → 건너뜀: {e}")
                                continue

                        # 정렬
                        try:
                            page_content_items.sort(key=lambda item: item[0])
                        except Exception:
                            pass

                        # Document 생성
                        for _, item_type, content in page_content_items:
                            try:
                                text_chunks = []
                                page_number = i

                                if item_type == "text":
                                    if content and content.strip():
                                        text_chunks = split_texts(content)

                                elif item_type == "table":
                                    df: pd.DataFrame = content.df.copy()
                                    df = df.fillna("").map(
                                        lambda x: " ".join(str(x).split()))

                                    # 마크다운 테이블로 변환 (표 구조 유지)
                                    try:
                                        table_text = df.to_markdown(
                                            index=False)
                                        logger.debug(
                                            f"페이지 {i} - 표를 마크다운으로 변환 성공")
                                    except Exception as e:
                                        # to_markdown 실패 시 fallback
                                        logger.warning(
                                            f"페이지 {i} - to_markdown 실패, fallback 사용: {str(e)}")
                                        table_text = df.to_string(
                                            index=False, header=False)

                                    if table_text.strip():
                                        text_chunks = split_texts(table_text)
                                    page_number = content.page

                                if not text_chunks:
                                    continue

                                new_docs, doc_idx = to_documents(
                                    file_path=self.file_name,
                                    file_name=self.file_name,
                                    text_chunks=text_chunks,
                                    doc_idx=doc_idx,
                                    page_num=page_number,
                                    total_pages=total_pages,
                                )

                                for doc in new_docs:
                                    doc.metadata.update({
                                        "extract_mode": f"{item_type}_{extract_mode_suffix}",
                                        "table_count": table_count
                                    })

                                all_docs.extend(new_docs)

                            except Exception as e:
                                logger.error(f"Document 생성 실패: {e}")
                                continue

                except Exception as e:
                    logger.error(f"페이지 전체 처리 실패 (p{i}) → 건너뜀: {e}")
                    continue

            logger.info(f"전체 처리 완료 - 총 {len(all_docs)}개 문서 생성")

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return all_docs
