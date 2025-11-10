import os
import io
import tempfile
import fitz
import re
from typing import List, Union, Optional

import camelot
import pandas as pd
import pdfplumber
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader

from pdfplumber.utils import get_bbox_overlap, obj_to_bbox, extract_text

from utils.loader_modules import to_documents, split_texts



class PdfLoader(BaseLoader):

    def __init__(self, file_path: str, file: Union[str, io.BytesIO], file_name: str,):
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
            raise TypeError("file must be a filesystem path (str) or a io.BytesIO object")
        
        try:
            # 1. 텍스트/표 추출 
            general_docs = self._load_hybrid(file_path)
            for doc in general_docs:
                doc.metadata["is_clause"] = False # 메타데이터 추가

            all_docs = general_docs 

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
            char for char in chars 
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
        #  반환 형식을 (y_pos, type, content) 튜플 리스트로 통일
        page_content_items = []
        center_x = page.width / 2
        
        # 왼쪽 컬럼 처리
        left_bbox = (0, 0, center_x, page.height)
        left_text = self._extract_content(page.crop(left_bbox))
        if left_text and left_text.strip():
            # y_pos를 0으로 설정하여 순서상 가장 앞에 오도록 함
            page_content_items.append((0, 'text', left_text))
        
        # 오른쪽 컬럼 처리
        right_bbox = (center_x, 0, page.width, page.height)
        right_text = self._extract_content(page.crop(right_bbox))
        if right_text and right_text.strip():
            # y_pos를 1로 설정하여 왼쪽 컬럼 다음에 오도록 함
            page_content_items.append((1, 'text', right_text))
        
        return page_content_items

    def _single_column(self, page, tables_on_page):
        """[Camelot + Plumber] 1단 페이지를 처리하는 하이브리드 메서드"""
        page_content_items = []
        # Camelot이 찾은 테이블의 bbox를 사용
        table_bboxes = [t._bbox for t in tables_on_page]
        
        # 1. Camelot 테이블 콘텐츠 추가
        for table in tables_on_page:
            y_pos = page.height - table._bbox[3]
            page_content_items.append((y_pos, 'table', table))

        # 2. Plumber로 테이블 외부 텍스트 추출
        all_words = page.extract_words()
        non_table_words = []
        for word in all_words:
            is_in_table = False
            for bbox in table_bboxes:
                table_bbox_plumber = (bbox[0], page.height - bbox[3], bbox[2], page.height - bbox[1])
                word_center_x = (word['x0'] + word['x1']) / 2
                word_center_y = (word['top'] + word['bottom']) / 2
                if (word_center_x >= table_bbox_plumber[0] and word_center_x <= table_bbox_plumber[2] and
                    word_center_y >= table_bbox_plumber[1] and word_center_y <= table_bbox_plumber[3]):
                    is_in_table = True
                    break
            if not is_in_table:
                non_table_words.append(word)
        
        if non_table_words:
            full_text = " ".join([w['text'] for w in sorted(non_table_words, key=lambda w: (w['top'], w['x0']))])
            y_pos = non_table_words[0]['top'] if non_table_words else 0
            page_content_items.append((y_pos, 'text', full_text))
            
        return page_content_items
    
    def _is_two_column(self, page):
        center_x = page.width / 2
        words = page.extract_words()
        if not words:
            return False

        left = any(w["x1"] < center_x for w in words)
        right = any(w["x0"] > center_x for w in words)
        crossing = sum(1 for w in words if w["x0"] < center_x and w["x1"] > center_x)
        ratio = crossing / len(words)

        return (ratio < 0.05) and left and right

    def _load_hybrid(self, file: Union[str, io.BytesIO]) -> List[Document]:
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

        try:
            try:
                pdf = pdfplumber.open(file_path)
            except Exception as e:
                print(f"[ERROR] pdfplumber.open() 실패 → 파일 전체 건너뜀: {e}")
                return []

            total_pages = len(pdf.pages)

            for i in range(1, total_pages + 1):
                # 페이지 접근 자체 보호
                try:
                    page = pdf.pages[i - 1]
                except Exception as e:
                    print(f"[ERROR] 페이지 로딩 실패 (p{i}) → 건너뜀: {e}")
                    continue

                # 페이지 처리 전체 
                try:
                    page_content_items = []
                    extract_mode_suffix = ""

                    #  시도 자체 보호
                    try:
                        tables_on_page = camelot.read_pdf(
                            file_path,
                            pages=str(i),
                            flavor="stream",
                            suppress_stdout=True
                        )
                    except Exception:
                        tables_on_page = []

                    # 테이블 없는 경우 fallback
                    if not tables_on_page or len(tables_on_page) == 0:
                        try:
                            if self._is_two_column(page):
                                page_content_items = self._two_column(page)
                            else:
                                page_content_items = [(0, "text", page.extract_text())]
                            extract_mode_suffix = "plumber_only"

                        except Exception as e:
                            print(f"[ERROR] plumber fallback 실패 (p{i}) → 건너뜀: {e}")
                            continue

                    else:
                        # 테이블 + 텍스트 혼합 처리
                        try:
                            is_two_col = self._is_two_column(page)
                            if is_two_col:
                                page_content_items = self._two_column(page)
                                extract_mode_suffix = "2col"
                            else:
                                page_content_items = self._single_column(page, tables_on_page)
                                extract_mode_suffix = "1col"
                        except Exception as e:
                            print(f"[ERROR] table/text hybrid 처리 실패 (p{i}) → 건너뜀: {e}")
                            continue

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
                                df = df.fillna("").map(lambda x: " ".join(str(x).split()))
                                table_text = df.to_string(index=False, header=False)
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
                                total_pages=total_pages
                            )

                            for doc in new_docs:
                                doc.metadata.update({"extract_mode": f"{item_type}_{extract_mode_suffix}"})

                            all_docs.extend(new_docs)

                        except Exception as e:
                            print(f"[ERROR] Document 생성 실패: {e}")
                            continue

                except Exception as e:
                    print(f"[ERROR] 페이지 전체 처리 실패 (p{i}) → 건너뜀: {e}")
                    continue

        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return all_docs