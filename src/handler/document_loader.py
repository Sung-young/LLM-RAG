from typing import Union, Callable, Dict
import os
import json
import io
import pdfplumber
import tempfile
import pandas as pd
import re
import subprocess
import time
import asyncio
import pypdf
from tqdm import tqdm

import traceback

from PIL import Image

from langchain_upstage import UpstageDocumentParseLoader

from pathlib import Path
from langchain_core.document_loaders import BaseLoader
from langchain_community.document_loaders import (
    UnstructuredHTMLLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredCSVLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from langchain_teddynote.document_loaders import HWPLoader

import sys
# Add parent directory to path for direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.loader_modules import (
    preprocessing,
    split_texts,
    to_documents,
    ocr_with_gpt,
    ocr_pdf_with_gpt,
)
# from image_util import ImageUtil
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomDocumentLoader(BaseLoader):
    def __init__(
        self,
        file: Union[str, io.BytesIO],
        file_path: str,
        file_name: str,
    ) -> None:
        self.file = file
        self.file_path = file_path
        self.file_name = (
            file_name if file_name is not None else file_path.split("/")[-1]
        )
        self.file_handlers = self._get_file_handlers()

        # 디버깅: 전달받은 인자 타입 확인
        logger.info(
            f"[CustomDocumentLoader] file 타입: {type(file)}, file_path: {file_path}"
        )

    def load(self) -> list[Document]:
        file_to_process = self.file
        name_for_logging = self.file_path

        if isinstance(file_to_process, str):
            if not os.path.exists(file_to_process):
                raise FileNotFoundError(f"Path does not exist: {file_to_process}")
            file_type = file_to_process.split(".")[-1].lower()
        elif isinstance(file_to_process, io.BytesIO):
            file_type = self.file_name.split(".")[-1].lower()
            name_for_logging = self.file_name
        else:
            raise TypeError("file must be a file path or BytesIO object")

        handler = self.file_handlers.get(file_type)
        handler_name = file_type if handler else "etc"
        if not handler:
            handler = self.file_handlers.get("etc")

        logger.info(f"Using '{handler_name}' loader for {name_for_logging}")
        return handler(self.file, self.file_path, self.file_name)

    def _get_file_handlers(
        self,
    ) -> Dict[str, Callable[[Union[str, io.BytesIO]], list[Document]]]:
        return {
            "pdf": self._load_pdf,
        }

    @staticmethod
    async def _async_ocr_task(page_num, page_image):
        """비동기 OCR 태스크"""
        page_text = await ocr_with_gpt(page_image)
        return page_num, split_texts(page_text)

    @staticmethod
    def clean_html(html: str) -> str:
        """table, tr, td, th 태그는 유지하고 나머지 태그 제거"""
        text = re.sub(r"<(?!/?(table|tr|td|th)\b)[^>]+>", "", html)
        return text.strip()

    @staticmethod
    def processing_table_tags(chunks: list[str]) -> list[str]:
        """
        표 청크에서만 table/tr/td/th 태그를 | 로 변환
        """
        converted_chunks = []
        for chunk in chunks:
            if "<table" in chunk:  # 표 청크만 변환
                chunk = re.sub(r"</?table.*?>", "", chunk)
                chunk = re.sub(r"<tr[^>]*>", "\n| ", chunk)
                chunk = re.sub(r"</tr>", " | \n", chunk)
                chunk = re.sub(r"</?td.*?>", " | ", chunk)
                chunk = re.sub(r"</?th.*?>", " | ", chunk)
                chunk = re.sub(r"<.*?>", "", chunk)
            converted_chunks.append(chunk.strip())
        return converted_chunks

    def split_texts_preserve_table(text: str) -> list[str]:
        """
        순서를 유지하면서 표(<table>)는 하나의 청크로 넣되,
        표 앞 일반 텍스트 마지막 일부(overlap)를 표 청크에 포함
        """
        chunks = []
        pos = 0
        table_overlap = 100  # overlap 길이

        for match in re.finditer(
            r"<table.*?>.*?</table>", text, re.DOTALL | re.IGNORECASE
        ):
            start, end = match.span()

            # last_overlap 초기화
            last_overlap = ""
            
            # 표 앞 일반 텍스트
            if start > pos:
                normal_text = text[pos:start]
                normal_chunks = split_texts(normal_text)
                chunks.extend(normal_chunks)

                # 마지막 청크 일부를 overlap
                last_overlap = (
                    normal_chunks[-1][-table_overlap:] if normal_chunks else ""
                )

            # 표 내용 하나의 청크로
            table_html = match.group()
            table_chunk = (last_overlap + "\n" + table_html).strip()
            chunks.append(table_chunk)

            pos = end

        # 마지막 표 이후 일반 텍스트
        if pos < len(text):
            chunks.extend(split_texts(text[pos:]))

        return chunks

    @staticmethod
    def _load_pdf_upstage_only(
        file: Union[str, io.BytesIO],
        file_path: str,
        file_name: str,
    ) -> list[Document]:
        """Upstage Document Parser만 사용하여 PDF 처리 (LLM OCR 없음)"""
        new_documents = []
        doc_idx = 0

        # 파일 임시 저장
        if isinstance(file, io.BytesIO):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name
            file_bytes = open(tmp_path, "rb").read()
        elif isinstance(file, str):
            tmp_path = file_path
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        else:
            # 파일 객체인 경우 (docx 변환 후 열린 파일)
            tmp_path = file.name if hasattr(file, "name") else None
            file.seek(0)  # 파일 포인터를 처음으로 이동
            file_bytes = file.read()

        try:
            logger.info("Upstage Document Parser만 사용하여 PDF 처리 시작")
            start_time = time.time()

            loader = UpstageDocumentParseLoader(tmp_path, ocr="force", split="page")
            pages = loader.load()
            total_page = len(pages)

            # Upstage 결과를 모두 텍스트로 변환 (LLM OCR 건너뛰기)
            for page_num, page in enumerate(pages, start=1):
                html_content = page.page_content

                # HTML 내용을 텍스트로 변환
                text = CustomDocumentLoader.clean_html(html_content)
                text_chunks = CustomDocumentLoader.split_texts_preserve_table(text)
                text_chunks = CustomDocumentLoader.processing_table_tags(text_chunks)

                # Document 생성
                docs, doc_idx = to_documents(
                    file_name, file_path, text_chunks, doc_idx, page_num, total_page
                )
                new_documents.extend(docs)

            elapsed = time.time() - start_time
            logger.info(
                f"Upstage만으로 PDF 처리 완료 - 소요시간: {elapsed:.2f}초, 페이지: {total_page}개"
            )

        finally:
            if isinstance(file, io.BytesIO) and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return new_documents

    @staticmethod
    def _split_pdf_by_pages(pdf_bytes: bytes) -> list[bytes]:
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
            # 분할 실패 시 원본 PDF를 리스트로 반환
            return [pdf_bytes]

    @staticmethod
    def _load_pdf(
        file: Union[str, io.BytesIO], file_path: str, file_name: str
    ) -> list[Document]:
        """PDF 파일을 페이지별로 분할하여 GPT에 전송하여 텍스트 추출"""
        try:
            # 파일을 bytes로 읽기
            if isinstance(file, io.BytesIO):
                file.seek(0)
                file_bytes = file.read()
            elif isinstance(file, str):
                with open(file_path, "rb") as f:
                    file_bytes = f.read()
            else:
                # 파일 객체인 경우 (docx 변환 후 열린 파일)
                file.seek(0)
                file_bytes = file.read()

            logger.info(f"PDF 파일을 페이지별로 분할하여 GPT OCR 처리 시작: {file_path}")
            start_time = time.time()

            # PDF를 페이지별로 분할
            page_pdfs = CustomDocumentLoader._split_pdf_by_pages(file_bytes)
            total_pages = len(page_pdfs)
            
            all_extracted_texts = []
            all_documents = []
            doc_idx = 0

            # 모든 페이지를 한 번의 이벤트 루프에서 처리
            async def process_all_pages():
                results = []
                for page_num, page_pdf_bytes in enumerate(page_pdfs, 1):
                    logger.info(f"페이지 {page_num}/{total_pages} 처리 중...")
                    try:
                        page_text = await ocr_pdf_with_gpt(page_pdf_bytes)
                        results.append((page_num, page_text, None))
                    except Exception as page_error:
                        results.append((page_num, None, page_error))
                return results

            # 모든 페이지 처리 실행
            page_results = asyncio.run(process_all_pages())
            
            # 결과 처리
            for page_num, page_text, page_error in page_results:
                if page_error:
                    logger.error(f"페이지 {page_num} 처리 중 오류: {str(page_error)}")
                    # 페이지 처리 실패 시 기본 텍스트 추가
                    error_text = f"페이지 {page_num} 처리 실패: {str(page_error)[:100]}"
                    docs, doc_idx = to_documents(
                        file_name, file_path, [error_text], doc_idx, page_num, total_pages
                    )
                    all_documents.extend(docs)
                elif page_text and page_text.strip():
                    all_extracted_texts.append(page_text)
                    
                    # 페이지별 텍스트 청킹
                    text_chunks = split_texts(page_text)
                    
                    # 페이지별 Document 생성
                    docs, doc_idx = to_documents(
                        file_name, file_path, text_chunks, doc_idx, page_num, total_pages
                    )
                    all_documents.extend(docs)
                    
                    logger.info(f"페이지 {page_num} 처리 완료 - 청크 수: {len(text_chunks)}")
                else:
                    logger.warning(f"페이지 {page_num} OCR 결과가 비어있음")
            
            elapsed = time.time() - start_time
            logger.info(f"전체 PDF 처리 완료 - 소요시간: {elapsed:.2f}초, 총 페이지: {total_pages}, 문서 수: {len(all_documents)}")

            if not all_documents:
                logger.warning(f"PDF 전체 처리 실패: {file_path}")
                error_text = f"PDF 파일 처리 실패: {file_path.split('/')[-1]}"
                documents, _ = to_documents(file_name, file_path, [error_text])
                return documents

            return all_documents

        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {file_path}, 오류: {str(e)}")
            # 오류 발생 시 기본 메타데이터만 포함
            error_text = f"PDF 처리 실패: {str(e)[:200]}"
            documents, _ = to_documents(file_name, file_path, [error_text])
            return documents