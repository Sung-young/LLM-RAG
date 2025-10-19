# korean_pdf_loader.py
import os
import io
import tempfile
from typing import List, Union, Optional

import pdfplumber
import pandas as pd

try:
    import camelot  # 표 추출(벡터 기반 PDF에서만)
    _HAS_CAMELOT = True
except Exception:
    _HAS_CAMELOT = False

try:
    from pdf2image import convert_from_path
    import pytesseract
    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


def _chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> List[str]:
    """아주 단순한 청킹: 문단 기준 우선, 부족하면 글자 수 기준."""
    if not text:
        return []
    # 문단 우선
    paras = [p.strip() for p in text.replace('\r', '').split('\n\n') if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p) if buf else p
        else:
            if buf:
                chunks.append(buf)
            # p 자체가 너무 길면 강제 분할
            start = 0
            while start < len(p):
                end = min(start + chunk_size, len(p))
                chunks.append(p[start:end])
                start = max(end - chunk_overlap, end)
            buf = ""
    if buf:
        chunks.append(buf)
    return [c for c in chunks if c.strip()]


class KoreanPdfLoader(BaseLoader):
    """
    한글 PDF 로더:
    1) 텍스트 PDF: pdfplumber로 텍스트/단어 추출
    2) 표: (가능하면) camelot(stream)으로 표 DataFrame -> 문자열 변환
    3) 스캔 PDF: ocr=True 이면 pdf2image + pytesseract로 OCR ('kor+eng')
    반환: LangChain Document 리스트
    """

    def __init__(
        self,
        file: Union[str, io.BytesIO],
        file_name: Optional[str] = None,
        *,
        try_tables: bool = True,
        ocr: bool = False,
        tesseract_lang: str = "kor+eng",
        table_as: str = "markdown",  # "markdown" | "tsv" | "plain"
        max_table_rows: int = 2000
    ):
        self.file = file
        self.file_name = file_name or (file if isinstance(file, str) else "in-memory.pdf")
        self.try_tables = try_tables and _HAS_CAMELOT
        self.ocr = ocr and _HAS_OCR
        self.tesseract_lang = tesseract_lang
        self.table_as = table_as
        self.max_table_rows = max_table_rows

    def load(self) -> List[Document]:
        tmp_path = None
        # BytesIO -> 임시 파일
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
            raise TypeError("file must be a path (str) or io.BytesIO")

        try:
            # 1) 텍스트 PDF 경로 우선
            docs = self._load_text_first(file_path)

            # 2) 텍스트가 거의 없고 OCR 허용이면 OCR로 보강
            if self.ocr and sum(len(d.page_content.strip()) for d in docs) < 40:
                ocr_docs = self._load_with_ocr(file_path)
                # OCR 결과가 있다면 교체/병합
                if ocr_docs:
                    docs = ocr_docs

            return docs
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    # -------- 내부 구현 --------

    def _load_text_first(self, file_path: str) -> List[Document]:
        out: List[Document] = []
        doc_idx = 0

        # (선택) 표 먼저 추출해 페이지별로 보관
        tables_by_page = {}
        total_pages = 0
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)

        if self.try_tables:
            try:
                camelot_tables = camelot.read_pdf(file_path, pages=f"1-{total_pages}", flavor="stream")
                for t in camelot_tables:
                    p = int(getattr(t, "page", 0))
                    tables_by_page.setdefault(p, []).append(t)
            except Exception:
                # 표 추출 실패 시 무시
                tables_by_page = {}

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text(x_tolerance=1.5, y_tolerance=1.5) or ""
                chunks = _chunk_text(page_text)

                # 표를 문자열로 변환해 추가
                if self.try_tables and i in tables_by_page:
                    for t in tables_by_page[i]:
                        df: pd.DataFrame = t.df.copy()
                        df = df.fillna("").map(lambda x: " ".join(str(x).split()))
                        if len(df) > self.max_table_rows:
                            df = df.iloc[: self.max_table_rows, :]
                        # if self.table_as == "markdown":
                        #     tbl_str = df.to_markdown(index=False, header=True)
                        # elif self.table_as == "tsv":
                        #     tbl_str = df.to_csv(sep="\t", index=False)
                        else:
                            tbl_str = df.to_string(index=False)
                        if tbl_str.strip():
                            chunks.append(tbl_str)

                for ch in chunks:
                    meta = {
                        "file_name": self.file_name,
                        "page_num": i,
                        "total_pages": total_pages,
                        "extract_mode": "text+tables" if self.try_tables else "text-only",
                        "source": file_path,
                    }
                    out.append(Document(page_content=ch, metadata=meta))
                    doc_idx += 1

        return out

    def _load_with_ocr(self, file_path: str) -> List[Document]:
        """스캔 PDF 대비: 페이지를 이미지로 변환 후 Tesseract OCR('kor+eng')."""
        if not _HAS_OCR:
            return []
        images = convert_from_path(file_path, dpi=300)  # 선명도↑
        docs: List[Document] = []
        total_pages = len(images)

        for i, img in enumerate(images, 1):
            text = pytesseract.image_to_string(img, lang=self.tesseract_lang)
            chunks = _chunk_text(text)
            for ch in chunks:
                meta = {
                    "file_name": self.file_name,
                    "page_num": i,
                    "total_pages": total_pages,
                    "extract_mode": "ocr",
                    "source": file_path,
                }
                docs.append(Document(page_content=ch, metadata=meta))
        return docs
