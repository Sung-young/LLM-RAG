from handler.jy_document_loader import KoreanPdfLoader

# 1) 일반 텍스트 기반 한글 PDF
loader = KoreanPdfLoader(
    file="data/policy/[2025.04.01]송배전용전기설비 이용규정.pdf" ,
    try_tables=True,          # 표도 추출
    ocr=False                 # 기본은 비-OCR
)
docs = loader.load()
print("문서 개수:", len(docs))
print(docs[0].metadata, "\n", docs[0].page_content[:300])

# 2) 스캔 기반(이미지) PDF → OCR 가동
ocr_loader = KoreanPdfLoader(
    file="data/policy/[2025.04.01]송배전용전기설비 이용규정.pdf" ,
    try_tables=False,         # 스캔은 표 추출이 거의 안 되므로 생략 권장
    ocr=True,                 # OCR 사용
    tesseract_lang="kor+eng"  # 한국어+영어 동시 인식
)
ocr_docs = ocr_loader.load()
print("OCR 문서 개수:", len(ocr_docs))

# 3) BytesIO로 읽기
import io
with open("data/policy/[2025.04.01]송배전용전기설비 이용규정.pdf", "rb") as f:
    buf = io.BytesIO(f.read())

mem_loader = KoreanPdfLoader(file=buf, file_name="in-memory.pdf", try_tables=True)
mem_docs = mem_loader.load()
print("메모리 로드:", len(mem_docs))
