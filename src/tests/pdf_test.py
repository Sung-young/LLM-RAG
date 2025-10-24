import os
import io
from src.handler.document_loader import CustomDocumentLoader
from src.handler.new_document_loader import PdfLoader

# 테스트할 xlsx 경로
pdf_path = "company_policy/2column.pdf"  # 테스트용 xlsx 경로

# 파일 존재 확인
if not os.path.exists(pdf_path):
    print(f" 파일이 존재하지 않습니다: {pdf_path}")
    exit(1)

with open(pdf_path, "rb") as f:
    file_bytes = io.BytesIO(f.read())

# 로더 호출

# upstage 사용 시 
# loader = CustomDocumentLoader(file_path=pdf_path, file=file_bytes, file_name=pdf_path)
# loader.file_handlers["pdf"] = loader._load_pdf_upstage_only. 

# llm (gpt) 사용 시 이미지 용 
# loader = CustomDocumentLoader(file_path=pdf_path, file=file_bytes, file_name=pdf_path)
# documents = loader.load()

# 일반 로더 사용 시 
loader = PdfLoader(file_path=pdf_path, file=file_bytes, file_name=pdf_path)
documents = loader.load()

# 결과 출력
print(f" 추출된 Document 개수: {len(documents)}개\n")


matched_docs = [
    doc for doc in documents if doc.metadata.get("page") 
]

if not matched_docs:
    print(f" 페이지에 해당하는 Document가 없습니다.")
else:
    print(f" 페이지 Document 출력 (총 {len(matched_docs)}개 청크):\n")
    for i, doc in enumerate(matched_docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content.strip())  
        print(f"Metadata: {doc.metadata}\n")















