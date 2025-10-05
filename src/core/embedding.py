import os
import io
import logging
from tqdm import tqdm
from handler.document_loader import CustomDocumentLoader
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import FAISS
from core import config

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# OpenAI API 키

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = UpstageEmbeddings(model="embedding-query")


def append_to_vectorstore(pdf_files: list[str], index_path: str = "faiss_index", batch_size: int = 500):
    """
    기존 FAISS 인덱스에 새로운 문서를 추가하거나
    인덱스가 없으면 새로 생성하여 저장합니다.
    """
    all_documents = []

    logging.info("파일 로딩을 시작합니다...")
    for pdf_path in tqdm(pdf_files, desc="PDF 파일 로딩 중"):
        if not os.path.exists(pdf_path):
            logging.warning(f"파일이 존재하지 않음: {pdf_path}")
            continue
        try:
            with open(pdf_path, "rb") as f:
                file_bytes = io.BytesIO(f.read())
            loader = CustomDocumentLoader(file=file_bytes, file_path=pdf_path, file_name=pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
        except Exception as e:
            logging.error(f"{pdf_path} 파일 처리 중 오류 발생: {e}")

    if not all_documents:
        logging.warning("로딩된 문서가 없습니다.")
        return

    logging.info(f"총 {len(all_documents)}개의 새로운 문서를 임베딩합니다...")

    os.makedirs(index_path, exist_ok=True)
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")
    
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        logging.info("기존 인덱스를 로드하여 문서를 추가합니다.")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        # 기존 인덱스에 추가할 때도 배치 처리
        for i in tqdm(range(0, len(all_documents), batch_size), desc="문서 추가 중"):
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)
    else:
        logging.info("기존 인덱스가 없어 새로 생성합니다.")
        # 첫 번째 배치로 인덱스 생성
        first_batch = all_documents[:batch_size]
        vectorstore = FAISS.from_documents(first_batch, embeddings)
        
        # 나머지 배치들을 순차적으로 추가
        for i in tqdm(range(batch_size, len(all_documents), batch_size), desc="인덱스 생성 중"):
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)

    vectorstore.save_local(index_path)
    logging.info(f"벡터스토어가 '{index_path}'에 저장되었습니다.")


if __name__ == "__main__":
    pdf_files = [
        "파일",
    ]
    append_to_vectorstore(pdf_files, index_path="tests/vectordb", batch_size=500)
