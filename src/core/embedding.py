import os
import io
import logging
from tqdm import tqdm
from src.handler.document_loader import CustomDocumentLoader
from src.handler.new_document_loader import PdfLoader
from langchain_openai import OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Embedding 모델 설정 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = UpstageEmbeddings(model="embedding-query")


def append_to_vectorstore(input_path: str, index_path: str = "faiss_index", batch_size: int = 500):
    """폴더 내 파일을 벡터DB에 추가하되, 이미 임베딩된 파일은 건너뜀."""
    all_documents = []

    # 폴더/파일 탐색
    file_list = []
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.lower().endswith((".pdf", ".xlsx", ".xls", ".txt", ".csv", ".docx")):
                    file_list.append(os.path.join(root, file))
    elif os.path.isfile(input_path):
        file_list.append(input_path)
    else:
        logging.error(f"유효하지 않은 경로입니다: {input_path}")
        return

    if not file_list:
        logging.warning("처리할 파일이 없습니다.")
        return

    logging.info(f"총 {len(file_list)}개의 파일을 감지했습니다.")

    # 기존 인덱스 로드 
    existing_sources = set()
    os.makedirs(index_path, exist_ok=True)
    faiss_file = os.path.join(index_path, "index.faiss")
    pkl_file = os.path.join(index_path, "index.pkl")

    vectorstore = None
    if os.path.exists(faiss_file) and os.path.exists(pkl_file):
        logging.info("기존 인덱스를 로드합니다...")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        # 기존 문서들의 'source' 경로 추출
        existing_sources = {doc.metadata.get("source") for doc in vectorstore.docstore._dict.values()}
        logging.info(f"이미 임베딩된 파일 {len(existing_sources)}개를 확인했습니다.")

    # 새로 추가할 파일만 필터링
    new_files = [f for f in file_list if f not in existing_sources]
    if not new_files:
        logging.info("새로 추가할 파일이 없습니다. 모든 파일이 이미 임베딩되어 있습니다.")
        return

    logging.info(f"새로 임베딩할 파일 {len(new_files)}개: {new_files}")

    # 새 파일 임베딩
    for path in tqdm(new_files, desc="새 문서 로딩 중"):
        try:
            with open(path, "rb") as f:
                file_bytes = io.BytesIO(f.read())
            # loader = CustomDocumentLoader(file_path=path,file=file_bytes, file_name=path)
            loader = PdfLoader(file_path=path,file=file_bytes, file_name=path)
            docs = loader.load()
            all_documents.extend(docs)
        except Exception as e:
            logging.error(f"{path} 처리 중 오류 발생: {e}")

    if not all_documents:
        logging.warning("새 문서가 없습니다.")
        return

    logging.info(f"총 {len(all_documents)}개의 새 문서를 임베딩 중...")

    # 벡터스토어에 추가
    if vectorstore:
        for i in tqdm(range(0, len(all_documents), batch_size), desc="문서 추가 중"):
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)
    else:
        first_batch = all_documents[:batch_size]
        vectorstore = FAISS.from_documents(first_batch, embeddings)
        for i in tqdm(range(batch_size, len(all_documents), batch_size), desc="인덱스 생성 중"):
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)

    vectorstore.save_local(index_path)
    logging.info(f"벡터스토어가 '{index_path}'에 저장되었습니다.")


if __name__ == "__main__":
    input_folder = "data/지침/한전인의 윤리헌장(20191205) 제6차.pdf"
    append_to_vectorstore(input_folder, index_path="vectordb", batch_size=500)