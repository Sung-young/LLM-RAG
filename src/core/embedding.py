import os
import io
import logging
import torch
from datetime import datetime
from tqdm import tqdm
from src.handler.document_loader import CustomDocumentLoader
from src.handler.hybrid_llm_document_loader import HybridLLMPdfLoader
from src.handler.new_document_loader import PdfLoader
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
torch.mps.empty_cache()

# Embedding 모델 설정 
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
model_name = "dragonkue/bge-m3-ko"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"}   
)


def append_to_vectorstore(input_path: str, index_path: str = "faiss_index", batch_size: int = 500):
    """폴더 내 파일을 벡터DB에 추가하되, 이미 임베딩된 파일은 건너뜀."""
    all_documents = []

    # 폴더/파일 탐색
    file_list = []
    skipped_count = 0
    if os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                # macOS 메타데이터 파일 필터링 (._로 시작하는 파일)
                if file.startswith("._") or file.startswith(".DS_Store"):
                    skipped_count += 1
                    continue
                # 숨김 파일 필터링 (점으로 시작하는 파일)
                if file.startswith("."):
                    skipped_count += 1
                    continue
                if file.lower().endswith((".pdf", ".xlsx", ".xls", ".txt", ".csv", ".docx")):
                    file_list.append(os.path.join(root, file))
    elif os.path.isfile(input_path):
        # 단일 파일인 경우에도 메타데이터 파일 체크
        basename = os.path.basename(input_path)
        if not (basename.startswith("._") or basename.startswith(".DS_Store") or basename.startswith(".")):
            file_list.append(input_path)
        else:
            logging.error(f"메타데이터 파일은 처리할 수 없습니다: {input_path}")
            return
    else:
        logging.error(f"유효하지 않은 경로입니다: {input_path}")
        return
    
    if skipped_count > 0:
        logging.info(f"메타데이터/숨김 파일 {skipped_count}개를 제외했습니다.")

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

    logging.info(f"새로 임베딩할 파일 {len(new_files)}개를 처리합니다.")

    # 새 파일 임베딩
    success_count = 0
    error_count = 0
    error_files = []
    
    for path in tqdm(new_files, desc="새 문서 로딩 중"):
        try:
            with open(path, "rb") as f:
                file_bytes = io.BytesIO(f.read())
            # loader = CustomDocumentLoader(file_path=path,file=file_bytes, file_name=path)
            loader = HybridLLMPdfLoader(file_path=path,file=file_bytes, file_name=path)
            docs = loader.load()
            if docs:  # 문서가 실제로 생성된 경우만 성공으로 카운트
                all_documents.extend(docs)
                success_count += 1
            else:
                # 빈 문서 리스트는 손상된 PDF로 간주
                error_count += 1
                error_files.append(path)
                logging.warning(f"⚠️ 파일에서 문서를 추출하지 못했습니다 (손상된 PDF 가능): {os.path.basename(path)}")
        except Exception as e:
            error_count += 1
            error_files.append(path)
            logging.error(f"❌ {os.path.basename(path)} 처리 중 오류: {str(e)[:100]}")

    # 처리 결과 요약 및 오류 파일 저장
    logging.info(f"✅ 처리 완료 - 성공: {success_count}개, 실패/건너뜀: {error_count}개")
    if error_files:
        logging.warning(f"⚠️ 처리되지 않은 파일 {len(error_files)}개 (손상된 PDF 등)")
        if len(error_files) <= 10:  # 10개 이하면 모두 출력
            for err_file in error_files:
                logging.warning(f"   - {os.path.basename(err_file)}")
        else:  # 10개 초과면 처음 10개만 출력
            for err_file in error_files[:10]:
                logging.warning(f"   - {os.path.basename(err_file)}")
            logging.warning(f"   ... 외 {len(error_files) - 10}개")
        
        # 처리되지 않은 파일 목록을 텍스트 파일에 저장
        error_log_path = os.path.join(index_path, "failed_files.txt")
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(error_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"처리 일시: {timestamp}\n")
                f.write(f"처리되지 않은 파일 수: {len(error_files)}개\n")
                f.write(f"{'='*80}\n")
                for err_file in error_files:
                    # 전체 경로와 파일명 모두 저장
                    f.write(f"{err_file}\n")
            logging.info(f"📝 처리되지 않은 파일 목록이 '{error_log_path}'에 저장되었습니다.")
        except Exception as e:
            logging.error(f"오류 파일 목록 저장 실패: {e}")

    if not all_documents:
        logging.warning("새 문서가 없습니다.")
        return

    logging.info(f"총 {len(all_documents)}개의 새 문서를 임베딩 중...")

    # 벡터스토어에 추가
    if vectorstore:
        for i in tqdm(range(0, len(all_documents), batch_size), desc="문서 추가 중"):
            torch.mps.empty_cache()
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)
    else:
        first_batch = all_documents[:batch_size]
        vectorstore = FAISS.from_documents(first_batch, embeddings)
        for i in tqdm(range(batch_size, len(all_documents), batch_size), desc="인덱스 생성 중"):
            torch.mps.empty_cache()
            batch = all_documents[i:i + batch_size]
            vectorstore.add_documents(batch)

    vectorstore.save_local(index_path)
    logging.info(f"벡터스토어가 '{index_path}'에 저장되었습니다.")


if __name__ == "__main__":
    input_folder = "failed_files/b"  
    append_to_vectorstore(input_folder, index_path="vectordb-failed-files-b", batch_size=16)
