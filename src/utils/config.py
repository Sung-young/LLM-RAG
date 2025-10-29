"""
설정 관리 모듈
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# .env 파일 로드
load_dotenv(PROJECT_ROOT / '.env')

# 데이터 경로
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "data1"
PROCESSED_DATA_DIR = DATA_DIR / "vectorized1"

# 벡터 DB 경로
VECTORDB_PATH = PROCESSED_DATA_DIR / "vectordb"
FAISS_INDEX_PATH = VECTORDB_PATH / "index.faiss"
PKL_FILE_PATH = VECTORDB_PATH / "index.pkl"

# 테스트 경로
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = TESTS_DIR / "results"

# API 키
API_KEY_FILE = PROJECT_ROOT / "config" / "api_keys.txt"

def get_openai_api_key():
    """OpenAI API 키 가져오기"""
    # 1순위: 환경변수 (.env 파일 포함)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 2순위: config/api_keys.txt 파일에서 읽기 (하위 호환성)
    try:
        if API_KEY_FILE.exists():
            with open(API_KEY_FILE, 'r') as f:
                for line in f:
                    if 'openai' in line.lower():
                        return line.split('=')[1].strip()
    except:
        pass
    
    return None

def get_upstage_api_key():
    """Upstage API 키 가져오기"""
    return os.getenv("UPSTAGE_API_KEY")

# Retrieval 설정
DENSE_RETRIEVAL_CONFIG = {
    'model_name': 'text-embedding-3-small',
    'embedding_dim': 1536,
    'top_k': 5
}

SPARSE_RETRIEVAL_CONFIG = {
    'tokenizer': 'mecab',
    'top_k': 5
}

