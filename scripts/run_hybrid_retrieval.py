"""
Hybrid Retrieval 실행 스크립트
Dense + Sparse 결과를 RRF로 결합
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense_retrieval import VectorDBTester
from src.retrieval.sparse_retrieval import SparseRetriever
from src.retrieval.hybrid_retrieval import HybridRetriever
from src.utils.config import (
    FAISS_INDEX_PATH, 
    PKL_FILE_PATH,
    get_openai_api_key,
    DENSE_RETRIEVAL_CONFIG,
    RESULTS_DIR
)
import json
from datetime import datetime


def main():
    """메인 실행 함수"""
    
    # 테스트 질의 세트
    test_queries = [
        {
            'query': '설비 인계전 설비건설부서가 집계해야 하는 불량 내역', 
            'category': '기준',
            'type': '키워드 기반'
        },
        {
            'query': '설비를 인계하기 전에 설비건설부서에서 해야 할 작업',
            'category': '기준',
            'type': '의미 기반'
        },
        {
            'query': '제작불량 및 제작사의 시공불량',
            'category': '기준',
            'type': '키워드 기반'
        },
        {
            'query': '미공개 중요정보의 보관',
            'category': '사규',
            'type': '키워드 기반'
        },
        {
            'query': '공개되지 않은 중요한 정보의 보관',
            'category': '사규',
            'type': '의미 기반'
        },
        {
            'query': '가전배전전공 실기평가 중 부정행위 간주 사항',
            'category': '절차서',
            'type': '키워드 기반'
        },
        {
            'query': '퇴직자의 재취업 제한',
            'category': '지침',
            'type': '키워드 기반'
        },
        {
            'query': '퇴직한 사람이 다시 취업하는 것',
            'category': '지침',
            'type': '키워드의 말바꿈 기반'
        },
        

    ]
    
    print("=" * 100)
    print("Hybrid Retrieval (하이브리드 검색) 테스트")
    print("Dense (의미 기반) + Sparse (키워드 기반) → RRF 결합")
    print("=" * 100)
    
    # API 키 가져오기
    api_key = get_openai_api_key()
    if not api_key:
        print("✗ OpenAI API 키를 찾을 수 없습니다.")
        print("환경변수 OPENAI_API_KEY를 설정하거나 config/api_keys.txt에 저장하세요.")
        return
    
    # 1. Dense Retriever 초기화 및 로드
    print("\n[1단계] Dense Retriever 초기화")
    print("-" * 100)
    dense_retriever = VectorDBTester(
        faiss_index_path=str(FAISS_INDEX_PATH),
        pkl_file_path=str(PKL_FILE_PATH),
        model_name=DENSE_RETRIEVAL_CONFIG['model_name']
    )
    
    if not dense_retriever.load_vector_db():
        print("✗ 벡터 DB 로드 실패")
        return
    
    if not dense_retriever.load_embedding_model():
        print("✗ 임베딩 모델 로드 실패")
        return
    
    # 2. Sparse Retriever 초기화 및 로드
    print("\n[2단계] Sparse Retriever 초기화")
    print("-" * 100)
    sparse_retriever = SparseRetriever(pkl_file_path=str(PKL_FILE_PATH))
    
    if not sparse_retriever.load_documents():
        print("✗ 문서 로드 실패")
        return
    
    if not sparse_retriever.initialize_tokenizer():
        print("⚠️  Mecab 초기화 실패, 간단한 토크나이저 사용")
    
    if not sparse_retriever.build_index():
        print("✗ BM25 인덱스 구축 실패")
        return
    
    # 3. Hybrid Retriever 초기화
    print("\n[3단계] Hybrid Retriever 초기화")
    print("-" * 100)
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_k=60
    )
    print("✓ 하이브리드 검색기 준비 완료 (RRF k=60)")
    
    # 4. 검색 테스트
    print("\n[4단계] 검색 성능 테스트")
    print("=" * 100)
    
    all_results = []
    for i, test_item in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}/{len(test_queries)}]")
        print(f"  카테고리: {test_item['category']}")
        print(f"  질의 타입: {test_item['type']}")
        
        # Hybrid 검색 수행 (Dense top 10 + Sparse top 10 → 최종 top 10)
        results = hybrid_retriever.search(
            query=test_item['query'],
            top_k=10,
            dense_k=10,
            sparse_k=10
        )
        
        hybrid_retriever.print_results(test_item['query'], results)
        
        all_results.append({
            'query': test_item['query'],
            'category': test_item['category'],
            'query_type': test_item['type'],
            'results': [{
                'rank': r['rank'],
                'index': r['index'],
                'rrf_score': r['rrf_score'],
                'source': r['source'],
                'page': r['page'],
                'dense_rank': r['dense_rank'],
                'dense_score': r['dense_score'],
                'sparse_rank': r['sparse_rank'],
                'sparse_score': r['sparse_score']
            } for r in results]
        })
        
        print("-" * 100)
    
    # 5. 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "hybrid_retrieval_results.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': 'Hybrid Retrieval (Dense + Sparse with RRF)',
        'rrf_k': 60,
        'dense_top_k': 10,
        'sparse_top_k': 10,
        'final_top_k': 10,
        'total_queries': len(test_queries),
        'test_results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 100)
    print("테스트 완료!")
    print("=" * 100)
    print(f"결과 저장 위치: {output_path}")
    

if __name__ == "__main__":
    main()

