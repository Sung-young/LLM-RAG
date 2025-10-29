"""
Dense Retrieval 실행 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.dense_retrieval import VectorDBTester
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
            'query': '설비 인계 전 설비건설부서가 집계해야 하는 불량 내역', 
            'category': '기준'
        },
        {
            'query': '장비를 넘기기 전에 건설 부서에서 해야 할 작업',
            'category': '기준'
        },
        {
            'query': '설비건설부서가 통계 및 집계할 불량 내역의 세부 조건과 원인 분석의 범위',
            'category': '기준'
        },
        {
            'query': '미공개 중요정보의 보관',
            'category': '사규'
        },
        {
            'query': '가전배전전공 실기평가 중 부정행위 간주 사항은?',
            'category': '절차서'
        },
        {
            'query': '퇴직자의 재취업 제한',
            'category': '지침'
        }
    ]
    
    print("=" * 80)
    print("Dense Retrieval (밀집 검색) 테스트")
    print("=" * 80)
    
    # API 키 가져오기
    api_key = get_openai_api_key()
    if not api_key:
        print("✗ OpenAI API 키를 찾을 수 없습니다.")
        print("환경변수 OPENAI_API_KEY를 설정하거나 config/api_keys.txt에 저장하세요.")
        return
    
    # Dense Retriever 초기화
    retriever = VectorDBTester(
        faiss_index_path=str(FAISS_INDEX_PATH),
        pkl_file_path=str(PKL_FILE_PATH),
        model_name=DENSE_RETRIEVAL_CONFIG['model_name']
    )
    
    # 1. 벡터 DB 로드
    if not retriever.load_vector_db():
        print("✗ 벡터 DB 로드 실패")
        return
    
    # 2. 임베딩 모델 로드
    if not retriever.load_embedding_model():
        print("✗ 임베딩 모델 로드 실패")
        return
    
    # 4. 검색 테스트
    print("\n[4단계] 검색 성능 테스트")
    print("=" * 80)
    
    all_results = []
    for i, test_item in enumerate(test_queries, 1):
        print(f"\n[테스트 {i}/{len(test_queries)}] 카테고리: {test_item['category']}")
        results = retriever.search_similar_documents(test_item['query'], top_k=5)
        retriever.print_search_results(test_item['query'], results)
        
        all_results.append({
            'query': test_item['query'],
            'category': test_item['category'],
            'results': [{
                'rank': r['rank'],
                'index': r['index'],
                'similarity_score': r['similarity_score'],
                'distance': r['distance'],
                'source': r.get('source', 'Unknown'),
                'page': r.get('page', 'N/A')
            } for r in results]
        })
        
        print("-" * 80)
    
    # 5. 결과 저장
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "dense_retrieval_results.json"
    
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'method': 'Dense Retrieval (FAISS + OpenAI Embeddings)',
        'model': DENSE_RETRIEVAL_CONFIG['model_name'],
        'total_queries': len(test_queries),
        'test_results': all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)
    print(f"결과 저장 위치: {output_path}")
    

if __name__ == "__main__":
    main()

