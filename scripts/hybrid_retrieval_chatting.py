"""
Hybrid Retrieval 대화형 검색 스크립트
Dense + Sparse 결과를 RRF로 결합하여 검색
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
    DENSE_RETRIEVAL_CONFIG
)


def main():
    """대화형 하이브리드 검색"""
    
    print("=" * 100)
    print("Hybrid Retrieval 대화형 검색")
    print("Dense (의미 기반) + Sparse (키워드 기반) → RRF 결합")
    print("=" * 100)
    print("\n사용법:")
    print("  - 질문을 입력하면 관련 문서를 검색합니다")
    print("  - 'quit', 'exit', '종료', '나가기'를 입력하면 종료됩니다")
    print("-" * 100)
    
    # API 키 확인
    api_key = get_openai_api_key()
    if not api_key:
        print("\n✗ OpenAI API 키를 찾을 수 없습니다.")
        print("환경변수 OPENAI_API_KEY를 설정하거나 config/api_keys.txt에 저장하세요.")
        return
    
    # Dense Retriever 초기화
    print("\n[초기화] Dense Retriever 로딩 중...")
    dense_retriever = VectorDBTester(
        faiss_index_path=str(FAISS_INDEX_PATH),
        pkl_file_path=str(PKL_FILE_PATH),
        model_name=DENSE_RETRIEVAL_CONFIG['model_name']
    )
    
    if not dense_retriever.load_vector_db() or not dense_retriever.load_embedding_model():
        print("✗ Dense Retriever 초기화 실패")
        return
    print("✓ Dense Retriever 준비 완료")
    
    # Sparse Retriever 초기화
    print("\n[초기화] Sparse Retriever 로딩 중...")
    sparse_retriever = SparseRetriever(pkl_file_path=str(PKL_FILE_PATH))
    
    if not sparse_retriever.load_documents():
        print("✗ 문서 로드 실패")
        return
    
    if not sparse_retriever.initialize_tokenizer():
        print("⚠️  Mecab 초기화 실패, 간단한 토크나이저 사용")
    
    if not sparse_retriever.build_index():
        print("✗ BM25 인덱스 구축 실패")
        return
    print("✓ Sparse Retriever 준비 완료")
    
    # Hybrid Retriever 초기화
    print("\n[초기화] Hybrid Retriever 준비 중...")
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        rrf_k=60
    )
    print("✓ 하이브리드 검색기 준비 완료 (RRF k=60)")
    
    print("\n" + "=" * 100)
    print("검색을 시작합니다! 질문을 입력하세요.")
    print("=" * 100)
    
    # 대화형 검색 루프
    while True:
        try:
            # 사용자 입력
            query = input("\n질문: ").strip()
            
            # 종료 명령 확인
            if query.lower() in ['quit', 'exit', '종료', '나가기', 'q']:
                print("\n검색을 종료합니다.")
                break
            
            # 빈 입력 무시
            if not query:
                continue
            
            # 하이브리드 검색 수행
            results = hybrid_retriever.search(
                query=query,
                top_k=10,
                dense_k=10,
                sparse_k=10
            )
            
            # 결과 출력
            hybrid_retriever.print_results(query, results)
            
            # 검색 품질 요약
            if results:
                both_count = sum(1 for r in results if r['dense_rank'] and r['sparse_rank'])
                dense_only = sum(1 for r in results if r['dense_rank'] and not r['sparse_rank'])
                sparse_only = sum(1 for r in results if not r['dense_rank'] and r['sparse_rank'])
                
                print(f"\n검색 품질 요약:")
                print(f"  ✅ 두 방식 모두 발견: {both_count}개")
                print(f"  🔵 Dense만 발견: {dense_only}개")
                print(f"  🟠 Sparse만 발견: {sparse_only}개")
                
                if both_count >= len(results) * 0.5:
                    print(f"  💡 하이브리드 효과가 좋습니다! (절반 이상이 두 방식 모두에서 발견)")
        
        except KeyboardInterrupt:
            print("\n\n검색을 종료합니다.")
            break
        except Exception as e:
            print(f"\n검색 중 오류 발생: {str(e)}")
            continue


if __name__ == "__main__":
    main()

