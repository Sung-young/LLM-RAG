"""
Sparse Retrieval 대화형 검색 스크립트

실시간으로 질문을 입력하고 BM25 키워드 검색 결과를 확인할 수 있습니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.sparse_retrieval import SparseRetriever
from src.utils.config import PKL_FILE_PATH, SPARSE_RETRIEVAL_CONFIG


def main():
    """대화형 Sparse Retrieval 실행"""
    
    print("=" * 80)
    print("Sparse Retrieval 대화형 검색 (BM25)")
    print("=" * 80)
    print("\n💡 사용법:")
    print("  - 질문을 입력하면 키워드 기반으로 관련 문서를 검색합니다")
    print("  - 정확한 용어나 키워드가 포함된 질문에 효과적입니다")
    print("  - 'quit', 'exit', '종료', '나가기'를 입력하면 종료됩니다")
    print("=" * 80)
    
    # Sparse Retriever 초기화
    print("\n[초기화 중...]")
    retriever = SparseRetriever(pkl_file_path=str(PKL_FILE_PATH))
    
    # 문서 로드
    print("문서 로딩 중...")
    if not retriever.load_documents():
        print("✗ 문서 로드 실패")
        return
    
    # 토크나이저 초기화
    print("\n토크나이저 초기화 중...")
    if not retriever.initialize_tokenizer():
        print("✗ 토크나이저 초기화 실패")
        return
    
    # BM25 인덱스 구축
    print("\nBM25 인덱스 구축 중...")
    if not retriever.build_index():
        print("✗ 인덱스 구축 실패")
        return
    
    # 대화형 검색 시작
    print("\n" + "=" * 80)
    print("✓ 초기화 완료! 질문을 입력하세요.")
    print("=" * 80)
    
    top_k = 5  # 반환할 상위 결과 개수
    
    while True:
        try:
            # 질문 입력
            query = input("\n💬 질문: ").strip()
            
            # 종료 명령 확인
            if query.lower() in ['quit', 'exit', '종료', '나가기', 'q']:
                print("\n검색을 종료합니다. 👋")
                break
            
            # 빈 입력 무시
            if not query:
                print("질문을 입력해주세요.")
                continue
            
            # 검색 실행
            print(f"\n🔍 검색 중... (상위 {top_k}개 결과)")
            results = retriever.search(query, top_k=top_k)
            
            # 결과 출력
            if not results:
                print("검색 결과가 없습니다.")
                continue
            
            retriever.print_results(query, results)
            
            # 상위 결과 요약
            if results:
                best_score = results[0]['bm25_score']
                # 토큰 추출 정보
                tokens = retriever.tokenize(query)
                print(f"\n💡 추출된 키워드: {', '.join(tokens[:10])}")
                print(f"   최고 BM25 점수: {best_score:.4f}")
                
                if best_score > 15:
                    quality = "🟢 높은 키워드 매칭"
                elif best_score > 8:
                    quality = "🟡 중간 키워드 매칭"
                else:
                    quality = "🟠 낮은 키워드 매칭"
                print(f"   검색 품질: {quality}")
        
        except KeyboardInterrupt:
            print("\n\n검색을 종료합니다. 👋")
            break
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            print("다시 시도해주세요.")


if __name__ == "__main__":
    main()

