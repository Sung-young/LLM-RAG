"""
Dense Retrieval 대화형 검색 스크립트

실시간으로 질문을 입력하고 벡터 검색 결과를 확인할 수 있습니다.
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
    DENSE_RETRIEVAL_CONFIG
)


def main():
    """대화형 Dense Retrieval 실행"""
    
    print("=" * 80)
    print("Dense Retrieval 대화형 검색")
    print("=" * 80)
    print("\n사용법:")
    print("  - 질문을 입력하면 관련 문서를 검색합니다")
    print("  - 'quit', 'exit', '종료', '나가기'를 입력하면 종료됩니다")
    print("=" * 80)
    
    # API 키 가져오기
    api_key = get_openai_api_key()
    if not api_key:
        print("\n✗ OpenAI API 키를 찾을 수 없습니다.")
        print("\n다음 중 하나의 방법으로 API 키를 설정하세요:")
        print("1. .env 파일에 OPENAI_API_KEY 추가")
        print("2. 환경변수 설정: export OPENAI_API_KEY='your-api-key'")
        print("3. config/api_keys.txt 파일에 저장")
        return
    
    # Dense Retriever 초기화
    print("\n[초기화 중...]")
    retriever = VectorDBTester(
        faiss_index_path=str(FAISS_INDEX_PATH),
        pkl_file_path=str(PKL_FILE_PATH),
        model_name=DENSE_RETRIEVAL_CONFIG['model_name']
    )
    
    # 벡터 DB 로드
    print("벡터 DB 로딩 중...")
    if not retriever.load_vector_db():
        print("✗ 벡터 DB 로드 실패")
        return
    
    # 임베딩 모델 로드
    print("\nOpenAI 임베딩 모델 초기화 중...")
    if not retriever.load_embedding_model():
        print("✗ 임베딩 모델 초기화 실패")
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
                print("\n검색을 종료합니다.")
                break
            
            # 빈 입력 무시
            if not query:
                print("질문을 입력해주세요.")
                continue
            
            # 검색 실행
            print(f"\n🔍 검색 중... (상위 {top_k}개 결과)")
            results = retriever.search_similar_documents(query, top_k=top_k)
            
            # 결과 출력
            if not results:
                print("검색 결과가 없습니다.")
                continue
            
            retriever.print_search_results(query, results)
            
            # 상위 결과 요약
            if results:
                best_score = results[0]['similarity_score']
                if best_score > 0.7:
                    quality = " 0.7 이상 관련성"
                elif best_score > 0.5:
                    quality = " 0.5 이상"
                else:
                    quality = "0.5 미만 관련성"
                print(f"\n검색 품질: {quality} (최고 유사도: {best_score:.4f})")
        
        except KeyboardInterrupt:
            print("\n\n검색을 종료합니다.")
            break
        except Exception as e:
            print(f"\n✗ 오류 발생: {e}")
            print("다시 시도해주세요.")


if __name__ == "__main__":
    main()

