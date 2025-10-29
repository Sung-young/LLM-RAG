"""
RAG 벡터 데이터베이스 검색 성능 테스트 스크립트

이 스크립트는 성영님이 전처리한 벡터 DB의 검색(Retrieval) 성능을 평가함

현재 Retrieval 방식:
- 임베딩 모델 : OpenAI text-embedding-3-small (1536차원)
- 벡터 검색 : FAISS (Cosine Distance 기반)
- @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@중요)검색 방식 : 순수 벡터 유사도 검색 (Dense Retrieval)
- 검색 방식은 순수 벡터 유사도 검색 말고, 다른 방식(ex )

"""

import pickle
import faiss
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
from datetime import datetime
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# ===== 설정 =====
VECTORDB_PATH = "/data/한전프로젝트/data/vectorized1/vectordb"
FAISS_INDEX_PATH = f"{VECTORDB_PATH}/index.faiss"
PKL_FILE_PATH = f"{VECTORDB_PATH}/index.pkl"

# 임베딩 모델 설정 (실제 임베딩에 사용한 모델이랑 같게)
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class VectorDBTester:
    """벡터 DB 검색 성능 테스트 클래스"""
    
    def __init__(self, faiss_index_path: str, pkl_file_path: str, model_name: str):
        self.faiss_index_path = faiss_index_path
        self.pkl_file_path = pkl_file_path
        self.model_name = model_name
        self.index = None
        self.metadata = None
        self.embedding_model = None
        
    def load_vector_db(self):
        """벡터 DB 로드"""
        print("=" * 80)
        print("1. 벡터 데이터베이스 로딩")
        print("=" * 80)
        
        # FAISS 인덱스 로드
        try:
            self.index = faiss.read_index(self.faiss_index_path)
            print(f"✓ FAISS 인덱스 로드 성공")
            print(f"  - 총 벡터 개수: {self.index.ntotal}")
            print(f"  - 벡터 차원: {self.index.d}")
        except Exception as e:
            print(f"✗ FAISS 인덱스 로드 실패: {e}")
            return False
        
        # 메타데이터 로드
        try:
            with open(self.pkl_file_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"\n✓ 메타데이터 로드 성공")
            self._print_metadata_info()
            
        except Exception as e:
            print(f"✗ 메타데이터 로드 실패: {e}")
            self.metadata = None
        
        return True
    
    def _print_metadata_info(self):
        """메타데이터 정보 출력"""
        print(f"\n메타데이터 구조:")
        
        if isinstance(self.metadata, tuple) and len(self.metadata) == 2:
            docstore, index_to_docstore_id = self.metadata
            print(f"  - Type: Tuple (LangChain 구조)")
            print(f"  - Docstore: {type(docstore).__name__}")
            print(f"  - Index mapping: {type(index_to_docstore_id).__name__} (길이: {len(index_to_docstore_id)})")
            
            # 첫 번째 문서 샘플 확인
            try:
                first_doc_id = index_to_docstore_id[0]
                first_doc = docstore.search(first_doc_id)
                print(f"\n  첫 번째 문서 샘플:")
                print(f"    - ID: {first_doc_id}")
                if hasattr(first_doc, 'page_content'):
                    content_preview = first_doc.page_content[:200] + "..." if len(first_doc.page_content) > 200 else first_doc.page_content
                    print(f"    - 내용: {content_preview}")
                if hasattr(first_doc, 'metadata'):
                    print(f"    - 메타데이터: {first_doc.metadata}")
            except Exception as e:
                print(f"    - 문서 로드 실패: {e}")
        
        elif isinstance(self.metadata, dict):
            print(f"  - Type: Dictionary")
            print(f"  - Keys: {list(self.metadata.keys())}")
            
            for key, value in self.metadata.items():
                print(f"\n  [{key}]")
                print(f"    - Type: {type(value).__name__}")
                if isinstance(value, (list, tuple)):
                    print(f"    - Length: {len(value)}")
                    if len(value) > 0:
                        print(f"    - First item type: {type(value[0]).__name__}")
                        if isinstance(value[0], dict):
                            print(f"    - First item keys: {list(value[0].keys())}")
                        # 샘플 출력
                        sample = str(value[0])
                        if len(sample) > 200:
                            sample = sample[:200] + "..."
                        print(f"    - Sample: {sample}")
        
        elif isinstance(self.metadata, list):
            print(f"  - Type: List")
            print(f"  - Length: {len(self.metadata)}")
            if len(self.metadata) > 0:
                print(f"  - First item type: {type(self.metadata[0]).__name__}")
                sample = str(self.metadata[0])
                if len(sample) > 200:
                    sample = sample[:200] + "..."
                print(f"  - Sample: {sample}")
        else:
            print(f"  - Type: {type(self.metadata).__name__}")
            sample = str(self.metadata)
            if len(sample) > 500:
                sample = sample[:500] + "..."
            print(f"  - Content: {sample}")
    
    def load_embedding_model(self):
        """임베딩 모델 로드 - OpenAI API"""
        print("\n" + "=" * 80)
        print("2. OpenAI 임베딩 모델 설정")
        print("=" * 80)
        
        try:
            from openai import OpenAI
            
            if not OPENAI_API_KEY:
                print("✗ OpenAI API 키가 설정되지 않았습니다.")
                print("\n다음 중 하나의 방법으로 API 키를 설정하세요:")
                print("1. 환경 변수 설정: export OPENAI_API_KEY='your-api-key'")
                print("2. 스크립트에 직접 입력 (보안에 주의)")
                api_key = input("\nAPI 키를 입력하세요 (또는 Enter를 눌러 건너뛰기): ").strip()
                if api_key:
                    self.embedding_model = OpenAI(api_key=api_key)
                else:
                    return False
            else:
                self.embedding_model = OpenAI(api_key=OPENAI_API_KEY)
            
            print(f"✓ OpenAI 클라이언트 초기화 성공")
            print(f"  - 모델: {self.model_name}")
            
            # text-embedding-3-small의 기본 차원은 1536
            expected_dim = 1536
            print(f"  - 예상 임베딩 차원: {expected_dim}")
            print(f"  - FAISS 인덱스 차원: {self.index.d}")
            
            if expected_dim != self.index.d:
                print(f"\n⚠ 경고: 임베딩 차원 불일치!")
                print(f"  → 벡터 DB 생성 시 차원을 축소했을 수 있습니다.")
            
            return True
            
        except Exception as e:
            print(f"✗ OpenAI 클라이언트 초기화 실패: {e}")
            print("\n다음을 확인하세요:")
            print("1. openai 라이브러리가 설치되어 있는지: pip install openai")
            print("2. API 키가 올바른지")
            print("3. 인터넷 연결이 되어 있는지")
            return False
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        질의에 대해 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 질의
            top_k: 반환할 상위 문서 개수
        
        Returns:
            검색된 문서 리스트 (각 문서는 dict 형태)
        """
        if self.embedding_model is None:
            print("✗ 임베딩 모델이 로드되지 않았습니다.")
            return []
        
        try:
            # 1. 질의 임베딩 (OpenAI API 사용)
            response = self.embedding_model.embeddings.create(
                input=query,
                model=self.model_name
            )
            query_embedding = np.array([response.data[0].embedding]).astype('float32')
            
            # 차원이 다르면 조정 (벡터 DB 생성 시 차원을 축소했을 경우)
            if query_embedding.shape[1] != self.index.d:
                query_embedding = query_embedding[:, :self.index.d]
            
        except Exception as e:
            print(f"✗ 임베딩 생성 실패: {e}")
            return []
        
        # 2. FAISS 검색
        distances, indices = self.index.search(query_embedding, top_k)
        
        # 3. 결과 포매팅
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            result = {
                'rank': i + 1,
                'index': int(idx),
                'distance': float(dist),
                'similarity_score': 1 / (1 + float(dist))  # 거리를 유사도로 변환
            }
            
            # 메타데이터가 있으면 추가
            if self.metadata is not None:
                if isinstance(self.metadata, tuple) and len(self.metadata) == 2:
                    # LangChain 구조: (docstore, index_to_docstore_id)
                    docstore, index_to_docstore_id = self.metadata
                    try:
                        if idx in index_to_docstore_id:
                            doc_id = index_to_docstore_id[idx]
                            doc = docstore.search(doc_id)
                            
                            if hasattr(doc, 'page_content'):
                                result['content'] = doc.page_content
                            if hasattr(doc, 'metadata'):
                                result['source'] = doc.metadata.get('source', 'Unknown')
                                result['page'] = doc.metadata.get('page', 'N/A')
                                result['doc_metadata'] = doc.metadata
                    except Exception as e:
                        result['error'] = f"문서 로드 실패: {e}"
                
                elif isinstance(self.metadata, dict):
                    # 메타데이터가 dict 형태일 경우
                    for key, values in self.metadata.items():
                        if isinstance(values, (list, tuple)) and idx < len(values):
                            result[key] = values[idx]
                elif isinstance(self.metadata, list) and idx < len(self.metadata):
                    # 메타데이터가 list 형태일 경우
                    result['metadata'] = self.metadata[idx]
            
            results.append(result)
        
        return results
    
    def print_search_results(self, query: str, results: List[Dict]):
        """검색 결과를 보기 좋게 출력합니다."""
        print(f"\n{'='*80}")
        print(f"질의: {query}")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"[순위 {result['rank']}] 유사도: {result['similarity_score']:.4f}")
            
            # 문서 출처 정보
            if 'source' in result:
                source_file = result['source'].split('/')[-1] if '/' in result['source'] else result['source']
                print(f"  📄 출처: {source_file}")
                if 'page' in result and result['page'] != 'N/A':
                    print(f"     페이지: {result['page']}")
            
            # 오류가 있으면 표시
            if 'error' in result:
                print(f"  ❌ {result['error']}")
            
            print()
    
    def run_comprehensive_tests(self, test_queries: List[Dict]):
        """검색 성능 테스트 실행"""
        print("\n" + "=" * 80)
        print("3. 검색 성능 테스트")
        print("=" * 80)
        
        all_results = []
        
        for i, test_item in enumerate(test_queries, 1):
            print(f"\n[테스트 {i}/{len(test_queries)}] 카테고리: {test_item.get('category', 'N/A')}")
            results = self.search_similar_documents(test_item['query'], top_k=5)
            self.print_search_results(test_item['query'], results)
            
            all_results.append({
                'query': test_item['query'],
                'category': test_item.get('category', 'N/A'),
                'results': results
            })
            
            print("-" * 80)
        
        return all_results
    
    
    def save_test_results(self, test_queries: List[Dict], output_path: str = "search_test_results.json"):
        """테스트 결과를 JSON 파일로 저장"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_vectors': self.index.ntotal,
            'embedding_dimension': self.index.d,
            'embedding_model': self.model_name,
            'test_queries': []
        }
        
        for test_item in test_queries:
            search_results = self.search_similar_documents(test_item['query'], top_k=5)
            
            results['test_queries'].append({
                'query': test_item['query'],
                'category': test_item.get('category', 'N/A'),
                'top_results': [
                    {
                        'rank': r['rank'],
                        'index': r['index'],
                        'similarity_score': r['similarity_score'],
                        'distance': r['distance']
                    }
                    for r in search_results
                ]
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 테스트 결과 저장 완료: {output_path}")
    
    def interactive_search(self, top_k=5):
        """대화형 검색 인터페이스"""
        print("\n" + "=" * 80)
        print("대화형 검색 모드 (종료하려면 'quit' 또는 'exit' 입력)")
        print("=" * 80 + "\n")
        
        while True:
            query = input("\n질문을 입력하세요: ").strip()
            
            if query.lower() in ['quit', 'exit', '종료', '나가기']:
                print("검색을 종료합니다.")
                break
            
            if not query:
                continue
            
            results = self.search_similar_documents(query, top_k=top_k)
            self.print_search_results(query, results)


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
    
    # 테스터 초기화
    tester = VectorDBTester(
        faiss_index_path=FAISS_INDEX_PATH,
        pkl_file_path=PKL_FILE_PATH,
        model_name=EMBEDDING_MODEL_NAME
    )
    
    # 1. 벡터 DB 로드
    if not tester.load_vector_db():
        print("\n벡터 DB 로드에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 임베딩 모델 로드
    if not tester.load_embedding_model():
        print("\n임베딩 모델 로드에 실패했습니다.")
        print("모델명을 확인하고 EMBEDDING_MODEL_NAME 변수를 수정하세요.")
        return
    
    # 3. 검색 성능 테스트
    tester.run_comprehensive_tests(test_queries)
    
    # 4. 결과 저장
    output_path = "/data/한전프로젝트/search_test_results.json"
    tester.save_test_results(test_queries, output_path)
    
    # 5. 대화형 검색 (선택사항)
    print("\n\n" + "=" * 80)
    print("대화형 검색 모드를 시작하시겠습니까? (y/n)")
    print("=" * 80)
    choice = input("선택: ").strip().lower()
    
    if choice in ['y', 'yes', 'ㅛ', '예']:
        tester.interactive_search(top_k=5)
    
    print("\n\n" + "=" * 80)
    print("테스트 완료!")
    print("=" * 80)
    
    '''
    다음 단계:
    1. 검색 결과를 분석하여 품질을 평가
    2. 필요시 리랭킹, 하이브리드 검색 등 추가 기법 적용
    3. LLM과 연결하여 전체 RAG 파이프라인을 구축
    '''
    print(f"테스트 결과는 다음 파일에 저장되었습니다: {output_path}")
    
if __name__ == "__main__":
    main()


