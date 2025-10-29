"""
Hybrid Retrieval: Dense + Sparse 결합 검색
RRF (Reciprocal Rank Fusion) 알고리즘 사용
"""

import numpy as np
from typing import List, Dict, Any
from .dense_retrieval import VectorDBTester
from .sparse_retrieval import SparseRetriever


class HybridRetriever:
    """
    Dense Retrieval과 Sparse Retrieval을 결합한 하이브리드 검색기
    RRF (Reciprocal Rank Fusion)를 사용하여 두 결과를 병합
    """
    
    def __init__(
        self,
        dense_retriever: VectorDBTester,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60
    ):
        """
        Args:
            dense_retriever: Dense Retrieval 객체
            sparse_retriever: Sparse Retrieval 객체
            rrf_k: RRF 상수 (기본값: 60)
        """
        self.dense_retriever = dense_retriever
        self.sparse_retriever = sparse_retriever
        self.rrf_k = rrf_k
    
    def search(self, query: str, top_k: int = 10, dense_k: int = 10, sparse_k: int = 10) -> List[Dict[str, Any]]:
        """
        하이브리드 검색 수행
        
        Args:
            query: 검색 질의
            top_k: 최종 반환할 문서 개수
            dense_k: Dense에서 가져올 문서 개수
            sparse_k: Sparse에서 가져올 문서 개수
            
        Returns:
            RRF로 결합된 검색 결과 리스트
        """
        # 1. Dense Retrieval 수행
        dense_results = self.dense_retriever.search_similar_documents(query, top_k=dense_k)
        
        # 2. Sparse Retrieval 수행
        sparse_results = self.sparse_retriever.search(query, top_k=sparse_k)
        
        # 3. RRF로 결합
        combined_results = self._combine_with_rrf(dense_results, sparse_results)
        
        # 4. 상위 k개만 반환
        return combined_results[:top_k]
    
    def _combine_with_rrf(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        RRF (Reciprocal Rank Fusion)로 두 검색 결과 결합
        
        RRF Score = Σ(1 / (k + rank_i))
        
        Args:
            dense_results: Dense Retrieval 결과
            sparse_results: Sparse Retrieval 결과
            
        Returns:
            RRF 점수로 정렬된 결합 결과
        """
        # 문서별 RRF 점수 계산
        rrf_scores = {}
        document_info = {}  # 문서 정보 저장
        
        # Dense 결과 처리
        for rank, result in enumerate(dense_results, start=1):
            doc_index = result['index']
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_index not in rrf_scores:
                rrf_scores[doc_index] = 0.0
                document_info[doc_index] = {
                    'index': doc_index,
                    'source': result.get('source', 'Unknown'),
                    'page': result.get('page', 'N/A'),
                    'dense_rank': rank,
                    'dense_score': result.get('similarity_score', 0.0),
                    'sparse_rank': None,
                    'sparse_score': None
                }
            
            rrf_scores[doc_index] += rrf_score
            document_info[doc_index]['dense_rank'] = rank
            document_info[doc_index]['dense_score'] = result.get('similarity_score', 0.0)
        
        # Sparse 결과 처리
        for rank, result in enumerate(sparse_results, start=1):
            doc_index = result['index']
            rrf_score = 1.0 / (self.rrf_k + rank)
            
            if doc_index not in rrf_scores:
                rrf_scores[doc_index] = 0.0
                document_info[doc_index] = {
                    'index': doc_index,
                    'source': result.get('source', 'Unknown'),
                    'page': result.get('page', 'N/A'),
                    'dense_rank': None,
                    'dense_score': None,
                    'sparse_rank': rank,
                    'sparse_score': result.get('bm25_score', 0.0)
                }
            
            rrf_scores[doc_index] += rrf_score
            document_info[doc_index]['sparse_rank'] = rank
            document_info[doc_index]['sparse_score'] = result.get('bm25_score', 0.0)
        
        # RRF 점수로 정렬
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 최종 결과 생성
        combined_results = []
        for final_rank, (doc_index, rrf_score) in enumerate(sorted_docs, start=1):
            info = document_info[doc_index]
            combined_results.append({
                'rank': final_rank,
                'index': doc_index,
                'rrf_score': rrf_score,
                'source': info['source'],
                'page': info['page'],
                'dense_rank': info['dense_rank'],
                'dense_score': info['dense_score'],
                'sparse_rank': info['sparse_rank'],
                'sparse_score': info['sparse_score']
            })
        
        return combined_results
    
    def print_results(self, query: str, results: List[Dict[str, Any]]):
        """
        검색 결과를 보기 좋게 출력
        
        Args:
            query: 검색 질의
            results: 검색 결과 리스트
        """
        print(f"\n질의: {query}")
        print(f"검색 결과: {len(results)}개")
        print("-" * 100)
        
        for result in results:
            print(f"\n[순위 {result['rank']}] 문서 인덱스: {result['index']}")
            print(f"  - 출처: {result['source']}")
            print(f"  - 페이지: {result['page']}")
            print(f"  - RRF Score: {result['rrf_score']:.6f}")
            
            # Dense/Sparse 개별 순위 및 점수
            dense_info = f"순위 {result['dense_rank']}, 유사도 {result['dense_score']:.4f}" if result['dense_rank'] else "검색 안됨"
            sparse_info = f"순위 {result['sparse_rank']}, BM25 {result['sparse_score']:.4f}" if result['sparse_rank'] else "검색 안됨"
            
            print(f"  - Dense:  {dense_info}")
            print(f"  - Sparse: {sparse_info}")
            
            # 어느 방식이 더 기여했는지 표시
            if result['dense_rank'] and result['sparse_rank']:
                print(f"  - 두 방식 모두 발견 (하이브리드 효과)")
            elif result['dense_rank']:
                print(f"  -  Dense만 발견 (의미 기반)")
            elif result['sparse_rank']:
                print(f"  -  Sparse만 발견 (키워드 기반)")

