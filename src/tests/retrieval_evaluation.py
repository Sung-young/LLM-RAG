"""
Retrieval Quality Evaluation Script
====================================
LLM generation 단계를 제외하고 Retrieval 품질만 테스트하는 스크립트

- BM25 (Sparse Retrieval)
- FAISS (Dense Retrieval)  
- Hybrid Retrieval (BM25 + FAISS)

검색 결과를 관련성 점수와 함께 내림차순으로 표시합니다.
"""

import os
import sys
import json
from typing import List, Dict, Tuple
from pathlib import Path

# 상위 디렉토리를 import path에 추가
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
import numpy as np

# --- 설정 값 (src/main.py와 동일) ---
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "vectordb_1109")
MODEL_NAME = "dragonkue/bge-m3-ko"
FAISS_SEARCH_K = 15
BM25_SEARCH_K = 15
FAISS_FINAL_K = 6
BM25_FINAL_K = 6


class RetrievalEvaluator:
    """Retrieval 품질 평가 클래스"""
    
    def __init__(self, index_path: str, model_name: str):
        """
        Args:
            index_path: FAISS 인덱스 경로
            model_name: Embedding 모델 이름
        """
        print("🔧 Retriever 초기화 중...")
        
        # 1. FAISS Retriever 로드
        print(f"  ✓ FAISS 인덱스 로드 중: {index_path}")
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        self.faiss_retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': FAISS_SEARCH_K}
        )
        print(f"  ✓ FAISS Retriever 생성 완료 (k={FAISS_SEARCH_K})")
        
        # 2. BM25 Retriever 생성
        print(f"  ✓ BM25 Retriever 생성 중...")
        all_docs = list(self.vectorstore.docstore._dict.values())
        new_docs = []
        for doc in all_docs:
            new_docs.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata,
                id=str(doc.metadata.get("id", ""))
            ))
        all_docs = new_docs
        
        self.bm25_retriever = BM25Retriever.from_documents(all_docs)
        self.bm25_retriever.k = BM25_SEARCH_K
        print(f"  ✓ BM25 Retriever 생성 완료 (k={BM25_SEARCH_K})")
        print("✅ Retriever 초기화 완료\n")
    
    def retrieve_bm25(self, query: str) -> List[Tuple[Document, float]]:
        """
        BM25 검색 수행 (관련성 점수 포함)
        
        Returns:
            List of (Document, relevance_score) tuples, sorted by relevance (descending)
        """
        docs = self.bm25_retriever.invoke(query)
        
        # BM25 점수 계산 (retriever 내부 scores 사용)
        results_with_scores = []
        for i, doc in enumerate(docs):
            # BM25는 점수가 높을수록 관련성이 높음
            # retriever.k개만큼 반환되므로 순서대로 점수 부여
            score = len(docs) - i  # 간단한 ranking 점수
            results_with_scores.append((doc, score))
        
        return results_with_scores
    
    def retrieve_faiss(self, query: str) -> List[Tuple[Document, float]]:
        """
        FAISS 검색 수행 (관련성 점수 포함)
        
        Returns:
            List of (Document, relevance_score) tuples, sorted by relevance (descending)
        """
        # FAISS는 similarity_search_with_score를 사용하여 점수 획득
        docs_with_scores = self.vectorstore.similarity_search_with_score(
            query, k=FAISS_SEARCH_K
        )
        
        # FAISS의 L2 거리는 낮을수록 유사도가 높으므로 음수로 변환하여 내림차순 정렬
        # (거리 -> 유사도 점수로 변환)
        results_with_scores = []
        for doc, distance in docs_with_scores:
            similarity_score = -distance  # 거리를 음수로 변환
            results_with_scores.append((doc, similarity_score))
        
        # 점수 기준 내림차순 정렬 (이미 정렬되어 있지만 명시적으로)
        results_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return results_with_scores
    
    def retrieve_hybrid(self, query: str) -> List[Tuple[Document, str, float]]:
        """
        Hybrid 검색 수행 (BM25 + FAISS)
        src/main.py의 get_hybrid_retrieved_docs와 동일한 로직
        
        Returns:
            List of (Document, source_type, rank_score) tuples
            - source_type: "BM25", "FAISS", "BOTH"
            - rank_score: 높을수록 관련성 높음 (순위 기반)
        """
        # 1. BM25 및 FAISS 검색 동시 실행
        bm25_docs = self.bm25_retriever.invoke(query)
        faiss_docs = self.faiss_retriever.invoke(query)
        
        # 2. 각각 상위 K개씩 선택하여 결과 병합
        initial_docs = bm25_docs[:BM25_FINAL_K] + faiss_docs[:FAISS_FINAL_K]
        
        # 3. 중복 제거 (source, page 기준)
        final_docs_map = {}
        doc_sources = {}  # 각 문서가 어디서 왔는지 추적
        
        # BM25 문서 추가
        for i, doc in enumerate(bm25_docs[:BM25_FINAL_K]):
            source = doc.metadata.get("source")
            doc_id = doc.metadata.get("rows") or doc.metadata.get("page")
            page_str = str(doc_id)
            
            if source is None or page_str is None:
                continue
            
            try:
                page = int(page_str)
            except (ValueError, TypeError):
                continue
            
            key = (source, page)
            if key not in final_docs_map:
                final_docs_map[key] = doc
                doc_sources[key] = {"bm25": i, "faiss": None}
            else:
                doc_sources[key]["bm25"] = i
        
        # FAISS 문서 추가
        for i, doc in enumerate(faiss_docs[:FAISS_FINAL_K]):
            source = doc.metadata.get("source")
            doc_id = doc.metadata.get("rows") or doc.metadata.get("page")
            page_str = str(doc_id)
            
            if source is None or page_str is None:
                continue
            
            try:
                page = int(page_str)
            except (ValueError, TypeError):
                continue
            
            key = (source, page)
            if key not in final_docs_map:
                final_docs_map[key] = doc
                doc_sources[key] = {"bm25": None, "faiss": i}
            else:
                doc_sources[key]["faiss"] = i
        
        # 4. 관련성 점수 계산 (순위 기반 점수 매김)
        results = []
        for key, doc in final_docs_map.items():
            bm25_rank = doc_sources[key]["bm25"]
            faiss_rank = doc_sources[key]["faiss"]
            
            # 순위를 점수로 변환 (낮은 순위 = 높은 점수)
            bm25_score = (BM25_FINAL_K - bm25_rank) if bm25_rank is not None else 0
            faiss_score = (FAISS_FINAL_K - faiss_rank) if faiss_rank is not None else 0
            
            # 합산 점수
            combined_score = bm25_score + faiss_score
            
            # 출처 타입 결정
            if bm25_rank is not None and faiss_rank is not None:
                source_type = "BOTH"
            elif bm25_rank is not None:
                source_type = "BM25"
            else:
                source_type = "FAISS"
            
            results.append((doc, source_type, combined_score))
        
        # 5. 점수 기준 내림차순 정렬
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def evaluate_single_query(
        self, 
        query: str, 
        expected_info: Dict = None,
        show_details: bool = True
    ) -> Dict:
        """
        단일 질문에 대한 retrieval 평가
        
        Args:
            query: 검색 질문
            expected_info: 예상 정답 정보 (filename, page 등)
            show_details: 상세 결과 출력 여부
        
        Returns:
            평가 결과 딕셔너리
        """
        print("=" * 100)
        print(f"📝 질문: {query}")
        if expected_info:
            print(f"🎯 정답 문서: {expected_info.get('filename', 'N/A')}, 페이지: {expected_info.get('page', 'N/A')}")
        print("=" * 100)
        
        # 1. BM25 검색
        print("\n🔍 [1] BM25 검색 결과 (Sparse Retrieval)")
        print("-" * 100)
        bm25_results = self.retrieve_bm25(query)
        
        if show_details:
            for idx, (doc, score) in enumerate(bm25_results[:10], 1):
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("rows") or doc.metadata.get("page", "N/A")
                content_preview = doc.page_content[:100].replace("\n", " ")
                
                print(f"  [{idx}] 점수: {score:.2f}")
                print(f"      파일: {source}")
                print(f"      페이지: {page}")
                print(f"      내용: {content_preview}...")
                print()
        
        # 2. FAISS 검색
        print("\n🔍 [2] FAISS 검색 결과 (Dense Retrieval)")
        print("-" * 100)
        faiss_results = self.retrieve_faiss(query)
        
        if show_details:
            for idx, (doc, score) in enumerate(faiss_results[:10], 1):
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("rows") or doc.metadata.get("page", "N/A")
                content_preview = doc.page_content[:100].replace("\n", " ")
                
                print(f"  [{idx}] 점수: {score:.2f}")
                print(f"      파일: {source}")
                print(f"      페이지: {page}")
                print(f"      내용: {content_preview}...")
                print()
        
        # 3. Hybrid 검색
        print("\n🔍 [3] Hybrid 검색 결과 (BM25 + FAISS)")
        print("-" * 100)
        hybrid_results = self.retrieve_hybrid(query)
        
        if show_details:
            for idx, (doc, source_type, score) in enumerate(hybrid_results, 1):
                source = doc.metadata.get("source", "N/A")
                page = doc.metadata.get("rows") or doc.metadata.get("page", "N/A")
                content_preview = doc.page_content[:100].replace("\n", " ")
                
                # 출처 타입에 따라 색상 이모지 추가
                if source_type == "BOTH":
                    type_icon = "🟣"
                elif source_type == "BM25":
                    type_icon = "🔵"
                else:  # FAISS
                    type_icon = "🟢"
                
                print(f"  [{idx}] {type_icon} {source_type} | 점수: {score:.2f}")
                print(f"      파일: {source}")
                print(f"      페이지: {page}")
                print(f"      내용: {content_preview}...")
                print()
        
        # 4. 정답 문서 위치 확인 (expected_info가 있는 경우)
        result_summary = {
            "query": query,
            "bm25_count": len(bm25_results),
            "faiss_count": len(faiss_results),
            "hybrid_count": len(hybrid_results)
        }
        
        if expected_info:
            expected_file = expected_info.get("filename", "")
            expected_page = expected_info.get("page", -1)
            
            # BM25에서 정답 위치 찾기
            bm25_rank = None
            for idx, (doc, _) in enumerate(bm25_results, 1):
                if (doc.metadata.get("source", "").endswith(expected_file) and 
                    (doc.metadata.get("rows") == expected_page or doc.metadata.get("page") == expected_page)):
                    bm25_rank = idx
                    break
            
            # FAISS에서 정답 위치 찾기
            faiss_rank = None
            for idx, (doc, _) in enumerate(faiss_results, 1):
                if (doc.metadata.get("source", "").endswith(expected_file) and 
                    (doc.metadata.get("rows") == expected_page or doc.metadata.get("page") == expected_page)):
                    faiss_rank = idx
                    break
            
            # Hybrid에서 정답 위치 찾기
            hybrid_rank = None
            for idx, (doc, _, _) in enumerate(hybrid_results, 1):
                if (doc.metadata.get("source", "").endswith(expected_file) and 
                    (doc.metadata.get("rows") == expected_page or doc.metadata.get("page") == expected_page)):
                    hybrid_rank = idx
                    break
            
            result_summary["expected_info"] = expected_info
            result_summary["bm25_rank"] = bm25_rank
            result_summary["faiss_rank"] = faiss_rank
            result_summary["hybrid_rank"] = hybrid_rank
            
            print("\n📊 정답 문서 검색 순위:")
            print(f"  🔵 BM25:   {bm25_rank if bm25_rank else '❌ 검색 안됨'}")
            print(f"  🟢 FAISS:  {faiss_rank if faiss_rank else '❌ 검색 안됨'}")
            print(f"  🟣 Hybrid: {hybrid_rank if hybrid_rank else '❌ 검색 안됨'}")
        
        print("\n" + "=" * 100 + "\n")
        
        return result_summary
    
    def evaluate_from_json(self, json_path: str, num_samples: int = None):
        """
        JSON 파일에서 질문을 읽어와 평가
        
        Args:
            json_path: QA set JSON 파일 경로
            num_samples: 평가할 샘플 수 (None이면 전체)
        """
        print(f"📂 JSON 파일 로드 중: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # JSON 구조 파악 (리스트의 리스트인 경우)
        if isinstance(qa_data, list) and len(qa_data) > 0 and isinstance(qa_data[0], list):
            qa_list = qa_data[0]
        else:
            qa_list = qa_data
        
        if num_samples:
            qa_list = qa_list[:num_samples]
        
        print(f"✅ {len(qa_list)}개의 질문을 평가합니다.\n")
        
        all_results = []
        
        for i, qa_item in enumerate(qa_list, 1):
            print(f"\n{'#' * 100}")
            print(f"# 테스트 {i}/{len(qa_list)}")
            print(f"# 타입: {qa_item.get('type', 'N/A')}")
            print(f"{'#' * 100}\n")
            
            question = qa_item.get("question", "")
            expected_info = {
                "filename": qa_item.get("filename", ""),
                "page": qa_item.get("page", -1),
                "answer": qa_item.get("answer", ""),
                "type": qa_item.get("type", "")
            }
            
            result = self.evaluate_single_query(
                query=question,
                expected_info=expected_info,
                show_details=True
            )
            
            all_results.append(result)
        
        # 전체 통계 출력
        self._print_overall_statistics(all_results)
        
        return all_results
    
    def _print_overall_statistics(self, results: List[Dict]):
        """전체 평가 통계 출력"""
        print("\n" + "🎯" * 50)
        print("전체 평가 통계")
        print("🎯" * 50 + "\n")
        
        # 정답 문서를 찾은 경우만 필터링
        results_with_expected = [r for r in results if "expected_info" in r]
        
        if not results_with_expected:
            print("⚠️  정답 정보가 있는 질문이 없습니다.")
            return
        
        total = len(results_with_expected)
        
        # Top-1, Top-3, Top-5, Top-10 정확도
        bm25_top1 = sum(1 for r in results_with_expected if r.get("bm25_rank") == 1)
        bm25_top3 = sum(1 for r in results_with_expected if r.get("bm25_rank") and r["bm25_rank"] <= 3)
        bm25_top5 = sum(1 for r in results_with_expected if r.get("bm25_rank") and r["bm25_rank"] <= 5)
        bm25_top10 = sum(1 for r in results_with_expected if r.get("bm25_rank") and r["bm25_rank"] <= 10)
        
        faiss_top1 = sum(1 for r in results_with_expected if r.get("faiss_rank") == 1)
        faiss_top3 = sum(1 for r in results_with_expected if r.get("faiss_rank") and r["faiss_rank"] <= 3)
        faiss_top5 = sum(1 for r in results_with_expected if r.get("faiss_rank") and r["faiss_rank"] <= 5)
        faiss_top10 = sum(1 for r in results_with_expected if r.get("faiss_rank") and r["faiss_rank"] <= 10)
        
        hybrid_top1 = sum(1 for r in results_with_expected if r.get("hybrid_rank") == 1)
        hybrid_top3 = sum(1 for r in results_with_expected if r.get("hybrid_rank") and r["hybrid_rank"] <= 3)
        hybrid_top5 = sum(1 for r in results_with_expected if r.get("hybrid_rank") and r["hybrid_rank"] <= 5)
        hybrid_top10 = sum(1 for r in results_with_expected if r.get("hybrid_rank") and r["hybrid_rank"] <= 10)
        
        print(f"📊 총 평가 질문 수: {total}개\n")
        
        print("🔵 BM25 (Sparse) 정확도:")
        print(f"  - Top-1:  {bm25_top1}/{total} ({bm25_top1/total*100:.1f}%)")
        print(f"  - Top-3:  {bm25_top3}/{total} ({bm25_top3/total*100:.1f}%)")
        print(f"  - Top-5:  {bm25_top5}/{total} ({bm25_top5/total*100:.1f}%)")
        print(f"  - Top-10: {bm25_top10}/{total} ({bm25_top10/total*100:.1f}%)")
        
        print("\n🟢 FAISS (Dense) 정확도:")
        print(f"  - Top-1:  {faiss_top1}/{total} ({faiss_top1/total*100:.1f}%)")
        print(f"  - Top-3:  {faiss_top3}/{total} ({faiss_top3/total*100:.1f}%)")
        print(f"  - Top-5:  {faiss_top5}/{total} ({faiss_top5/total*100:.1f}%)")
        print(f"  - Top-10: {faiss_top10}/{total} ({faiss_top10/total*100:.1f}%)")
        
        print("\n🟣 Hybrid (BM25 + FAISS) 정확도:")
        print(f"  - Top-1:  {hybrid_top1}/{total} ({hybrid_top1/total*100:.1f}%)")
        print(f"  - Top-3:  {hybrid_top3}/{total} ({hybrid_top3/total*100:.1f}%)")
        print(f"  - Top-5:  {hybrid_top5}/{total} ({hybrid_top5/total*100:.1f}%)")
        print(f"  - Top-10: {hybrid_top10}/{total} ({hybrid_top10/total*100:.1f}%)")
        
        # 질문 타입별 분석
        print("\n📋 질문 타입별 분석:")
        types = {}
        for r in results_with_expected:
            q_type = r["expected_info"].get("type", "unknown")
            if q_type not in types:
                types[q_type] = {"total": 0, "hybrid_found": 0}
            types[q_type]["total"] += 1
            if r.get("hybrid_rank"):
                types[q_type]["hybrid_found"] += 1
        
        for q_type, stats in types.items():
            success_rate = stats["hybrid_found"] / stats["total"] * 100
            print(f"  - {q_type}: {stats['hybrid_found']}/{stats['total']} ({success_rate:.1f}%)")
        
        print("\n" + "🎯" * 50 + "\n")


def main():
    """메인 실행 함수"""
    print("\n" + "🚀" * 50)
    print("Retrieval Quality Evaluation Script")
    print("🚀" * 50 + "\n")
    
    # Retrieval Evaluator 초기화
    print("⏳ Retriever를 초기화하는 중입니다. 잠시만 기다려주세요...\n")
    evaluator = RetrievalEvaluator(
        index_path=FAISS_INDEX_PATH,
        model_name=MODEL_NAME
    )
    
    print("\n" + "=" * 100)
    print("✅ 초기화 완료! 이제 대화형 모드로 질문할 수 있습니다.")
    print("=" * 100)
    print("\n💡 사용 방법:")
    print("  - 질문을 입력하면 BM25, FAISS, Hybrid 검색 결과를 확인할 수 있습니다")
    print("  - 'exit' 또는 'quit'를 입력하면 종료합니다")
    print("  - 'json'을 입력하면 QA 세트 평가 모드로 전환합니다")
    print("  - 'help'를 입력하면 이 도움말을 다시 볼 수 있습니다")
    print("\n" + "-" * 100 + "\n")
    
    # 대화형 루프
    while True:
        try:
            user_input = input("📝 질문을 입력하세요 (종료: 'exit', 도움말: 'help', JSON 평가: 'json'): ").strip()
            
            if not user_input:
                continue
            
            # 종료 명령
            if user_input.lower() in ['exit', 'quit', '종료', 'q']:
                print("\n👋 프로그램을 종료합니다. 감사합니다!\n")
                break
            
            # 도움말
            if user_input.lower() in ['help', '도움말', 'h']:
                print("\n" + "-" * 100)
                print("💡 사용 방법:")
                print("  - 질문을 입력하면 BM25, FAISS, Hybrid 검색 결과를 확인할 수 있습니다")
                print("  - 'exit' 또는 'quit'를 입력하면 종료합니다")
                print("  - 'json'을 입력하면 QA 세트 평가 모드로 전환합니다")
                print("  - 'help'를 입력하면 이 도움말을 다시 볼 수 있습니다")
                print("-" * 100 + "\n")
                continue
            
            # JSON 평가 모드
            if user_input.lower() == 'json':
                qa_json_path = os.path.join(BASE_DIR, "QA-set", "sample.json")
                
                if not os.path.exists(qa_json_path):
                    print(f"\n⚠️  QA 파일을 찾을 수 없습니다: {qa_json_path}\n")
                    continue
                
                # 평가할 샘플 수 선택
                try:
                    num_input = input("  평가할 샘플 수를 입력하세요 (전체: Enter 또는 0, 특정 수: 숫자 입력): ").strip()
                    num_samples = None if not num_input or num_input == '0' else int(num_input)
                except ValueError:
                    print("  ⚠️  잘못된 입력입니다. 전체 샘플을 평가합니다.")
                    num_samples = None
                
                print(f"\n📁 QA 세트를 사용하여 평가를 시작합니다...\n")
                evaluator.evaluate_from_json(qa_json_path, num_samples=num_samples)
                print("\n✅ 평가 완료! 계속 질문하실 수 있습니다.\n")
                continue
            
            # 질문 검색 실행
            print()  # 빈 줄 추가
            evaluator.evaluate_single_query(
                query=user_input,
                expected_info=None,  # 대화형 모드에서는 정답 정보 없음
                show_details=True
            )
            
        except KeyboardInterrupt:
            print("\n\n👋 프로그램을 종료합니다. 감사합니다!\n")
            break
        except Exception as e:
            print(f"\n❌ 오류가 발생했습니다: {e}\n")
            import traceback
            traceback.print_exc()
            print()


if __name__ == "__main__":
    main()

