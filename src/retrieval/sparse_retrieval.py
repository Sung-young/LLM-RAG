"""
Sparse Retrieval (희소 검색) 모듈

BM25 + 한국어 형태소 분석기를 사용한 키워드 기반 검색
"""

import pickle
import numpy as np
from typing import List, Dict
from rank_bm25 import BM25Okapi
import re
from dotenv import load_dotenv

# 한국어 형태소 분석기 import (우선순위: Kiwi > Okt > 없음)
TOKENIZER_AVAILABLE = None

try:
    from kiwipiepy import Kiwi
    TOKENIZER_AVAILABLE = 'kiwi'
except ImportError:
    try:
        from konlpy.tag import Okt
        TOKENIZER_AVAILABLE = 'okt'
    except ImportError:
        pass

# .env 파일 로드
load_dotenv()


class SparseRetriever:
    """BM25 기반 Sparse Retrieval 클래스"""
    
    def __init__(self, pkl_file_path: str):
        """
        Args:
            pkl_file_path: 메타데이터 파일 경로
        """
        self.pkl_file_path = pkl_file_path
        self.metadata = None
        self.documents = []
        self.bm25 = None
        self.tokenizer = None
        
    def load_documents(self):
        """문서 로드"""
        try:
            with open(self.pkl_file_path, 'rb') as f:
                self.metadata = pickle.load(f)
            
            print(f"✓ 메타데이터 로드 성공")
            
            # LangChain 구조에서 문서 추출
            if isinstance(self.metadata, tuple) and len(self.metadata) == 2:
                docstore, index_to_docstore_id = self.metadata
                
                print(f"  - 문서 개수: {len(index_to_docstore_id)}")
                
                # 모든 문서 추출
                for idx in sorted(index_to_docstore_id.keys()):
                    doc_id = index_to_docstore_id[idx]
                    doc = docstore.search(doc_id)
                    
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        self.documents.append({
                            'index': idx,
                            'content': doc.page_content,
                            'source': doc.metadata.get('source', 'Unknown'),
                            'page': doc.metadata.get('page', 'N/A'),
                            'metadata': doc.metadata
                        })
                
                print(f"✓ 문서 추출 완료: {len(self.documents)}개")
                return True
            else:
                print(f"✗ 지원하지 않는 메타데이터 구조입니다.")
                return False
                
        except Exception as e:
            print(f"✗ 문서 로드 실패: {e}")
            return False
    
    def initialize_tokenizer(self):
        """한국어 토크나이저 초기화"""
        # Kiwi 우선 시도
        if TOKENIZER_AVAILABLE == 'kiwi':
            try:
                self.tokenizer = Kiwi()
                self.tokenizer_type = 'kiwi'
                print("✓ Kiwi 토크나이저 로드 성공")
                
                # 테스트
                test_text = "한국전력공사 송변전 기자재 품질평가"
                tokens = self.tokenize(test_text)
                print(f"  - 테스트: '{test_text}' → {tokens[:10]}...")
                
                return True
            except Exception as e:
                print(f"✗ Kiwi 초기화 실패: {e}")
        
        # Okt 시도
        if TOKENIZER_AVAILABLE == 'okt':
            try:
                self.tokenizer = Okt()
                self.tokenizer_type = 'okt'
                print("✓ Okt 토크나이저 로드 성공")
                
                # 테스트
                test_text = "한국전력공사 송변전 기자재 품질평가"
                tokens = self.tokenize(test_text)
                print(f"  - 테스트: '{test_text}' → {tokens[:10]}...")
                
                return True
            except Exception as e:
                print(f"✗ Okt 초기화 실패: {e}")
        
        # 형태소 분석기가 없거나 실패한 경우
        print("✓ 간단한 토크나이저로 대체합니다.")
        self.tokenizer = None
        self.tokenizer_type = 'simple'
        
        # 테스트
        test_text = "한국전력공사 송변전 기자재 품질평가"
        tokens = self.tokenize(test_text)
        print(f"  - 테스트: '{test_text}' → {tokens[:10]}...")
        
        return True
    
    def tokenize(self, text: str) -> List[str]:
        """텍스트 토큰화"""
        # 정규화
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if self.tokenizer is None:
            # 간단한 토크나이저: 공백 기준 분리 + 한글 단어만 추출
            tokens = []
            for word in text.split():
                if len(word) > 1 and re.search(r'[가-힣]', word):
                    tokens.append(word)
            return tokens
        
        # Kiwi 형태소 분석 (명사, 동사, 형용사)
        if self.tokenizer_type == 'kiwi':
            try:
                tokens = []
                result = self.tokenizer.tokenize(text)
                for token in result:
                    # Kiwi 품사 태그: NNG/NNP(명사), VV(동사), VA(형용사), MAG(부사)
                    if token.tag in ['NNG', 'NNP', 'VV', 'VA', 'MAG']:
                        if len(token.form) > 1:
                            tokens.append(token.form)
                return tokens
            except:
                # 실패 시 간단한 토크나이저로 대체
                tokens = []
                for word in text.split():
                    if len(word) > 1 and re.search(r'[가-힣]', word):
                        tokens.append(word)
                return tokens
        
        # Okt 형태소 분석 (명사, 동사, 형용사)
        if self.tokenizer_type == 'okt':
            try:
                tokens = []
                # Okt는 nouns(), verbs() 등의 메서드도 제공하지만, pos()로 통일
                pos_tags = self.tokenizer.pos(text, stem=True)  # stem=True: 어간 추출
                for word, pos in pos_tags:
                    # Okt 품사 태그: Noun(명사), Verb(동사), Adjective(형용사)
                    if pos in ['Noun', 'Verb', 'Adjective']:
                        if len(word) > 1:
                            tokens.append(word)
                return tokens
            except:
                # 실패 시 간단한 토크나이저로 대체
                tokens = []
                for word in text.split():
                    if len(word) > 1 and re.search(r'[가-힣]', word):
                        tokens.append(word)
                return tokens
        
        return []
    
    def build_index(self):
        """BM25 인덱스 구축"""
        if not self.documents:
            print("✗ 문서가 로드되지 않았습니다.")
            return False
        
        try:
            print("문서 토큰화 중...")
            tokenized_corpus = []
            for doc in self.documents:
                tokens = self.tokenize(doc['content'])
                tokenized_corpus.append(tokens)
            
            # BM25 인덱스 생성
            self.bm25 = BM25Okapi(tokenized_corpus)
            
            print(f"✓ BM25 인덱스 구축 완료")
            print(f"  - 총 문서 수: {len(tokenized_corpus)}")
            print(f"  - 평균 문서 길이: {self.bm25.avgdl:.1f} 토큰")
            
            return True
        except Exception as e:
            print(f"✗ BM25 인덱스 구축 실패: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        질의에 대한 유사 문서 검색
        
        Args:
            query: 검색 질의
            top_k: 반환할 상위 문서 개수
        
        Returns:
            검색 결과 리스트
        """
        if self.bm25 is None:
            print("✗ BM25 인덱스가 구축되지 않았습니다.")
            return []
        
        try:
            # 1. 질의 토큰화
            query_tokens = self.tokenize(query)
            
            if not query_tokens:
                print("✗ 유효한 토큰을 추출할 수 없습니다.")
                return []
            
            # 2. BM25 스코어 계산
            scores = self.bm25.get_scores(query_tokens)
            
            # 3. 상위 K개 추출
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            # 4. 결과 포매팅
            results = []
            for rank, idx in enumerate(top_indices, 1):
                score = scores[idx]
                doc = self.documents[idx]
                
                result = {
                    'rank': rank,
                    'index': doc['index'],
                    'bm25_score': float(score),
                    'source': doc['source'],
                    'page': doc['page'],
                    'content': doc['content'],
                    'doc_metadata': doc['metadata']
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            print(f"✗ 검색 실패: {e}")
            return []
    
    def print_results(self, query: str, results: List[Dict]):
        """검색 결과 출력"""
        print(f"\n{'='*80}")
        print(f"질의: {query}")
        print(f"{'='*80}\n")
        
        for result in results:
            print(f"[순위 {result['rank']}] BM25 점수: {result['bm25_score']:.4f}")
            
            source_file = result['source'].split('/')[-1] if '/' in result['source'] else result['source']
            print(f"  📄 출처: {source_file}")
            if result['page'] != 'N/A':
                print(f"     페이지: {result['page']}")
            
            print()
