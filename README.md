# 한전 RAG 프로젝트

기업 내부 문서 기반 Hybrid Retrieval 시스템

## 빠른 시작

```bash
# 패키지 설치
pip install -r requirements.txt

# Hybrid Retrieval 실행 (권장)
python scripts/run_hybrid_retrieval.py

# 대화형 검색
python scripts/hybrid_retrieval_chatting.py
```

## Retrieval 방식

| 방식 | 알고리즘 | 강점 |
|------|----------|------|
| **Dense** | FAISS + OpenAI Embeddings | 의미/문맥 이해 |
| **Sparse** | BM25 + Mecab | 정확한 키워드 매칭 |
| **Hybrid** (선택) | RRF (Dense + Sparse) | 균형잡힌 검색 |

## 프로젝트 구조

```
├── src/retrieval/          # 검색 모듈
│   ├── dense_retrieval.py
│   ├── sparse_retrieval.py
│   └── hybrid_retrieval.py
├── scripts/                # 실행 스크립트
└── data/vectorized*/       # 벡터 DB
```

## 기술 스택

- **Python 3.12**
- FAISS, OpenAI Embeddings
- BM25, KoNLPy (Mecab)
- LangChain
