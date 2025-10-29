# Hybrid Retrieval 구현 가이드

## 📌 개요

**문제 상황:**
- 사용자가 "키워드 기반" 질의인지 "의미 기반" 질의인지 구분할 수 없음
- Dense Retrieval만 사용하면 정확한 키워드 매칭이 약함
- Sparse Retrieval만 사용하면 문맥 이해가 불가능

**해결책:**
- **Hybrid Retrieval** = Dense + Sparse 결과를 RRF로 결합
- 사용자가 질의 방식을 신경 쓸 필요 없음
- 두 방식의 장점을 모두 활용

---

## 🔧 구현된 내용

### 1. 핵심 모듈
- **`src/retrieval/hybrid_retrieval.py`**: Hybrid Retrieval 구현
  - `HybridRetriever` 클래스
  - RRF (Reciprocal Rank Fusion) 알고리즘
  - Dense + Sparse 결과 병합 및 중복 제거

### 2. 실행 스크립트
- **`scripts/run_hybrid_retrieval.py`**: 배치 테스트 실행
- **`scripts/hybrid_retrieval_chatting.py`**: 대화형 검색

### 3. 설정
- Dense top 10 + Sparse top 10 검색
- RRF 상수 k = 60
- 최종 top 10 문서 반환

---

## 🧮 RRF (Reciprocal Rank Fusion) 알고리즘

### 공식
```
RRF_score(document) = Σ (1 / (k + rank_i(document)))

여기서:
- k = 60 (상수)
- rank_i = 각 retriever에서 문서의 순위 (1위부터 시작)
```

### 예시 계산

**시나리오:**
- 문서 A: Dense 1위, Sparse 3위
- 문서 B: Dense 5위, Sparse 1위
- 문서 C: Dense에만 2위로 검색됨

**계산:**
```
문서 A RRF = 1/(60+1) + 1/(60+3) = 0.0164 + 0.0159 = 0.0323
문서 B RRF = 1/(60+5) + 1/(60+1) = 0.0154 + 0.0164 = 0.0318
문서 C RRF = 1/(60+2) + 0           = 0.0161 + 0      = 0.0161
```

**최종 순위:** A > B > C

**해석:**
- 두 방식 모두에서 발견된 문서가 최고 점수
- 한쪽에서만 발견되어도 순위에 포함
- 순위가 높을수록 영향력이 큼

---

## 🚀 사용 방법

### 배치 테스트
```bash
cd /data/한전프로젝트
source .venv/bin/activate
python scripts/run_hybrid_retrieval.py
```

**출력:**
- 6개 테스트 질의에 대한 검색 결과
- 각 문서가 Dense/Sparse 중 어디서 발견되었는지 표시
- 결과는 `tests/results/hybrid_retrieval_results.json`에 저장

### 대화형 검색
```bash
python scripts/hybrid_retrieval_chatting.py
```

**특징:**
- 실시간 질의응답
- 각 검색 결과에 대해 "하이브리드 효과" 분석 제공
- quit/exit로 종료

---

## 📊 출력 형식

```
[순위 1] 문서 인덱스: 17
  📄 출처: 송변전 기자재공급자 품질평가 기준(3차)_전문.pdf
  📖 페이지: 18
  🔢 RRF Score: 0.030415
  🔵 Dense:  순위 10, 유사도 0.4808
  🟠 Sparse: 순위 2, BM25 14.1673
  ✅ 두 방식 모두 발견 (하이브리드 효과)
```

**설명:**
- **RRF Score**: 최종 결합 점수 (높을수록 관련성 높음)
- **Dense**: 의미 기반 검색 순위 및 유사도 점수
- **Sparse**: 키워드 기반 검색 순위 및 BM25 점수
- **표시**: 어느 방식에서 발견되었는지 명시

---

## 💡 하이브리드 효과 분석

검색 완료 후 자동으로 표시:

```
검색 품질 요약:
  ✅ 두 방식 모두 발견: 5개
  🔵 Dense만 발견: 3개
  🟠 Sparse만 발견: 2개
  💡 하이브리드 효과가 좋습니다! (절반 이상이 두 방식 모두에서 발견)
```

**해석:**
- "두 방식 모두 발견" 비율이 높을수록 하이브리드 효과가 좋음
- 질의 타입에 관계없이 안정적인 검색 가능

---

## 🎯 장점

### 1. 사용자 관점
- ✅ 질의 방식(키워드/의미)을 구분할 필요 없음
- ✅ 어떤 질문을 해도 적절한 결과를 받을 수 있음
- ✅ 두 방식의 장점을 모두 활용

### 2. 검색 품질
- ✅ 키워드 기반 질의 → Sparse가 높은 순위, 자동 반영
- ✅ 의미 기반 질의 → Dense가 높은 순위, 자동 반영
- ✅ 두 방식 모두에서 발견된 문서 → 최고 점수

### 3. 확장성
- ✅ RRF 상수(k) 조정 가능
- ✅ Dense/Sparse의 검색 범위(top_k) 조정 가능
- ✅ 향후 Re-ranking 추가 가능

---

## 🔧 파라미터 조정

### `src/retrieval/hybrid_retrieval.py`

```python
# RRF 상수 조정 (기본: 60)
hybrid_retriever = HybridRetriever(
    dense_retriever=dense_retriever,
    sparse_retriever=sparse_retriever,
    rrf_k=60  # 값이 클수록 순위 차이의 영향이 줄어듦
)

# 검색 범위 조정
results = hybrid_retriever.search(
    query=query,
    top_k=10,      # 최종 반환 개수
    dense_k=10,    # Dense에서 가져올 개수
    sparse_k=10    # Sparse에서 가져올 개수
)
```

**조정 가이드:**
- **rrf_k 증가**: 순위 차이가 덜 중요해짐 (더 많은 문서에 기회)
- **rrf_k 감소**: 고순위 문서의 영향력 증가
- **dense_k/sparse_k 증가**: 더 넓은 범위에서 탐색 (느려짐)
- **dense_k/sparse_k 감소**: 빠른 검색, 정확도는 약간 떨어질 수 있음

---

## 📈 다음 단계 (선택사항)

### 1. Re-ranking 추가
하이브리드 결과를 더 정교하게 재정렬:
- Upstage Solar Embedding 사용
- Cohere Rerank API 사용
- Cross-encoder 모델 사용

### 2. 가중치 기반 결합
RRF 대신 가중치 조합:
```
Hybrid_score = α * Dense_score + (1-α) * Sparse_score
```

### 3. 질의 분석
질의를 분석하여 Dense/Sparse 가중치 동적 조정

---

## 🔍 테스트 결과 확인

```bash
# JSON 결과 파일 확인
cat tests/results/hybrid_retrieval_results.json

# 주요 내용:
# - timestamp: 실행 시각
# - method: "Hybrid Retrieval (Dense + Sparse with RRF)"
# - rrf_k: 60
# - test_results: 각 질의별 결과
#   - dense_rank: Dense에서 순위
#   - sparse_rank: Sparse에서 순위
#   - rrf_score: 최종 결합 점수
```

---

## ✅ 검증 체크리스트

- [x] Dense Retrieval 정상 작동
- [x] Sparse Retrieval 정상 작동
- [x] Hybrid Retrieval 구현 완료
- [x] RRF 알고리즘 적용
- [x] 배치 테스트 스크립트 작동
- [x] 대화형 검색 스크립트 작동
- [x] 결과 JSON 저장 확인
- [x] README.md 업데이트

---

## 📞 문의

하이브리드 검색 결과를 확인한 후, 필요에 따라:
1. Re-ranking 추가 검토
2. RRF 파라미터 튜닝
3. LLM 연결 (RAG 파이프라인 완성)

현재 구현으로 **기본적인 하이브리드 검색 시스템은 완성**되었습니다! 🎉

