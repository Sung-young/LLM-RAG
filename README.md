# LLM-RAG
### 핵심 플로우
```
사용자 질문 → RAG 검색 → 답변 생성 →  응답
```
```
src/
├── 📁 core/                 # 핵심 비즈니스 로직
│   └── embedding.py        # 벡터 DB 생성
├── 📁 handlers/             # 이벤트/요청 처리
│   ├── document_loader.py  # 문서 로더 (Upstage Document Parse, GPT OCR 사용)
│   └── new_document_loader.py # 문서 로더 (Plumber + camelot 사용)
├── 📁 tests/                # 테스트 파일
│   ├── pdf_test.py          # PDF 텍스트 추출 테스트
├── 📁 utils/                # 공통 유틸리티
│   └── loader_modules.py    
└── main.py                 # retriever를 통한 답변 생성
```

## 운영 가이드

### vectordb 확인
- vectordb를 디렉토리 최상의 루트에 경로 지정

### 시작 방법(로컬 진행)
```bash
# 환경 설정
pipenv shell
pipenv install

# 실행
python -m src.main
```

### 시작 방법(FastAPI)
```bash
# 환경 설정
pipenv shell
pipenv install

# 실행
python main.py

# 배포
nohup uvicorn main:app --host 0.0.0.0 --port 5000 > uvicorn.log 2>&1 &
```

### 모니터링 포인트
- ✅ RAG 리소스 초기화: "RAG 리소스 로드 완료" 메시지 확인
- ✅ RAG Retriever 진행 시 문서 및 답변 확인

