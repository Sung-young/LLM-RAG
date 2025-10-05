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
│   └── new_document_loader.py # 문서 로더 (Plumber + camelot 사용 )
├── 📁 tests/                # 테스트 파일
│   ├── pdf_test.py          # PDF 텍스트 추출 테스트
├── 📁 utils/                # 공통 유틸리티
│   └── loader_modules.py    
└── main.py                 # retriever를 통한 답변 생성
```
