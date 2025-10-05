import os
import time
import json
import asyncio
from dotenv import load_dotenv

# LangChain 및 Faiss 관련 라이브러리 임포트
from langchain_openai import ChatOpenAI
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from typing import List

# .env 파일에서 API 키를 로드합니다.
load_dotenv()

# --- 1. 기본 설정 ---
FAISS_INDEX_PATH = "tests/vectordb"
EMBEDDING_MODEL = "embedding-query"
LLM_MODEL = "gpt-4.1-mini"  

# --- 2. Faiss 벡터스토어 로드 및 Retriever 생성 ---
def create_faiss_retriever(index_path: str, embeddings_model_name: str):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Faiss 인덱스 경로를 찾을 수 없습니다: {index_path}")
    
    embeddings = UpstageEmbeddings(model=embeddings_model_name)
    vectorstore = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3}) # 검색 결과는 3개로 설정
    
    print(" Faiss Retriever가 성공적으로 생성되었습니다.")
    return retriever, vectorstore

# --- 3. 주변 청크 검색 ---
def get_retrieved_docs_with_surroundings(query: str, retriever, doc_map: dict) -> List[Document]:
    initial_docs = retriever.invoke(query)
    final_docs = {}
    for doc in initial_docs:
        source = doc.metadata.get("source")
        # 'rows' 메타데이터를 고유 ID로 사용 (없으면 'page' 사용)
        doc_id = doc.metadata.get("rows") or doc.metadata.get("page")
        page_str = str(doc_id)

        if source is None or page_str is None:
            continue

        try:
            page = int(page_str)
        except (ValueError, TypeError):
            continue

        final_docs[(source, page)] = doc
        next_chunk_1 = doc_map.get((source, page + 1))
        if next_chunk_1:
            final_docs[(source, page + 1)] = next_chunk_1
        # next_chunk_2 = doc_map.get((source, page + 2))
        # if next_chunk_2:
        #     final_docs[(source, page + 2)] = next_chunk_2
            
    return sorted(final_docs.values(), key=lambda d: (d.metadata.get("source", ""), d.metadata.get("rows", 0) or d.metadata.get("page", 0)))

def create_document_lookup_map(vectorstore: FAISS) -> dict:
    doc_map = {}
    for doc in vectorstore.docstore._dict.values():
        source = doc.metadata.get("source")
        doc_id = doc.metadata.get("rows") or doc.metadata.get("page")
        if source is not None and doc_id is not None:
            try:
                page = int(doc_id)
                doc_map[(source, page)] = doc
            except (ValueError, TypeError):
                continue
    print(f"총 {len(doc_map)}개의 문서에 대한 조회 맵을 생성했습니다.")
    return doc_map

# --- 4. RAG 체인 구성 ---
PROMPT_TEMPLATE = """
당신은 주어진 컨텍스트에서 사용자의 질문과 가장 유사한 질문을 찾아 해당 질문의 답변을 제공하는 AI 어시스턴트입니다.

# 지침:
1. `[Context]`에는 여러 개의 `질문:`과 `답변:` 쌍이 제공됩니다.
2. `[Question]`을 보고, `[Context]`에 있는 `질문:`들 중에서 의미적으로 가장 유사한 것을 **단 하나만** 찾으세요.
3. 찾은 질문에 해당하는 `답변:`의 내용을 **그대로 출력**하세요.
4. 내용을 요약하거나 변경하지 마세요.
5. 만약 적절한 답변을 찾지 못했다면, "관련된 답변을 찾을 수 없습니다."라고만 출력하세요.

[Context]:
{context}

[Question]:
{question}

[Final Answer]:
"""

def create_rag_chain(llm_model: str):
    llm = ChatOpenAI(model=llm_model)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    output_parser = StrOutputParser()
    
    rag_chain = prompt | llm | output_parser
    return rag_chain

# --- 5. 메인 실행 부분  ---
async def main():
    print("애플리케이션을 초기화합니다. 잠시 기다려주세요...")
    faiss_retriever, faiss_vectorstore = create_faiss_retriever(FAISS_INDEX_PATH, EMBEDDING_MODEL)
    document_map = create_document_lookup_map(faiss_vectorstore)
    rag_chain = create_rag_chain(LLM_MODEL)
    
    print("\n초기화 완료. 이제 질문을 입력할 수 있습니다.")
    
    while True:
        user_query = input("\n\n 질문을 입력하세요 (종료하려면 'exit' 입력): ")
        if user_query.lower() == 'exit':
            print("프로그램을 종료합니다.")
            break
        
        start_time = time.time()
        print(f"\n [ 검색 시작 ] ")
        print(f" 질문: {user_query}")
            
        # 1. 유사도 검색을 통해 관련 문서(청크)를 가져옵니다.
        retrieved_docs = get_retrieved_docs_with_surroundings(user_query, faiss_retriever, document_map)
            
        if not retrieved_docs:
            print("  -> 답변: 관련된 문서를 찾지 못했습니다.")
            continue

        print(f"{len(retrieved_docs)}개의 관련 청크를 찾았습니다.")

        print("\n--- [컨텍스트 상세 내용 ] ---")
        for i, doc in enumerate(retrieved_docs):
            doc_id = doc.metadata.get('rows') or doc.metadata.get('page')
            print(f"  ({i+1}) [Source: {doc.metadata.get('source')}, Row/ID: {doc_id}]")
            print(f"      Q: {doc.page_content[:100]}...")
            print(f"      A: {doc.metadata.get('answer', '')[:100]}...")
        print("-------------------------------------------")

        # 2. 검색된 문서를 "질문: [page_content]\n답변: [metadata의 answer]" 형태로 재구성하여 컨텍스트 생성
        context_text = "\n\n".join(
            [f"질문: {doc.page_content}\n답변: {doc.metadata.get('answer', '')}" for doc in retrieved_docs]
        )
        
        # 3. RAG 체인을 통해 최종 답변 생성
        print("\n [ 답변 생성 시작 ] ")
        final_answer = rag_chain.invoke({
            "context": context_text,
            "question": user_query
        })
        print(" 최종 답변:")
        print(final_answer)

        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n[총 질문 처리 시간: {elapsed:.2f}초]")

if __name__ == "__main__":
    asyncio.run(main())