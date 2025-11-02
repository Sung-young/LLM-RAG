import uvicorn
import time
from fastapi import FastAPI
from pydantic import BaseModel
from src.main import get_rag_response

app = FastAPI(
    title="LLM RAG API",
    description="한전 챗봇 API",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str


@app.post("/query", response_model=QueryResponse, summary="RAG 질의응답")
async def handle_query(request: QueryRequest):
    """
    사용자 쿼리를 입력받아 RAG 체인을 통해 생성된 답변을 반환합니다.
    """
    print(f"\n [API 요청 수신] 질문: {request.query}")
    start_time = time.time()
    
    final_answer, retrieved_docs = await get_rag_response(request.query)
    
    if retrieved_docs:
        sources = list(set([doc.metadata.get("source", "N/A") for doc in retrieved_docs]))
        print(f"  -> (응답에 미포함) 참고한 소스: {', '.join(sources)}")
    else:
        print(f"  -> 참고한 소스를 찾지 못했습니다.")


    end_time = time.time()
    elapsed = end_time - start_time
    print(f" [API 응답 완료] 처리 시간: {elapsed:.2f}초")

    # answer반환
    return QueryResponse(
        answer=final_answer
    )

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"status": "RAG API is running"}

# Uvicorn으로 서버 실행 
if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. (http://127.0.0.1:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)