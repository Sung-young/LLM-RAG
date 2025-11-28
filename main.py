import uvicorn
import time
import os
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from src.main import get_rag_response
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LLM RAG API",
    description="한전 챗봇 API",
    version="1.0.0"
)

# 기존 쿼리 모델
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# OpenAI API 호환 모델
class Message(BaseModel):
    role: str = Field(default="user", description="메시지 역할은 user 만 지원합니다.", examples=["user"])
    content: str = Field(..., description="메시지 내용")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="kepco-rag-model", description="모델 이름")
    messages: List[Message] = Field(..., description="대화 메시지 리스트")
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2, description="응답 창의성")
    max_tokens: Optional[int] = Field(default=None, description="최대 생성 토큰 수")
    stream: Optional[bool] = Field(default=False, description="스트리밍 여부")

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

class ErrorDetail(BaseModel):
    type: str
    message: str
    code: str

class ErrorResponse(BaseModel):
    error: ErrorDetail


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

@app.post("/api/v1/chat/completions", response_model=ChatCompletionResponse, summary="OpenAI Chat Completions API 호환 엔드포인트", responses={
    401: {"model": ErrorResponse, "description": "인증 오류"}
})
async def chat_completions(
    request: ChatCompletionRequest,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
):
    """
    OpenAI Chat Completions API와 호환되는 엔드포인트입니다.
    
    - **model**: 모델 이름 (기본값: kepco-rag-model)
    - **role**: 메시지 역할 (user 만 지원)
    - **content**: 질문 내용 ( 한국어만 지원 )
    - **temperature**: 샘플링 온도 (0.0 ~ 2.0 - KEPCO 미지원 )
    - **stream**: 스트리밍 여부 (현재 미지원 )
    - **max_tokens**: 최대 생성 토큰 수( KEPCO 미지원 )
    - **X-API-Key**: API 키 (헤더에 포함)
    
    """
    # API 키 검증
    expected_api_key = os.getenv("KEPCO_API_KEY")
    if not x_api_key:
        raise HTTPException(
            status_code=401, 
            detail={
                "error": {
                    "type": "authentication_error",
                    "message": "X-API-Key 헤더가 필요합니다.",
                    "code": "missing_api_key"
                }
            }
        )
    
    if x_api_key != expected_api_key:
        raise HTTPException(
            status_code=401, 
            detail={
                "error": {
                    "type": "authentication_error",
                    "message": "유효하지 않은 API 키입니다.",
                    "code": "invalid_api_key"
                }
            }
        )
    
    print(f"\n [Chat Completions API 요청 수신]")
    print(f"  Model: {request.model}")
    print(f"  Messages: {len(request.messages)}개")
    
    # 마지막 user 메시지를 질의로 추출
    user_query = None
    for msg in reversed(request.messages):
        if msg.role == "user":
            user_query = msg.content
            break
    
    if not user_query:
        raise HTTPException(status_code=400, detail="No user message found in messages")
    
    print(f"  질문: {user_query}")
    start_time = time.time()
    
    # RAG 응답 생성
    final_answer, retrieved_docs = await get_rag_response(user_query)
    
    if retrieved_docs:
        sources = list(set([doc.metadata.get("source", "N/A") for doc in retrieved_docs]))
        print(f"  -> 참고한 소스: {', '.join(sources)}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f" [API 응답 완료] 처리 시간: {elapsed:.2f}초")
    
    # OpenAI 호환 응답 생성
    import uuid
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    
    # 토큰 수 추정 (실제로는 토크나이저 필요)
    prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
    completion_tokens = len(final_answer.split())
    
    return ChatCompletionResponse(
        id=response_id,
        created=int(time.time()),
        model=request.model,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=Message(role="assistant", content=final_answer),
                finish_reason="stop"
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens
        )
    )

# Uvicorn으로 서버 실행 
if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. (http://127.0.0.1:5000)")
    uvicorn.run(app, host="0.0.0.0", port=5000)