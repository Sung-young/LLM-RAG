import uvicorn
import time
from typing import Optional, Dict
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from src.main import get_rag_response 
from src.utils.conversation_manager import ConversationManager
from fastapi.responses import StreamingResponse

# ---------------------------------------------------------
# JWT 토큰 관련 설정
# ---------------------------------------------------------
SECRET_KEY = "my_super_secret_key_change_this_in_production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 # 토큰 유효시간 (60분)

app = FastAPI(
    title="LLM RAG API with Auth",
    description="한전 챗봇 API (인증 기능 포함)",
    version="1.1.0"
)

auth_scheme = HTTPBearer()

conversation_manager = ConversationManager()

# ---------------------------------------------------------
# Pydantic 데이터 모델 정의
# ---------------------------------------------------------

class LoginRequest(BaseModel):
    employee_id: str = Field(..., min_length=8, max_length=8, description="사번 8자리")
    name: str = Field(..., description="이름")

class Token(BaseModel):
    access_token: str
    token_type: str
    username: str

class QueryRequest(BaseModel):
    query: str
    stream: bool = False 

class QueryResponse(BaseModel):
    answer: str
    sender: str 
    session_id: str

# ---------------------------------------------------------
# 보안 및 토큰 처리 함수
# ---------------------------------------------------------


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_info(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Bearer 토큰만으로 인증"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_name = payload.get("name")
        employee_id = payload.get("sub")

        if not user_name or not employee_id:
            raise HTTPException(status_code=401, detail="토큰 정보가 올바르지 않습니다.")
        
        return {"id": employee_id, "name": user_name}

    except JWTError:
        raise HTTPException(status_code=401, detail="토큰 검증 실패")

# ---------------------------------------------------------
# [API] 엔드포인트 정의
# ---------------------------------------------------------

@app.post("/login", response_model=Token, summary="인증 및 토큰 발급")
async def login(request: LoginRequest):
    """
    사번(8자리)과 이름을 받아 인증 토큰을 발급합니다.
    """
    # 1. 사번 길이 검증 (Pydantic Field에서도 체크하지만 로직으로 명시)
    if len(request.employee_id) != 8:
        raise HTTPException(
            status_code=400,
            detail="사번은 8자리여야 합니다."
        )
    
    # 2. 토큰 생성 (Payload에 이름과 사번을 담음)
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": request.employee_id, "name": request.name},
        expires_delta=access_token_expires
    )
    
    print(f"\n [로그인 성공] 사번: {request.employee_id}, 이름: {request.name}")
    
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "username": request.name
    }


@app.post("/query", response_model=QueryResponse, summary="RAG 질의응답 (인증 필요)")
async def handle_query(
    request: QueryRequest, 
    user_info: Dict[str, str] = Depends(get_current_user_info)
):
    """
    사용자 쿼리를 입력받아 RAG 체인을 통해 생성된 답변을 반환합니다.
    (Header에 'Authorization: Bearer <Token>' 필요)
    """
    user_name = user_info["name"]
    employee_id = user_info["id"] # Session ID
    
    print(f"\n [API 요청] User: {user_name}({employee_id}) | Query: {request.query}")
    
    start_time = time.time()
    
    final_answer, retrieved_docs = await get_rag_response(
        user_query=request.query,
        user_name=user_name,
        conversation_manager=conversation_manager, 
        session_id=employee_id,
        stream_mode=request.stream                   
    )

    end_time = time.time()
    elapsed = end_time - start_time
    print(f" [API 응답 완료] 처리 시간: {elapsed:.2f}초")

    # --- 스트리밍 응답 ---
    if request.stream:
        return StreamingResponse(
            final_answer, 
            media_type="text/event-stream", 
        )
    
    # --- 일반 JSON 응답 ---
    else:
        # response_data는 여기서 str(완성된 답변)입니다.
        return QueryResponse(
            answer=final_answer,
            sender=user_name,
            session_id=employee_id
        )

@app.get("/history/{employee_id}", summary="[관리자용] 특정 사원 대화기록 조회")
async def get_history(employee_id: str):
    """
    특정 사번의 대화 내용을 조회합니다.
    """
    # 이미 만들어둔 conversation_manager 사용
    messages = conversation_manager.get_messages(employee_id)
    
    if not messages:
        return {"message": "대화 기록이 없거나 만료되었습니다."}
        
    return {
        "employee_id": employee_id,
        "count": len(messages),
        "messages": messages
    }

# Uvicorn으로 서버 실행
if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. (http://127.0.0.1:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)