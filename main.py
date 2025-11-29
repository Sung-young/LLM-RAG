import uvicorn
import time
from typing import Optional
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from src.main import get_rag_response 

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

class QueryResponse(BaseModel):
    answer: str
    sender: str # 응답을 요청한 사용자 이름 표시 (확인용)

# ---------------------------------------------------------
# 보안 및 토큰 처리 함수
# ---------------------------------------------------------


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=30))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user_name(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    """Bearer 토큰만으로 인증"""
    token = credentials.credentials

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_name = payload.get("name")
        employee_id = payload.get("sub")

        if not user_name or not employee_id:
            raise HTTPException(status_code=401, detail="토큰 정보가 올바르지 않습니다.")
        
        return user_name

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
    user_name: str = Depends(get_current_user_name) # 여기서 토큰 검증 및 이름 추출이 자동으로 수행됨
):
    """
    사용자 쿼리를 입력받아 RAG 체인을 통해 생성된 답변을 반환합니다.
    (Header에 'Authorization: Bearer <Token>' 필요)
    """
    print(f"\n [API 요청 수신] 질문: {request.query} | 요청자: {user_name}")
    
    start_time = time.time()
    
    # 수정된 부분: get_rag_response 함수에 query와 user_name을 함께 전달
    # 주의: src.main의 get_rag_response 함수도 인자를 받도록 수정되어야 합니다.
    try:
        # 만약 기존 함수가 이름 인자를 받지 않는다면 아래처럼 호출하고,
        # 기존 함수 수정이 필요하다면 get_rag_response(request.query, user_name) 으로 변경하세요.
        # 여기서는 요청사항에 따라 이름을 넣는 것으로 가정합니다.
        final_answer, retrieved_docs = await get_rag_response(request.query, user_name)
    except TypeError:
        # 만약 src.main을 아직 수정하지 않아 인자 오류가 날 경우를 대비한 처리
        final_answer, retrieved_docs = await get_rag_response(request.query)
        final_answer = f"(시스템 알림: {user_name}님, 백엔드 함수 인자 업데이트가 필요합니다.)\n" + final_answer

    if retrieved_docs:
        sources = list(set([doc.metadata.get("source", "N/A") for doc in retrieved_docs]))
        print(f" -> (응답에 미포함) 참고한 소스: {', '.join(sources)}")
    else:
        print(f" -> 참고한 소스를 찾지 못했습니다.")

    end_time = time.time()
    elapsed = end_time - start_time
    print(f" [API 응답 완료] 처리 시간: {elapsed:.2f}초")

    return QueryResponse(
        answer=final_answer,
        sender=user_name
    )

@app.get("/", summary="API 상태 확인")
def read_root():
    return {"status": "RAG API is running", "auth_mode": "enabled"}

# Uvicorn으로 서버 실행
if __name__ == "__main__":
    print("FastAPI 서버를 시작합니다. (http://127.0.0.1:8000)")
    uvicorn.run(app, host="0.0.0.0", port=8000)