import redis
import json
import datetime
from typing import List, Dict, Optional

class ConversationManager:
    """
    Redis를 활용한 대화 히스토리 관리 클래스
    (Rolling Summary 기능 제거됨)
    """
    
    def __init__(self, redis_host: str = "localhost", redis_port: int = 6379, redis_db: int = 0):
        try:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_timeout=5,
                socket_connect_timeout=5
            )
            # Redis 연결 테스트
            self.redis_client.ping()
            print(f"✅ Redis 연결 성공 ({redis_host}:{redis_port})")
        except (redis.ConnectionError, redis.TimeoutError) as e:
            print(f"⚠️ Redis 연결 실패: {e}")
            print("   -> 대화 히스토리 기능이 비활성화됩니다. (메모리 모드로 동작)")
            self.redis_client = None
            self.memory_storage = {}  # Redis 연결 실패 시 메모리 사용
    
    def add_message(self, session_id: str, role: str, content: str):
        """대화 메시지 추가 (role: 'user' or 'assistant')"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        if self.redis_client:
            key = f"conversation:{session_id}"
            self.redis_client.rpush(key, json.dumps(message, ensure_ascii=False))
            # 세션 만료 시간 설정 (예: 7일)
            self.redis_client.expire(key, 7 * 24 * 60 * 60)
        else:
            # 메모리에 저장
            if session_id not in self.memory_storage:
                self.memory_storage[session_id] = []
            self.memory_storage[session_id].append(message)
    
    def get_messages(self, session_id: str) -> List[Dict]:
        """세션의 모든 메시지 조회"""
        if self.redis_client:
            key = f"conversation:{session_id}"
            messages = self.redis_client.lrange(key, 0, -1)
            return [json.loads(msg) for msg in messages]
        else:
            return self.memory_storage.get(session_id, [])
    
    def clear_session(self, session_id: str):
        """세션 기록 전체 삭제"""
        if self.redis_client:
            self.redis_client.delete(f"conversation:{session_id}")
        else:
            self.memory_storage.pop(session_id, None)