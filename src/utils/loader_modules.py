import requests
import re, os
from PIL import Image
from io import BytesIO
import base64
import time
import json
import uuid
import asyncio
from openai import AsyncOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from dotenv import load_dotenv, find_dotenv
import datetime
import unicodedata


load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    print("⚠️ OPENAI_API_KEY가 .env에서 로드되지 않았습니다.")


def preprocessing(text):
    """
    향상된 텍스트 전처리 함수
    보험 문서에 특화된 정리 기능 포함
    """
    if not text:
        return ""
    
    # 1. 기본 정리
    # 불필요한 공백, 줄바꿈 반복 제거
    while True:
        original_text = text
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        if text == original_text:
            break
    
    # 2. OCR 오류 수정
    # 두 글자씩 연속으로 출력되는 텍스트 수정 (예: "보험험" → "보험")
    text = re.sub(r'(\D)\1', r'\1', text)
    
    # 세 글자 이상 연속되는 경우도 처리 (예: "보험험험" → "보험")
    text = re.sub(r'(\D)\1{2,}', r'\1', text)
    
    # 3. 특수문자 정리 (보험 문서에 필요한 것만 유지)
    # 허용할 특수문자: 한글, 영문, 숫자, 기본 문장부호, 보험 관련 특수문자
    allowed_chars = r'[^\s가-힣a-zA-Z0-9!@#%^&*()_+\-={}[\\]:;,.?<>|₩€¥$±×÷≠≡≈≤≥∞()[\]{}%‰°′″§¶†‡©®™]'
    text = re.sub(allowed_chars, '', text)
    
    # 4. 보험 문서 특화 정리
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text)
    
    # 줄바꿈 정리 (문단 구분 유지)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()


def split_texts(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)

    return text_chunks


def to_documents(file_name, file_path, text_chunks, doc_idx=0, page_num=1, total_pages=1):
    documents = []
    

    for text_chunk in text_chunks:
        documents.append(
            Document(
                page_content=file_name + "\n\n" + text_chunk,  # 본문만!
                metadata={
                    "source": unicodedata.normalize('NFC', file_path.split("/")[-1]),
                    "text_length": len(text_chunk),
                    "id": doc_idx,
                    "page": page_num,
                    "total_pages": total_pages,
                },
            )
        )
        doc_idx += 1
    return documents, doc_idx


def enhance_image_for_ocr(img: Image.Image) -> Image.Image:
    """
    OCR 인식률 향상을 위한 고급 이미지 전처리
    - 그레이스케일 변환
    - 대비 향상
    - 선명도 향상
    - 노이즈 제거
    - 이진화 (적응형 임계값)
    - 모폴로지 연산
    """
    # 그레이스케일 변환
    if img.mode != 'L':
        img = img.convert('L')
    
    # PIL의 ImageEnhance를 사용한 기본 향상
    from PIL import ImageEnhance
    import numpy as np
    import cv2
    
    # PIL 이미지를 OpenCV 형식으로 변환
    img_array = np.array(img)
    
    # 1. 노이즈 제거 (가우시안 블러)
    img_array = cv2.GaussianBlur(img_array, (1, 1), 0)
    
    # 2. 적응형 히스토그램 평활화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # 3. 모폴로지 연산으로 텍스트 선명도 향상
    kernel = np.ones((1,1), np.uint8)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    
    # 4. 적응형 이진화 (Otsu's method)
    _, img_array = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # OpenCV 배열을 PIL 이미지로 변환
    img = Image.fromarray(img_array)
    
    # 5. PIL의 ImageEnhance를 사용한 추가 향상
    # 대비 향상 (1.3배)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.3)
    
    # 선명도 향상 (1.2배)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.2)
    
    # 밝기 조정 (1.1배)
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(1.1)
    
    return img


def enhance_image_for_ocr_basic(img: Image.Image) -> Image.Image:
    """
    기본적인 OCR 전처리 (기존 방식)
    - 그레이스케일 변환
    - 대비 향상
    - 선명도 향상
    """
    # 그레이스케일 변환
    if img.mode != 'L':
        img = img.convert('L')
    
    # PIL의 ImageEnhance를 사용한 대비 향상
    from PIL import ImageEnhance
    
    # 대비 향상 (1.2배)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # 선명도 향상 (1.1배)
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    return img


def enhance_image_for_ocr_advanced(img: Image.Image) -> Image.Image:
    """
    고급 OCR 전처리 (더 강력한 방법)
    - 그레이스케일 변환
    - 노이즈 제거
    - 적응형 히스토그램 평활화
    - 적응형 이진화
    - 모폴로지 연산
    - 엣지 강화
    """
    # 그레이스케일 변환
    if img.mode != 'L':
        img = img.convert('L')
    
    import numpy as np
    import cv2
    
    # PIL 이미지를 OpenCV 형식으로 변환
    img_array = np.array(img)
    
    # 1. 노이즈 제거 (중간값 필터)
    img_array = cv2.medianBlur(img_array, 3)
    
    # 2. 적응형 히스토그램 평활화 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_array = clahe.apply(img_array)
    
    # 3. 가우시안 블러로 노이즈 추가 제거
    img_array = cv2.GaussianBlur(img_array, (1, 1), 0)
    
    # 4. 엣지 강화 (Unsharp Masking)
    gaussian = cv2.GaussianBlur(img_array, (0, 0), 2.0)
    img_array = cv2.addWeighted(img_array, 1.5, gaussian, -0.5, 0)
    
    # 5. 적응형 이진화
    img_array = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # 6. 모폴로지 연산으로 텍스트 정리
    kernel = np.ones((1,1), np.uint8)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    img_array = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    
    # OpenCV 배열을 PIL 이미지로 변환
    img = Image.fromarray(img_array)
    
    return img



async def ocr_with_gpt(img: Image.Image) -> str:
    # 이미지 전처리로 인식률 향상
    img = enhance_image_for_ocr(img)
    
    client = AsyncOpenAI()
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    summary_prompt = """
        당신은 이미지 기반 PDF 문서에서 텍스트를 완벽하게 추출하는 OCR 전문가입니다. RAG 시스템의 지식 베이스 구축을 위해 문서의 모든 텍스트 내용을 빠짐없이 추출해야 합니다.

        **핵심 목표: 완전한 텍스트 추출 (요약 금지)**
        - 문서에 있는 모든 텍스트를 하나도 빠뜨리지 않고 추출
        - 원본 텍스트의 정확성과 완전성을 최우선으로 함
        - 어떤 내용도 요약하거나 생략하지 않음

        **완전 추출 지침:**

        1. **전체 텍스트 추출:**
            - 제목, 부제목, 본문, 각주, 캡션 등 모든 텍스트
            - 헤더, 푸터, 페이지 번호까지 포함
            - 작은 글씨나 흐릿한 텍스트도 최대한 정확히 추출
            - 광고, 안내문구, 연락처 등도 모두 포함

        2. **구조 정보 유지:**
            - **제목 계층:** 크기와 굵기에 따라 # ## ### 마크다운으로 표현
            - **단락 구분:** 원본의 단락과 줄바꿈 구조 그대로 유지
            - **목록:** 번호나 불릿포인트 형식 원본 그대로 재현
            - **표:** 모든 행과 열을 마크다운 테이블로 완전 변환

        3. **정확한 데이터 추출:**
            - **숫자 데이터:** 금액, 비율, 날짜, 전화번호 등 한 글자도 틀리지 않게
            - **고유명사:** 회사명, 상품명, 인명 등 정확한 표기
            - **전문용어:** 보험/금융 용어의 정확한 한글 표기
            - **특수문자:** 괄호, 하이픈, 슬래시 등 모든 기호 포함

        4. **시각적 요소 텍스트화:**
            - **표와 차트:** 모든 데이터를 표 형식으로 완전 변환
            - **이미지 속 텍스트:** 로고나 이미지 안의 텍스트도 모두 추출
            - **도식/다이어그램:** 텍스트 요소만 추출하고 `[도식: 설명]` 추가
            - **순수 이미지:** `[이미지: 간단한 설명]`로만 표시

        5. **OCR 품질 최적화:**
            - 불분명한 글자는 맥락을 고려해 추정하되, 확실하지 않으면 `[불분명: ?]` 표시
            - 명백한 OCR 오류(예: "보험험"→"보험")는 수정
            - 띄어쓰기와 문장부호를 자연스럽게 보정

        6. **출력 형식 - 순수 텍스트만:**
            - 메타데이터나 분석 내용 추가 금지
            - "다음은 추출한 내용입니다" 같은 서론 금지
            - 오직 문서의 원본 텍스트만 출력
            - 마크다운 구조화는 허용 (가독성을 위해)

        **절대 금지 사항:**
        - 내용 요약이나 압축
        - 중요도에 따른 선별적 추출
        - 개인적 해석이나 분석 추가
        - 원본에 없는 설명이나 주석

        **완전성 체크리스트:**
        □ 모든 페이지의 텍스트 추출 완료
        □ 표, 목록, 각주까지 모든 구조적 요소 포함
        □ 숫자, 날짜, 고유명사의 정확성 확인
        □ 원본 문서와 대조하여 누락된 내용 없음 확인

        ---
        이제 주어진 PDF 이미지에서 모든 텍스트를 완전히 추출하세요. 한 글자도 놓치지 마세요.
        """

    response = await client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "/home/concat/바탕화면/insurance-news-rag/data0905/2. 소식지/2025년 9월/생명보험사/202509_농협생명 소식지.pdf",
                        "file_data": f"data:application/pdf;base64,{base64.b64decode(img_str)}",
                    },
                    {
                        "type": "input_text",
                        "text": summary_prompt,
                    },
                ],
            },
        ]
    )

    return response.output_text

async def ocr_pdf_with_gpt(pdf_bytes: bytes) -> str:
    """PDF 바이트를 직접 GPT로 OCR 처리"""
    client = AsyncOpenAI()
    base64_data = base64.b64encode(pdf_bytes).decode()

    summary_prompt = """
        당신은 이미지 기반 PDF 문서에서 텍스트를 완벽하게 추출하는 OCR 전문가입니다. RAG 시스템의 지식 베이스 구축을 위해 문서의 모든 텍스트 내용을 빠짐없이 추출해야 합니다.

        **핵심 목표: 완전한 텍스트 추출 (요약 금지)**
        - 문서에 있는 모든 텍스트를 하나도 빠뜨리지 않고 추출
        - 원본 텍스트의 정확성과 완전성을 최우선으로 함
        - 어떤 내용도 요약하거나 생략하지 않음

        **완전 추출 지침:**

        1. **전체 텍스트 추출:**
            - 제목, 부제목, 본문, 각주, 캡션 등 모든 텍스트
            - 헤더, 푸터, 페이지 번호까지 포함
            - 작은 글씨나 흐릿한 텍스트도 최대한 정확히 추출
            - 광고, 안내문구, 연락처 등도 모두 포함

        2. **구조 정보 유지:**
            - **제목 계층:** 크기와 굵기에 따라 # ## ### 마크다운으로 표현
            - **단락 구분:** 원본의 단락과 줄바꿈 구조 그대로 유지
            - **목록:** 번호나 불릿포인트 형식 원본 그대로 재현
            - **표:** 모든 행과 열을 마크다운 테이블로 완전 변환

        3. **정확한 데이터 추출:**
            - **숫자 데이터:** 금액, 비율, 날짜, 전화번호 등 한 글자도 틀리지 않게
            - **고유명사:** 회사명, 상품명, 인명 등 정확한 표기
            - **전문용어:** 보험/금융 용어의 정확한 한글 표기
            - **특수문자:** 괄호, 하이픈, 슬래시 등 모든 기호 포함

        4. **시각적 요소 텍스트화:**
            - **표와 차트:** 모든 데이터를 표 형식으로 완전 변환
            - **이미지 속 텍스트:** 로고나 이미지 안의 텍스트도 모두 추출
            - **도식/다이어그램:** 텍스트 요소만 추출하고 `[도식: 설명]` 추가
            - **순수 이미지:** `[이미지: 간단한 설명]`로만 표시

        5. **OCR 품질 최적화:**
            - 불분명한 글자는 맥락을 고려해 추정하되, 확실하지 않으면 `[불분명: ?]` 표시
            - 명백한 OCR 오류(예: "보험험"→"보험")는 수정
            - 띄어쓰기와 문장부호를 자연스럽게 보정

        6. **출력 형식 - 순수 텍스트만:**
            - 메타데이터나 분석 내용 추가 금지
            - "다음은 추출한 내용입니다" 같은 서론 금지
            - 오직 문서의 원본 텍스트만 출력
            - 마크다운 구조화는 허용 (가독성을 위해)

        **절대 금지 사항:**
        - 내용 요약이나 압축
        - 중요도에 따른 선별적 추출
        - 개인적 해석이나 분석 추가
        - 원본에 없는 설명이나 주석

        **완전성 체크리스트:**
        □ 모든 페이지의 텍스트 추출 완료
        □ 표, 목록, 각주까지 모든 구조적 요소 포함
        □ 숫자, 날짜, 고유명사의 정확성 확인
        □ 원본 문서와 대조하여 누락된 내용 없음 확인

        ---
        이제 주어진 PDF 문서에서 모든 텍스트를 완전히 추출하세요. 한 글자도 놓치지 마세요.
        """

    response = await client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "document.pdf",
                        "file_data": f"data:application/pdf;base64,{base64_data}",
                    },
                    {
                        "type": "input_text",
                        "text": summary_prompt,
                    },
                ],
            },
        ],
    )

    return response.output_text




class loader_modules:
    @staticmethod
    def preprocessing(text):
        return preprocessing(text)

    @staticmethod
    def split_texts(file_path, text):
        return split_texts(file_path, text)

    @staticmethod
    def to_documents(file_path, text_chunks, doc_idx, page_num, total_pages):
        return to_documents(file_path, text_chunks, doc_idx, page_num, total_pages)