"""
Hybrid LLM Document Loader 테스트 스크립트

사용법:
    python test_hybrid_loader.py <PDF_파일_경로>

예시:
    python test_hybrid_loader.py /Volumes/SSD_black/한전/원본문서/sample.pdf
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.handler.hybrid_llm_document_loader import HybridLLMPdfLoader
import json


def test_pdf_parsing(pdf_path: str, max_preview_length: int = 500):
    """
    PDF 파일을 파싱하고 결과를 출력합니다.
    
    Args:
        pdf_path: PDF 파일 경로
        max_preview_length: 각 청크의 미리보기 최대 길이
    """
    if not os.path.exists(pdf_path):
        print(f"❌ 파일을 찾을 수 없습니다: {pdf_path}")
        return
    
    file_name = os.path.basename(pdf_path)
    print(f"\n{'='*80}")
    print(f"📄 PDF 파일: {file_name}")
    print(f"📂 경로: {pdf_path}")
    print(f"{'='*80}\n")
    
    # 로더 생성
    print("🔧 HybridLLMPdfLoader 초기화 중...")
    loader = HybridLLMPdfLoader(
        file_path=pdf_path,
        file=pdf_path,
        file_name=file_name,
        # llm_model="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        llm_model="global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        aws_region="us-east-1"
    )
    
    # 문서 로드
    print("📖 PDF 파싱 시작...\n")
    documents = loader.load()
    
    print(f"\n{'='*80}")
    print(f"✅ 파싱 완료!")
    print(f"📊 총 {len(documents)}개의 Document 생성됨")
    print(f"{'='*80}\n")
    
    # 페이지별로 그룹화
    pages_data = {}
    for doc in documents:
        page_num = doc.metadata.get("page", 0)
        if page_num not in pages_data:
            pages_data[page_num] = []
        pages_data[page_num].append(doc)
    
    # 페이지별 통계 출력
    print("\n" + "="*80)
    print("📊 페이지별 처리 결과 요약")
    print("="*80 + "\n")
    
    for page_num in sorted(pages_data.keys()):
        docs_in_page = pages_data[page_num]
        extract_mode = docs_in_page[0].metadata.get("extract_mode", "unknown")
        table_count = docs_in_page[0].metadata.get("table_count", 0)
        
        # 처리 방식 표시
        if "llm" in extract_mode:
            method_icon = "🤖"
            method_name = "LLM 처리"
        else:
            method_icon = "📝"
            method_name = "pdfplumber 처리"
        
        print(f"{method_icon} 페이지 {page_num:2d}: {method_name:20s} | "
              f"테이블 {table_count}개 | 청크 {len(docs_in_page):2d}개 | "
              f"모드: {extract_mode}")
    
    # 상세 내용 출력
    print("\n" + "="*80)
    print("📝 페이지별 상세 내용")
    print("="*80 + "\n")
    
    for page_num in sorted(pages_data.keys()):
        docs_in_page = pages_data[page_num]
        extract_mode = docs_in_page[0].metadata.get("extract_mode", "unknown")
        table_count = docs_in_page[0].metadata.get("table_count", 0)
        
        print(f"\n{'─'*80}")
        print(f"📄 페이지 {page_num} (청크 {len(docs_in_page)}개)")
        print(f"{'─'*80}")
        print(f"  처리 방식: {extract_mode}")
        print(f"  테이블 개수: {table_count}")
        print(f"  청크 개수: {len(docs_in_page)}")
        
        for idx, doc in enumerate(docs_in_page, 1):
            content = doc.page_content
            
            # 파일명 제거 (파일명\n\n실제내용 형식)
            if "\n\n" in content:
                _, actual_content = content.split("\n\n", 1)
            else:
                actual_content = content
            
            # 미리보기
            preview = actual_content[:max_preview_length]
            if len(actual_content) > max_preview_length:
                preview += "..."
            
            print(f"\n  [{idx}번째 청크] (길이: {len(actual_content)}자)")
            print(f"  ┌{'─'*76}┐")
            for line in preview.split("\n"):
                print(f"  │ {line[:74]:<74s} │")
            print(f"  └{'─'*76}┘")
    
    # 통계 요약
    print("\n" + "="*80)
    print("📊 최종 통계")
    print("="*80 + "\n")
    
    llm_pages = sum(1 for p in pages_data.values() if "llm" in p[0].metadata.get("extract_mode", ""))
    plumber_pages = len(pages_data) - llm_pages
    total_chunks = len(documents)
    
    print(f"  총 페이지 수: {len(pages_data)}")
    print(f"  🤖 LLM 처리 페이지: {llm_pages}")
    print(f"  📝 pdfplumber 처리 페이지: {plumber_pages}")
    print(f"  📦 총 청크 수: {total_chunks}")
    
    # 테이블 개수별 통계
    table_stats = {}
    for docs_in_page in pages_data.values():
        count = docs_in_page[0].metadata.get("table_count", 0)
        table_stats[count] = table_stats.get(count, 0) + 1
    
    print(f"\n  테이블 개수별 페이지 분포:")
    for count in sorted(table_stats.keys()):
        print(f"    테이블 {count}개: {table_stats[count]}페이지")
    
    print("\n" + "="*80 + "\n")
    
    # 결과를 JSON 파일로 저장 (옵션)
    save_json = input("📁 결과를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
    if save_json == 'y':
        # 저장 디렉토리 설정
        output_dir = "/Volumes/SSD_black/한전/parsing_result"
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 생성 (PDF 파일명에서 확장자 제거 후 _parsed_result.json 추가)
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_parsed_result.json"
        output_path = os.path.join(output_dir, output_filename)
        
        result = {
            "file_name": file_name,
            "total_pages": len(pages_data),
            "total_chunks": total_chunks,
            "llm_pages": llm_pages,
            "plumber_pages": plumber_pages,
            "pages": []
        }
        
        for page_num in sorted(pages_data.keys()):
            docs_in_page = pages_data[page_num]
            page_info = {
                "page_num": page_num,
                "extract_mode": docs_in_page[0].metadata.get("extract_mode"),
                "table_count": docs_in_page[0].metadata.get("table_count"),
                "chunks": []
            }
            
            for doc in docs_in_page:
                content = doc.page_content
                if "\n\n" in content:
                    _, actual_content = content.split("\n\n", 1)
                else:
                    actual_content = content
                
                page_info["chunks"].append({
                    "content": actual_content,
                    "length": len(actual_content),
                    "metadata": doc.metadata
                })
            
            result["pages"].append(page_info)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"✅ JSON 파일 저장 완료: {output_path}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python test_hybrid_loader.py <PDF_파일_경로>")
        print("예시: python test_hybrid_loader.py /Volumes/SSD_black/한전/원본문서/sample.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    test_pdf_parsing(pdf_path)

