import asyncio
from document_processors import ContentExtractor

async def test_ocr_extraction():
    """Test OCR extraction capabilities"""
    
    file_path = "/Users/A200309906/Documents/large-file-rag-mcp/DTSE Documents/00179-000003-06-A_20180425_Agreement+Sideletter_sign.pdf"
    
    print("=== Enhanced PDF Extraction Test ===")
    
    # Test enhanced extraction
    try:
        content = await ContentExtractor.extract_pdf_content_enhanced(file_path)
        print(f"\nExtracted content length: {len(content)} characters")
        
        if content and len(content) > 100:
            print("SUCCESS! Content extracted:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
        else:
            print("FAILED: No substantial content extracted")
            print(f"Content: {content[:200]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_ocr_extraction())