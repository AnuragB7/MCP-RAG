import asyncio
import PyPDF2
from document_processors import ContentExtractor

async def test_pdf_extraction():
    """Test PDF extraction with different methods"""
    
    # Your problematic file
    file_path = "/Users/A200309906/Documents/large-file-rag-mcp/DTSE Documents/00179-000003-06-A_20180425_Agreement+Sideletter_sign.pdf"
    
    print("=== PDF Extraction Test ===")
    
    # Test 1: Basic PyPDF2
    print("\n1. Testing PyPDF2...")
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            print(f"Total pages: {total_pages}")
            
            # Test first few pages
            for i in range(min(3, total_pages)):
                page = pdf_reader.pages[i]
                text = page.extract_text()
                print(f"Page {i+1}: {len(text)} characters")
                if text:
                    print(f"Sample: {text[:100]}...")
                
    except Exception as e:
        print(f"PyPDF2 failed: {e}")
    
    # Test 2: pdfplumber
    print("\n2. Testing pdfplumber...")
    try:
        import pdfplumber
        with pdfplumber.open(file_path) as pdf:
            print(f"Total pages: {len(pdf.pages)}")
            
            for i in range(min(3, len(pdf.pages))):
                page = pdf.pages[i]
                text = page.extract_text()
                print(f"Page {i+1}: {len(text) if text else 0} characters")
                if text:
                    print(f"Sample: {text[:100]}...")
                    
    except Exception as e:
        print(f"pdfplumber failed: {e}")
    
    # Test 3: Our enhanced method
    print("\n3. Testing enhanced extraction...")
    try:
        content = await ContentExtractor.extract_pdf_content(file_path)
        print(f"Enhanced method: {len(content)} characters")
        if content:
            print(f"Sample: {content[:200]}...")
    except Exception as e:
        print(f"Enhanced method failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_pdf_extraction())