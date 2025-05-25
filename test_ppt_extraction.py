import asyncio
from document_processors import ContentExtractor

async def test_powerpoint_extraction():
    """Test PowerPoint extraction capabilities"""
    
    # Test with a sample PPTX file
    file_path = "path/to/your/presentation.pptx"
    
    print("=== PowerPoint Extraction Test ===")
    
    try:
        content, file_type = await ContentExtractor.extract_content(file_path)
        print(f"\nFile Type: {file_type}")
        print(f"Extracted content length: {len(content)} characters")
        
        if content and len(content) > 100:
            print("SUCCESS! Content extracted:")
            print("-" * 50)
            print(content[:1000] + "..." if len(content) > 1000 else content)
            print("-" * 50)
        else:
            print("FAILED: No substantial content extracted")
            print(f"Content: {content[:200]}")
            
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_powerpoint_extraction())