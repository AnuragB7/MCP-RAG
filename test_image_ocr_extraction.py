import asyncio
from document_processors import ContentExtractor

async def test_image_ocr():
    """Test image OCR capabilities"""
    
    # Test with sample images
    image_paths = [
        "path/to/screenshot.png",
        "path/to/receipt.jpg",
        "path/to/document.jpeg"
    ]
    
    print("=== Image OCR Test ===")
    
    for image_path in image_paths:
        try:
            print(f"\nTesting: {image_path}")
            content, file_type = await ContentExtractor.extract_content(image_path)
            
            print(f"File Type: {file_type}")
            print(f"Content Length: {len(content)} characters")
            print("Content Preview:")
            print("-" * 50)
            print(content[:500] + "..." if len(content) > 500 else content)
            print("-" * 50)
            
        except Exception as e:
            print(f"ERROR processing {image_path}: {e}")

if __name__ == "__main__":
    asyncio.run(test_image_ocr())