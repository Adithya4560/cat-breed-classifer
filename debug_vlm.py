#!/usr/bin/env python3
"""
Debug VLM functionality with actual image processing
"""
import asyncio
import io
from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main application components
from main import get_vlm_description, VLM_ENABLED

async def debug_vlm():
    print("🔍 Debugging VLM with realistic image...")
    
    # Create a more realistic test image (colorful cat-like image)
    test_image = Image.new('RGB', (224, 224), color=(255, 165, 0))  # Orange color like a cat
    
    # Add some pattern to make it more realistic
    for x in range(0, 224, 20):
        for y in range(0, 224, 20):
            if (x + y) % 40 == 0:
                for i in range(10):
                    for j in range(10):
                        if x+i < 224 and y+j < 224:
                            test_image.putpixel((x+i, y+j), (139, 69, 19))  # Brown stripes
    
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG', quality=90)
    img_data = img_byte_arr.getvalue()
    
    print(f"📊 Image data size: {len(img_data)} bytes")
    print(f"🔧 VLM_ENABLED before test: {VLM_ENABLED}")
    
    try:
        result = await get_vlm_description(img_data)
        
        print(f"🔧 VLM_ENABLED after test: {VLM_ENABLED}")
        print(f"📤 API Result: {result}")
        
        if result["success"]:
            print("✅ VLM test successful!")
            print(f"📝 Description: {result['description']}")
            return True
        else:
            print(f"❌ VLM test failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ VLM test error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_vlm())
    if success:
        print("\n🎉 VLM is working correctly!")
    else:
        print("\n⚠️ VLM has issues that need attention.")
