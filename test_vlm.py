#!/usr/bin/env python3
"""
Test script for VLM (Vision Language Model) functionality
"""
import asyncio
import io
from PIL import Image
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main application components
from main import get_vlm_description, DEEPSEEK_API_KEY

async def test_vlm():
    print("Testing VLM functionality...")
    
    # Check API key
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
        print("❌ API key not configured")
        return False
    
    print(f"✅ API key configured: {DEEPSEEK_API_KEY[:20]}...")
    
    # Create a simple test image (white square)
    test_image = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    test_image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    
    print("🧪 Testing VLM with sample image...")
    
    try:
        result = await get_vlm_description(img_byte_arr)
        
        if result["success"]:
            print("✅ VLM test successful!")
            print(f"Description: {result['description']}")
            return True
        else:
            print(f"❌ VLM test failed: {result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ VLM test error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_vlm())
    if success:
        print("\n🎉 VLM functionality is working correctly!")
    else:
        print("\n⚠️ VLM functionality needs attention.")
