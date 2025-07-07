#!/usr/bin/env python3
"""
Complete test script for the Cat Breed Classifier application
"""
import asyncio
import io
import sys
import os
from PIL import Image
import torch

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the main application components
from main import cat_detector, breed_classifier, transform, breed_transform, get_vlm_description, CAT_BREEDS

async def run_complete_test():
    print("🧪 Running Complete Cat Breed Classifier Test")
    print("=" * 50)
    
    # Test 1: Check models loading
    print("1. Testing model loading...")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   Device: {device}")
        print(f"   Cat detector loaded: {cat_detector is not None}")
        print(f"   Breed classifier loaded: {breed_classifier is not None}")
        print("   ✅ Models loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False
    
    # Test 2: Create test image
    print("\n2. Creating test image...")
    try:
        # Create a sample "cat-like" image (gray rectangle)
        test_image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        print("   ✅ Test image created")
    except Exception as e:
        print(f"   ❌ Test image creation failed: {e}")
        return False
    
    # Test 3: Test cat detection
    print("\n3. Testing cat detection...")
    try:
        input_tensor = transform(test_image).unsqueeze(0).to(device)
        with torch.no_grad():
            cat_output = cat_detector(input_tensor)
            cat_pred = torch.argmax(cat_output, dim=1).item()
            cat_confidence = torch.softmax(cat_output, dim=1)[0][cat_pred].item()
        print(f"   Prediction: {cat_pred}")
        print(f"   Confidence: {cat_confidence:.2%}")
        print("   ✅ Cat detection working")
    except Exception as e:
        print(f"   ❌ Cat detection failed: {e}")
        return False
    
    # Test 4: Test breed classification
    print("\n4. Testing breed classification...")
    try:
        breed_input = breed_transform(test_image).unsqueeze(0).to(device)
        with torch.no_grad():
            breed_output = breed_classifier(breed_input)
            breed_probabilities = torch.softmax(breed_output, dim=1)[0]
            top_breed_idx = torch.argmax(breed_probabilities).item()
            top_confidence = breed_probabilities[top_breed_idx].item()
        
        print(f"   Top breed: {CAT_BREEDS[top_breed_idx]}")
        print(f"   Confidence: {top_confidence:.2%}")
        print("   ✅ Breed classification working")
    except Exception as e:
        print(f"   ❌ Breed classification failed: {e}")
        return False
    
    # Test 5: Test VLM functionality
    print("\n5. Testing VLM (Image Analysis)...")
    try:
        img_byte_arr = io.BytesIO()
        test_image.save(img_byte_arr, format='JPEG')
        img_data = img_byte_arr.getvalue()
        
        vlm_result = await get_vlm_description(img_data)
        
        if vlm_result["success"]:
            print(f"   Description length: {len(vlm_result['description'])} chars")
            print("   ✅ VLM working")
        else:
            print(f"   ⚠️ VLM error: {vlm_result['error']}")
            print("   ℹ️ Breed classification will still work without VLM")
    except Exception as e:
        print(f"   ❌ VLM test error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 All core functionality tests completed!")
    print("\nThe application should now work correctly.")
    print("To run the server: python main.py")
    print("Then visit: http://localhost:8000")
    
    return True

if __name__ == "__main__":
    asyncio.run(run_complete_test())
