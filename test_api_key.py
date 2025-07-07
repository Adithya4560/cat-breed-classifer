#!/usr/bin/env python3
"""
Test OpenRouter API key authentication
"""
import requests
import os
from dotenv import load_dotenv

def test_api_key():
    load_dotenv()
    api_key = os.getenv('api_key')
    
    print("🔑 API Key Analysis:")
    print(f"   Length: {len(api_key) if api_key else 0}")
    print(f"   Format: {'✅ Valid sk-or- format' if api_key and api_key.startswith('sk-or-') else '❌ Invalid format'}")
    print(f"   Preview: {api_key[:30]}..." if api_key else "   ❌ No API key found")
    
    if not api_key:
        print("\n❌ No API key found in .env file")
        return False
    
    # Test 1: Try with a free model
    print("\n🧪 Testing with free model...")
    headers = {
        'Authorization': f'Bearer {api_key}',
        'HTTP-Referer': 'http://localhost:8000',
        'X-Title': 'Cat Breed Classifier',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': 'meta-llama/llama-3.2-3b-instruct:free',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'max_tokens': 10
    }
    
    try:
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', 
                               headers=headers, json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Free model test successful!")
        else:
            print(f"   ❌ Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    # Test 2: Try with DeepSeek R1 (original model)
    print("\n🧪 Testing with DeepSeek R1...")
    payload['model'] = 'deepseek/deepseek-r1'
    
    try:
        response = requests.post('https://openrouter.ai/api/v1/chat/completions', 
                               headers=headers, json=payload, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ DeepSeek R1 test successful!")
            return True
        else:
            print(f"   ❌ Error: {response.text[:300]}")
            
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    # Test 3: Check OpenRouter account status
    print("\n🔍 Checking account status...")
    try:
        response = requests.get('https://openrouter.ai/api/v1/auth/key', 
                              headers={'Authorization': f'Bearer {api_key}'})
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Account: {data.get('data', {}).get('label', 'Valid')}")
        else:
            print(f"   ❌ Account error: {response.text[:200]}")
            
    except Exception as e:
        print(f"   ❌ Account check error: {e}")
    
    return False

if __name__ == "__main__":
    test_api_key()
