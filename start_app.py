#!/usr/bin/env python3
"""
Startup script for the Cat Breed Classifier application
Includes pre-flight checks and helpful information
"""
import os
import sys
import subprocess
import asyncio
from pathlib import Path

def check_requirements():
    """Check if all required files and dependencies are present"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required files
    required_files = [
        "main.py",
        "requirements.txt",
        "cat_notcat_model.pth",
        "best_efficientnet_b0.pth",
        "templates/index.html",
        "static/style.css",
        ".env"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files present")
    
    # Check .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv('api_key')
        if not api_key or api_key == "YOUR_API_KEY_HERE":
            print("⚠️ API key not configured - VLM features will be limited")
        else:
            print("✅ API key configured")
    except ImportError:
        print("⚠️ python-dotenv not available")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✅ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def main():
    """Main startup function"""
    print("🐱 Cat Breed Classifier - Startup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path("main.py").exists():
        print("❌ Please run this script from the project directory")
        return
    
    # Check requirements
    if not check_requirements():
        print("\n🛠️ Attempting to install dependencies...")
        if not install_dependencies():
            print("\n❌ Setup failed. Please check the error messages above.")
            return
    
    print("\n🚀 Starting the Cat Breed Classifier...")
    print("📍 The application will be available at: http://localhost:8000")
    print("⏹️ Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Start the main application
        subprocess.run([sys.executable, "main.py"])
    except KeyboardInterrupt:
        print("\n\n👋 Cat Breed Classifier stopped. Thanks for using our app!")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("1. Make sure all model files (.pth) are present")
        print("2. Check that port 8000 is not in use")
        print("3. Verify your Python environment")
        print("4. Run 'python test_app.py' to diagnose issues")

if __name__ == "__main__":
    main()
