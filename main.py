from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.staticfiles import StaticFiles
import json
import base64
import requests
from typing import Dict, Any, Optional
import logging
import re
import os
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Configure logging for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Cat Breed Classifier",
    description="AI-powered cat breed classification with detailed image analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# DeepSeek API configuration
DEEPSEEK_API_KEY = os.getenv("api_key") or os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Global VLM state
VLM_ENABLED = False
VLM_LAST_CHECK = None
VLM_ERROR_COUNT = 0
MAX_VLM_ERRORS = 3

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Cat breeds configuration
CAT_BREEDS = [
    "Abyssinian",
    "Bengal",
    "Birman", 
    "Bombay",
    "British_Shorthair",
    "Egyptian_Mau",
    "Maine_Coon",
    "Persian",
    "Ragdoll",
    "Russian_Blue",
    "Siamese",
    "Sphynx"
]

# Breed explanations
BREED_EXPLANATIONS = {
    "Abyssinian": "Abyssinians are elegant, slender cats with large ears and a playful, active nature. They have a distinctive ticked coat that gives a shimmering effect and are known for their curiosity and love for high places.",
    "Bengal": "Bengals have a wild appearance with a sleek, spotted coat resembling a leopard. They are energetic, intelligent, and love water, often displaying dog-like behaviors such as fetching.",
    "Birman": "Birmans are known for their striking blue eyes, silky medium-long fur, and color-pointed coat. They are affectionate, gentle, and often form strong bonds with their owners, earning the nickname 'Sacred Cat of Burma.'",
    "Bombay": "Bombays resemble miniature black panthers with their sleek, jet-black coats and copper or gold eyes. They are affectionate, playful, and thrive on attention, often acting like a lap cat.",
    "British_Shorthair": "British Shorthairs are sturdy, round-faced cats with dense, plush coats. They are calm, easygoing, and affectionate, making them great companions for families.",
    "Egyptian_Mau": "Egyptian Maus are sleek, athletic cats with a naturally spotted coat, giving them a wild appearance. They are fast, agile, and loyal, often forming strong bonds with their owners.",
    "Maine_Coon": "Maine Coons are one of the largest domesticated cat breeds, known for their tufted ears, bushy tails, and friendly, dog-like personalities. They have a shaggy coat and are highly sociable.",
    "Persian": "Persians are known for their long, luxurious fur and flat faces. They are calm, gentle, and prefer quiet environments, requiring regular grooming to maintain their coat.",
    "Ragdoll": "Ragdolls are large, relaxed cats with semi-long fur and striking blue eyes. They are known for their docile, affectionate nature and tendency to go limp when held, hence the name.",
    "Russian_Blue": "Russian Blues have short, dense, and plush blue-gray coats with vivid green eyes. They are elegant, intelligent, and reserved but form strong bonds with their owners.",
    "Siamese": "Siamese cats are slender with large ears, almond-shaped blue eyes, and vocal personalities. They are highly social, intelligent, and often demand attention from their owners.",
    "Sphynx": "Sphynx cats are hairless with wrinkled skin and large ears. They are energetic, affectionate, and love warmth due to their lack of fur, making them highly interactive pets."
}

# Model loading with error handling
def load_models():
    """Load the cat detection and breed classification models"""
    try:
        logger.info("Loading cat detection model...")
        cat_detector = models.mobilenet_v2(weights=None)
        cat_detector.classifier[1] = nn.Linear(cat_detector.last_channel, 2)
        
        if os.path.exists("cat_notcat_model.pth"):
            cat_detector.load_state_dict(torch.load("cat_notcat_model.pth", map_location=device))
            logger.info("✅ Cat detection model loaded successfully")
        else:
            logger.error("❌ cat_notcat_model.pth not found")
            raise FileNotFoundError("Cat detection model file not found")
        
        cat_detector.eval().to(device)
        
        logger.info("Loading breed classification model...")
        breed_classifier = models.efficientnet_b0(weights=None)
        breed_classifier.classifier[1] = nn.Linear(breed_classifier.classifier[1].in_features, len(CAT_BREEDS))
        
        if os.path.exists("best_efficientnet_b0.pth"):
            breed_classifier.load_state_dict(torch.load("best_efficientnet_b0.pth", map_location=device))
            logger.info("✅ Breed classification model loaded successfully")
        else:
            logger.error("❌ best_efficientnet_b0.pth not found")
            raise FileNotFoundError("Breed classification model file not found")
        
        breed_classifier.eval().to(device)
        
        return cat_detector, breed_classifier
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

# Load models
try:
    cat_detector, breed_classifier = load_models()
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    cat_detector = None
    breed_classifier = None

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

breed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def clean_description(text: str) -> str:
    """Clean VLM description text by removing markdown formatting"""
    if not text:
        return text
    
    original_text = text
    original_length = len(text)
    
    # Remove markdown formatting carefully
    text = re.sub(r'\*{3,}', '', text)  # Remove *** and more
    text = re.sub(r'\*{2}([^*]+)\*{2}', r'\1', text)  # Convert **text** to text
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # Convert *text* to text
    text = re.sub(r'#{1,6}\s*', '', text)  # Remove # headers
    
    # Remove list formatting more carefully
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)  # Remove bullet points
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)  # Remove numbered lists
    
    # Clean up spacing
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space
    text = text.strip()
    
    # If we've removed too much content, return the original
    if len(text) < original_length * 0.3:  # If we've removed more than 70% of content
        logger.warning(f"Cleaning removed too much content ({len(text)}/{original_length} chars), returning original")
        return original_text.strip()
    
    logger.info(f"Cleaned description: {original_length} -> {len(text)} chars")
    return text

async def test_api_key_validity() -> tuple[bool, str]:
    """Test if the API key is valid and return status with message"""
    global VLM_ENABLED, VLM_LAST_CHECK, VLM_ERROR_COUNT
    
    try:
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
            return False, "API key not configured"
        
        logger.info("Testing API key validity...")
        
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Cat Breed Classifier",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-r1",
            "messages": [{"role": "user", "content": "Hello, this is a test message."}],
            "max_tokens": 10
        }

        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            VLM_ENABLED = True
            VLM_ERROR_COUNT = 0
            VLM_LAST_CHECK = datetime.now()
            logger.info("✅ API key validation successful")
            return True, "API key is valid"
        elif response.status_code == 401:
            VLM_ENABLED = False
            VLM_ERROR_COUNT += 1
            logger.error("❌ API key validation failed - unauthorized")
            return False, "API key is invalid or expired"
        elif response.status_code == 402:
            VLM_ENABLED = False
            VLM_ERROR_COUNT += 1
            logger.error("❌ API key validation failed - payment required")
            return False, "API quota exceeded or payment required"
        else:
            VLM_ENABLED = False
            VLM_ERROR_COUNT += 1
            logger.warning(f"⚠️ API key validation returned status {response.status_code}")
            return False, f"API returned status {response.status_code}"
            
    except requests.exceptions.Timeout:
        VLM_ERROR_COUNT += 1
        logger.error("❌ API key validation timed out")
        return False, "API request timed out"
    except Exception as e:
        VLM_ERROR_COUNT += 1
        logger.error(f"❌ API key validation error: {str(e)}")
        return False, f"API validation error: {str(e)}"

async def get_vlm_description(image_data: bytes) -> Dict[str, Any]:
    """Get image description from DeepSeek R1 via OpenRouter API"""
    global VLM_ENABLED, VLM_ERROR_COUNT
    
    # Check if VLM is enabled
    if not VLM_ENABLED or VLM_ERROR_COUNT >= MAX_VLM_ERRORS:
        # Try to re-validate API key if it's been disabled
        if VLM_ERROR_COUNT < MAX_VLM_ERRORS:
            logger.info("VLM was disabled, attempting to re-validate API key...")
            api_valid, message = await test_api_key_validity()
            if not api_valid:
                return {"success": False, "error": f"Image analysis unavailable: {message}"}
        else:
            return {"success": False, "error": "Image analysis temporarily disabled due to repeated errors"}
    
    try:
        # Validate and process image
        try:
            img = Image.open(io.BytesIO(image_data))
            img_format = img.format.lower() if img.format else 'jpeg'
            
            # Convert to supported format if needed
            if img_format not in ['jpeg', 'png', 'jpg']:
                img_format = 'jpeg'
                output = io.BytesIO()
                img.convert("RGB").save(output, format="JPEG", quality=85)
                image_data = output.getvalue()
                logger.info("Converted image to JPEG format")
            
            # Check image size
            if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
                logger.warning("Image too large, compressing...")
                output = io.BytesIO()
                img.convert("RGB").save(output, format="JPEG", quality=70)
                image_data = output.getvalue()
                
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
            return {"success": False, "error": f"Image processing failed: {str(e)}"}
        
        # Prepare API request
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_uri = f"data:image/{img_format};base64,{base64_image}"

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Cat Breed Classifier",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "deepseek/deepseek-r1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": """Please describe this cat image in a concise paragraph, covering:

- Size and build
- Face and coat characteristics
- Any distinctive features

Keep it clear and brief without markdown formatting."""

                        },
                        {"type": "image_url", "image_url": {"url": image_uri}}
                    ]
                }
            ],
            "max_tokens": 500,
            "temperature": 0.7
        }

        logger.info("Sending request to DeepSeek API...")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        # Handle different response codes
        if response.status_code == 200:
            response_data = response.json()
            raw_description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            logger.info(f"API response received: {len(raw_description)} characters")
            logger.debug(f"Raw response: {raw_description[:200]}...")
            
            # Clean the description
            clean_desc = clean_description(raw_description) if raw_description else ""
            
            # Check if description is adequate
            if not clean_desc or len(clean_desc.strip()) < 20:
                logger.warning(f"Short description received. Raw: {len(raw_description)}, Clean: {len(clean_desc)}")
                # Don't use the generic fallback immediately - return what we have
                if clean_desc:
                    return {"success": True, "description": clean_desc}
                else:
                    return {"success": False, "error": "No description generated by the AI model"}
            
            logger.info(f"Description successfully generated: {len(clean_desc)} characters")
            return {"success": True, "description": clean_desc}
            
        elif response.status_code == 401:
            VLM_ENABLED = False
            VLM_ERROR_COUNT += 1
            logger.error("Authentication failed - API key invalid")
            return {"success": False, "error": "Authentication failed - API key appears to be invalid"}
            
        elif response.status_code == 429:
            VLM_ERROR_COUNT += 1
            logger.error("Rate limit exceeded")
            return {"success": False, "error": "Rate limit exceeded - please try again later"}
            
        elif response.status_code == 402:
            VLM_ENABLED = False
            VLM_ERROR_COUNT += 1
            logger.error("Payment required - quota exceeded")
            return {"success": False, "error": "API quota exceeded - payment required"}
            
        else:
            VLM_ERROR_COUNT += 1
            logger.error(f"API returned status {response.status_code}: {response.text}")
            return {"success": False, "error": f"API error: {response.status_code}"}

    except requests.exceptions.Timeout:
        VLM_ERROR_COUNT += 1
        logger.error("API request timed out")
        return {"success": False, "error": "Request timed out - please try again"}
        
    except requests.exceptions.RequestException as e:
        VLM_ERROR_COUNT += 1
        logger.error(f"Request error: {str(e)}")
        return {"success": False, "error": f"Network error: {str(e)}"}
        
    except Exception as e:
        VLM_ERROR_COUNT += 1
        logger.error(f"Unexpected error in VLM processing: {str(e)}")
        return {"success": False, "error": f"Unexpected error: {str(e)}"}

# Routes
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Serve the main application page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/classify-cat")
async def classify_cat(file: UploadFile = File(...)):
    """Main classification endpoint - determines if image is a cat and classifies breed"""
    try:
        # Validate models are loaded
        if cat_detector is None or breed_classifier is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Classification models not loaded. Please check server logs."}
            )
        
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Please upload an image."}
            )
        
        # Read and process image
        image_data = await file.read()
        if len(image_data) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file uploaded."}
            )
        
        logger.info(f"Processing image: {file.filename}, size: {len(image_data)} bytes")
        
        try:
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"error": f"Unable to process image: {str(e)}"}
            )
        
        # Get VLM description (async)
        vlm_task = asyncio.create_task(get_vlm_description(image_data))
        
        # Cat detection
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cat_output = cat_detector(input_tensor)
            cat_pred = torch.argmax(cat_output, dim=1).item()
            cat_confidence = torch.softmax(cat_output, dim=1)[0][cat_pred].item()
        
        # Determine if it's a cat (adjust based on your model's class mapping)
        is_cat = cat_pred == 0  # You may need to adjust this based on your training
        
        # Wait for VLM description
        vlm_result = await vlm_task
        
        if not is_cat:
            logger.info(f"Image classified as NOT a cat (confidence: {cat_confidence:.2f})")
            return {
                "is_cat": False,
                "message": "This doesn't appear to be a cat image",
                "cat_confidence": round(cat_confidence * 100, 2),
                "vlm_description": vlm_result
            }
        
        logger.info(f"Image classified as cat (confidence: {cat_confidence:.2f})")
        
        # Breed classification
        breed_input = breed_transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            breed_output = breed_classifier(breed_input)
            breed_probabilities = torch.softmax(breed_output, dim=1)[0]
            
            # Get top 3 predictions
            top3_indices = torch.topk(breed_probabilities, 3).indices
            top3_probs = torch.topk(breed_probabilities, 3).values
            
            predictions = []
            for i in range(3):
                breed_idx = top3_indices[i].item()
                confidence = top3_probs[i].item()
                breed_name = CAT_BREEDS[breed_idx]
                predictions.append({
                    "breed": breed_name,
                    "confidence": round(confidence * 100, 2),
                    "explanation": BREED_EXPLANATIONS.get(breed_name, "No explanation available")
                })
        
        logger.info(f"Top breed prediction: {predictions[0]['breed']} ({predictions[0]['confidence']:.1f}%)")
        
        return {
            "is_cat": True,
            "cat_confidence": round(cat_confidence * 100, 2),
            "breed_predictions": predictions,
            "top_breed": predictions[0]["breed"],
            "top_confidence": predictions[0]["confidence"],
            "top_breed_explanation": BREED_EXPLANATIONS.get(predictions[0]["breed"], "No explanation available"),
            "vlm_description": vlm_result
        }
        
    except Exception as e:
        logger.error(f"Classification error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Classification failed: {str(e)}",
                "vlm_description": {"success": False, "error": "Not processed due to classification error"}
            }
        )

@app.post("/vlm-describe")
async def vlm_describe(file: UploadFile = File(...)):
    """Dedicated endpoint for VLM description only"""
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Please upload an image."}
            )
        
        image_data = await file.read()
        if len(image_data) == 0:
            return JSONResponse(
                status_code=400,
                content={"error": "Empty file uploaded."}
            )
        
        vlm_result = await get_vlm_description(image_data)
        return vlm_result
        
    except Exception as e:
        logger.error(f"VLM description error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/is-cat")
async def is_cat(file: UploadFile = File(...)):
    """Simple cat detection endpoint"""
    try:
        if cat_detector is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Cat detection model not loaded"}
            )
        
        if not file.content_type or not file.content_type.startswith('image/'):
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid file type. Please upload an image."}
            )
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = cat_detector(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()
        
        # Adjust based on your class mapping
        if pred == 0:
            return {"result": "cat", "confidence": round(confidence * 100, 2)}
        else:
            return {"result": "not a cat", "confidence": round(confidence * 100, 2)}
    
    except Exception as e:
        logger.error(f"Cat detection error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/breeds")
async def get_supported_breeds():
    """Return list of supported cat breeds with explanations"""
    breeds_with_explanations = [
        {"breed": breed, "explanation": BREED_EXPLANATIONS.get(breed, "No explanation available")}
        for breed in CAT_BREEDS
    ]
    return {"breeds": breeds_with_explanations, "total_breeds": len(CAT_BREEDS)}

@app.get("/vlm-status")
async def get_vlm_status():
    """Return the status of VLM (Vision Language Model) functionality"""
    global VLM_ENABLED, VLM_LAST_CHECK, VLM_ERROR_COUNT
    
    status_message = "Image analysis is available"
    reason = "API key is valid"
    
    if not VLM_ENABLED:
        status_message = "Image analysis is temporarily unavailable"
        if VLM_ERROR_COUNT >= MAX_VLM_ERRORS:
            reason = f"Too many errors ({VLM_ERROR_COUNT})"
        else:
            reason = "API key is invalid or expired"
    
    return {
        "vlm_enabled": VLM_ENABLED,
        "message": status_message,
        "reason": reason,
        "error_count": VLM_ERROR_COUNT,
        "last_check": VLM_LAST_CHECK.isoformat() if VLM_LAST_CHECK else None,
        "api_key_configured": bool(DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_API_KEY_HERE")
    }

@app.post("/vlm-reset")
async def reset_vlm_status():
    """Reset VLM status and re-test API key"""
    global VLM_ENABLED, VLM_ERROR_COUNT
    
    logger.info("Manual VLM reset requested")
    
    # Reset error count
    VLM_ERROR_COUNT = 0
    
    # Re-test API key
    api_valid, message = await test_api_key_validity()
    
    return {
        "success": True,
        "vlm_enabled": VLM_ENABLED,
        "message": "VLM status reset and re-tested",
        "api_test_result": "valid" if api_valid else "invalid",
        "api_test_message": message,
        "error_count_reset": True
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "cat_detector": cat_detector is not None,
            "breed_classifier": breed_classifier is not None
        },
        "vlm_enabled": VLM_ENABLED,
        "device": str(device)
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    logger.info("🚀 Starting Cat Breed Classifier v2.0...")
    
    # Check models
    if cat_detector is None or breed_classifier is None:
        logger.error("❌ Models not loaded - classification features will be unavailable")
    else:
        logger.info("✅ Classification models loaded successfully")
    
    # Test API key
    if DEEPSEEK_API_KEY and DEEPSEEK_API_KEY != "YOUR_API_KEY_HERE":
        api_valid, message = await test_api_key_validity()
        if api_valid:
            logger.info("✅ VLM features enabled - API key is valid")
        else:
            logger.warning(f"⚠️ VLM features disabled - {message}")
    else:
        logger.warning("⚠️ VLM features disabled - API key not configured")
    
    logger.info("🎉 Cat Breed Classifier startup complete!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    host = os.environ.get('HOST', '0.0.0.0')
    
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")