from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
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
from typing import Dict, Any
import logging
import re
import os
from dotenv import load_dotenv


# Configure logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# DeepSeek API configuration (via OpenRouter)
load_dotenv()
DEEPSEEK_API_KEY=os.getenv("api_key")
DEEPSEEK_API_URL = "https://openrouter.ai/api/v1/chat/completions"

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Oxford-IIIT Pet Dataset - Cat breeds only (12 breeds) with explanations
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

# Dictionary of breed explanations
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

# Load cat/not-cat model first
cat_detector = models.mobilenet_v2(weights=None)
cat_detector.classifier[1] = nn.Linear(cat_detector.last_channel, 2)
cat_detector.load_state_dict(torch.load("cat_notcat_model.pth", map_location=device))
cat_detector.eval().to(device)

# Load breed classifier model (EfficientNetB0)
breed_classifier = models.efficientnet_b0(weights=None)
breed_classifier.classifier[1] = nn.Linear(breed_classifier.classifier[1].in_features, len(CAT_BREEDS))
breed_classifier.load_state_dict(torch.load("best_efficientnet_b0.pth", map_location=device))
breed_classifier.eval().to(device)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# EfficientNet might need different preprocessing
breed_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def clean_description(text:str) ->  str:
    if not text:
        return text 
    text = re.sub(r'\*{2,}', '', text)  # Remove ** and more
    text = re.sub(r'#{2,}', '', text)   # Remove ## and more
    text = re.sub(r'\*+', '', text)     # Remove single or multiple *
    text = re.sub(r'#+', '', text)      # Remove single or multiple #

    # Clean up any remaining formatting patterns
    text = re.sub(r'\s*-\s*\*\*[^*]*\*\*', '', text)  # Remove - **text** patterns
    text = re.sub(r'\s*-\s*\*[^*]*\*', '', text)      # Remove - *text* patterns
    
    # Remove extra spaces and clean up
    text = re.sub(r'\s+', ' ', text)    # Replace multiple spaces with single space
    text = text.strip()                 # Remove leading/trailing whitespace
    
    # Remove any remaining lone dashes or formatting remnants
    text = re.sub(r'^\s*-\s*', '', text)  # Remove leading dashes
    text = re.sub(r'\s*-\s*$', '', text)  # Remove trailing dashes
    
    return text

# Global variable to track API key validity
VLM_ENABLED = True

async def test_api_key_validity():
    """Test if the API key is valid by making a small request"""
    global VLM_ENABLED
    
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
        VLM_ENABLED = False
        return False
    
    try:
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Cat Breed Classifier",
            "Content-Type": "application/json"
        }
        
        # Test with a simple text-only request
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 5
        }
        
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=10)
        
        if response.status_code == 401:
            logger.warning("API key is invalid or expired - disabling VLM features")
            VLM_ENABLED = False
            return False
        elif response.status_code == 200:
            logger.info("API key validated successfully")
            VLM_ENABLED = True
            return True
        else:
            logger.warning(f"API key test returned status {response.status_code} - disabling VLM features")
            VLM_ENABLED = False
            return False
            
    except Exception as e:
        logger.warning(f"API key validation failed: {e} - disabling VLM features")
        VLM_ENABLED = False
        return False

async def get_vlm_description(image_data: bytes) -> Dict[str, Any]:
    """Get image description from DeepSeek R1 via OpenRouter API"""
    global VLM_ENABLED
    
    # Check if VLM is enabled
    if not VLM_ENABLED:
        return {
            "success": False, 
            "error": "Image analysis is temporarily unavailable. The API key may be invalid or expired. Please contact the administrator."
        }
    
    try:
        # Check if API key is configured
        if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "YOUR_API_KEY_HERE":
            logger.warning("VLM API key not configured")
            VLM_ENABLED = False
            return {"success": False, "error": "VLM API key not configured. Please set your OpenRouter API key in the .env file."}
        
        # Detect image format
        img = Image.open(io.BytesIO(image_data))
        img_format = img.format.lower() if img.format else 'jpeg'
        if img_format not in ['jpeg', 'png']:
            img_format = 'jpeg'
            output = io.BytesIO()
            img.convert("RGB").save(output, format="JPEG", quality=85)
            image_data = output.getvalue()
        
        # Convert image to base64 and prepend data URI
        base64_image = base64.b64encode(image_data).decode('utf-8')
        image_uri = f"data:image/{img_format};base64,{base64_image}"

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Cat Breed Classifier",
            "Content-Type": "application/json"
        }

        # Try with a free model first for image analysis
        payload = {
            "model": "meta-llama/llama-3.2-11b-vision-instruct:free",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this cat in detail, including its appearance, fur pattern, colors, and any distinctive features. Please provide a clean, readable description without any formatting symbols."},
                        {"type": "image_url", "image_url": {"url": image_uri}}
                    ]
                }
            ],
            "max_tokens": 200
        }

        logger.info(f"Sending request to OpenRouter API")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 401:
            logger.error("Authentication failed - disabling VLM")
            VLM_ENABLED = False
            return {"success": False, "error": "Authentication failed. The API key appears to be invalid or expired."}
        elif response.status_code == 429:
            logger.error("Rate limit exceeded")
            return {"success": False, "error": "Rate limit exceeded. Please try again later."}
        elif response.status_code == 402:
            logger.error("Payment required - trying fallback")
            return {"success": False, "error": "API quota exceeded. Image analysis is temporarily unavailable."}
        
        response.raise_for_status()  # Raise an exception for other 4xx/5xx errors

        response_data = response.json()
        logger.info(f"API response received successfully")
        raw_description = response_data.get("choices", [{}])[0].get("message", {}).get("content", "No description provided")
        
        # Clean the description before returning
        clean_desc = clean_description(raw_description)
        
        return {"success": True, "description": clean_desc}

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json().get('error', {}).get('message', 'Unknown error')
                error_msg += f" - {error_detail}"
                # If authentication error, disable VLM
                if 'auth' in error_detail.lower() or 'credential' in error_detail.lower():
                    VLM_ENABLED = False
            except:
                error_msg += f" - Status: {e.response.status_code}"
                if e.response.status_code == 401:
                    VLM_ENABLED = False
        logger.error(error_msg)
        return {"success": False, "error": error_msg}
    except Exception as e:
        logger.error(f"VLM processing error: {str(e)}")
        return {"success": False, "error": f"VLM processing error: {str(e)}"}
@app.post("/classify-cat")
async def classify_cat(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Get VLM description
        vlm_result = await get_vlm_description(image_data)
        
        # First, check if it's a cat
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            cat_output = cat_detector(input_tensor)
            cat_pred = torch.argmax(cat_output, dim=1).item()
            cat_confidence = torch.softmax(cat_output, dim=1)[0][cat_pred].item()
        
        # Adjust based on your class mapping (0 or 1 for cat)
        is_cat = cat_pred == 0  # Adjust this based on your model
        
        if not is_cat:
            return {
                "is_cat": False,
                "message": "This doesn't appear to be a cat image",
                "cat_confidence": round(cat_confidence * 100, 2),
                "vlm_description": vlm_result
            }
        
        # If it's a cat, classify the breed
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
        vlm_result = {"success": False, "error": "VLM not processed due to classification error"}
        return JSONResponse(status_code=500, content={"error": str(e), "vlm_description": vlm_result})

@app.post("/vlm-describe")
async def vlm_describe(file: UploadFile = File(...)):
    """Dedicated endpoint for VLM description only"""
    try:
        image_data = await file.read()
        vlm_result = await get_vlm_description(image_data)
        return vlm_result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/is-cat")
async def is_cat(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = cat_detector(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][pred].item()
        
        if pred == 0:  # Adjust based on your class mapping
            return {"result": "cat", "confidence": round(confidence * 100, 2)}
        else:
            return {"result": "not a cat", "confidence": round(confidence * 100, 2)}
    
    except Exception as e:
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
    global VLM_ENABLED
    return {
        "vlm_enabled": VLM_ENABLED,
        "message": "Image analysis is available" if VLM_ENABLED else "Image analysis is temporarily unavailable",
        "reason": "API key is valid" if VLM_ENABLED else "API key is invalid or expired"
    }

@app.on_event("startup")
async def startup_event():
    """Initialize the application and test API key validity"""
    logger.info("Starting Cat Breed Classifier...")
    
    # Test API key validity
    api_valid = await test_api_key_validity()
    if api_valid:
        logger.info("✅ VLM features enabled - API key is valid")
    else:
        logger.warning("⚠️ VLM features disabled - API key is invalid or expired")
        logger.warning("   Breed classification will still work normally")
    
    logger.info("Cat Breed Classifier startup complete!")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run(app, host='0.0.0.0', port=port)
