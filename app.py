import base64
import os
from io import BytesIO
from datetime import datetime
import importlib.util

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from services.processor import AnimalProcessor

# Serve the React frontend from the livestockai-frontend directory
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'livestockai-frontend')
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='/')

# Explicit CORS allowlist for frontend origins in development and production
ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "https://livestock-ai.vercel.app",
    "https://livestock-ai-frontend.vercel.app",
    "https://livestock-ai-engine.onrender.com",
    "https://jjhboruto-livestock-ai-engine.hf.space",
    "https://jjhmadeboruto-livestock-ai-engine.hf.space",
}

# Apply flask-cors for normal request/response flow
CORS(app, origins=list(ALLOWED_ORIGINS))


# Ensure CORS headers are always present, even on error responses
@app.after_request
def _ensure_cors_headers(response):
    origin = request.headers.get("Origin")
    if origin in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
    # Always allow common methods and headers for API usage
    response.headers.setdefault("Access-Control-Allow-Methods", "GET,POST,OPTIONS,PUT,DELETE,PATCH")
    response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type,Authorization")
    return response

# Session storage for calibration
session_calibration = {}
scan_history = []


def _read_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """Decode raw image bytes into an OpenCV BGR image."""
    if not file_bytes:
        return None
    file_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(file_array, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    try:
        fallback_array = np.frombuffer(bytearray(file_bytes), dtype=np.uint8)
        return cv2.imdecode(fallback_array, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


def _encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image to a Base64 PNG string."""
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError('Could not encode annotated image.')
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')


def _assess_image_quality(image: np.ndarray) -> dict:
    """Assess image quality for livestock weight estimation."""
    height, width = image.shape[:2]
    aspect_ratio = width / height if height > 0 else 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    brightness_ok = 50 < mean_brightness < 200
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    focus_ok = laplacian_var > 100

    quality_score = 0.0
    issues = []

    if brightness_ok:
        quality_score += 0.4
    else:
        issues.append("Image too dark." if mean_brightness <= 50 else "Image too bright.")

    if focus_ok:
        quality_score += 0.3
    else:
        issues.append("Image appears blurry. Take a clearer photo.")

    if 0.5 < aspect_ratio < 2.0:
        quality_score += 0.3
    else:
        issues.append("Unusual aspect ratio. Take a side-profile photo.")

    return {
        'quality_score': float(round(quality_score, 2)),
        'brightness': float(round(mean_brightness, 1)),
        'focus_quality': float(round(laplacian_var, 1)),
        'aspect_ratio': float(round(aspect_ratio, 2)),
        'issues': issues,
        'is_good_quality': bool(quality_score >= 0.7)
    }


@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/')
def index():
    """Serve the React frontend app, or JSON status for API clients."""
    # Serve the React app if the frontend directory exists
    if os.path.isfile(os.path.join(FRONTEND_DIR, 'index.html')):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    return jsonify({
        'status': 'live',
        'message': 'LivestockAI service. Use /api/health or /api/estimate-weight'
    }), 200


@app.route('/<path:filename>')
def serve_frontend_static(filename):
    """Serve any static assets (JS, CSS, images) from the frontend directory."""
    # Don't intercept /api/* routes
    if filename.startswith('api/'):
        from flask import abort
        abort(404)
    filepath = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(FRONTEND_DIR, filename)
    # SPA fallback — return index.html for client-side routing
    if os.path.isfile(os.path.join(FRONTEND_DIR, 'index.html')):
        return send_from_directory(FRONTEND_DIR, 'index.html')
    from flask import abort
    abort(404)


@app.route('/api/debug-yolo', methods=['GET'])
def debug_yolo() -> Response:
    import os
    import sys
    import traceback
    
    current_file_path = os.path.abspath(__file__)
    services_dir = os.path.join(os.path.dirname(current_file_path), "services")
    root_dir = os.path.dirname(current_file_path)
    
    path_options = [
        os.path.join(root_dir, "models", "yolov8n.pt"),
        os.path.join(os.getcwd(), "models", "yolov8n.pt"),
        os.path.join(os.getcwd(), "yolov8n.pt"),
        "yolov8n.pt"
    ]
    
    path_existence = {path: os.path.exists(path) if not path.endswith(".pt") or os.path.isabs(path) else "unknown" for path in path_options}
    
    # Try finding it globally or scanning directory
    dir_contents = []
    try:
        dir_contents = os.listdir(os.getcwd())
        if os.path.exists("models"):
            dir_contents.append({"models": os.listdir("models")})
    except Exception as e:
        dir_contents = [f"error listing: {e}"]
        
    yolo_load_status = "Not attempted"
    yolo_error = None
    
    try:
        import torch
        # Apply the override
        original_load = torch.load
        def safe_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = safe_load
        
        from ultralytics import YOLO
        
        # Test loading
        selected_path = None
        for path in path_options:
            if isinstance(path_existence.get(path), bool) and path_existence.get(path):
                selected_path = path
                break
        
        if not selected_path:
            selected_path = "yolov8n.pt"
            
        model = YOLO(selected_path)
        yolo_load_status = f"Successfully loaded YOLO using {selected_path}!"
        
        # Test inference with a dummy image first
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        dummy_res = model(dummy_img, verbose=False)[0]
        yolo_load_status += " Tested dummy inference successfully."
        
        # Now try download and detect chicken
        import urllib.request
        temp_chicken_path = "/tmp/test_chicken.jpg"
        urllib.request.urlretrieve("https://images.unsplash.com/photo-1548550023-2bdb3c5beed7?w=640", temp_chicken_path)
        chicken_img = cv2.imread(temp_chicken_path)
        
        detection_results = []
        if chicken_img is not None:
            results = model(chicken_img, verbose=False, conf=0.10)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detection_results.append({
                    'class_id': cls_id,
                    'class_name': model.names[cls_id] if hasattr(model, 'names') and cls_id in model.names else 'unknown',
                    'confidence': conf,
                    'box': [x1, y1, x2, y2]
                })
        else:
            yolo_load_status += " Failed to read downloaded chicken image."
            
    except Exception as e:
        yolo_load_status = "Failed to load/run YOLO"
        yolo_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        
    return jsonify({
        'cwd': os.getcwd(),
        'sys_path': sys.path,
        'path_options': path_options,
        'path_existence': path_existence,
        'dir_contents': dir_contents,
        'yolo_load_status': yolo_load_status,
        'yolo_error': yolo_error,
        'detection_results': detection_results,
        'torch_version': torch.__version__ if 'torch' in sys.modules else None
    }), 200


@app.route('/api/debug-mediapipe', methods=['GET'])
def debug_mediapipe() -> Response:
    mp_spec = importlib.util.find_spec('mediapipe')
    python_spec = importlib.util.find_spec('mediapipe.python')
    return jsonify({
        'mediapipe_file': mp_spec.origin if mp_spec else None,
        'mediapipe_exists': mp_spec is not None,
        'mediapipe_python_exists': python_spec is not None,
        'mediapipe_python_origin': python_spec.origin if python_spec else None,
    }), 200


@app.route('/api/debug-gemini', methods=['GET'])
def debug_gemini() -> Response:
    """Test Gemini REST API connectivity, list models, and verify key validity directly."""
    import os, json, urllib.request, urllib.error, ssl, traceback

    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return jsonify({
            'gemini_key_set': False,
            'status': 'GEMINI_API_KEY is not set in Space Secrets.',
            'fix': 'Go to HF Space Settings -> Variables and Secrets -> add GEMINI_API_KEY'
        }), 200

    models_url = f'https://generativelanguage.googleapis.com/v1/models?key={api_key}'
    models_data = None
    models_error = None

    try:
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(models_url, context=ctx, timeout=20) as r:
            models_data = json.loads(r.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        models_error = f"HTTP_{e.code}: {e.read().decode('utf-8', errors='replace')[:400]}"
    except Exception as e:
        models_error = f"{type(e).__name__}: {str(e)}"

    # Text-only test (no image) - avoids 400 errors from strict image validators on newer models
    payload_text_only = {
        'contents': [{'parts': [{'text': 'Respond with only the word: ok'}]}],
        'generationConfig': {'maxOutputTokens': 5}
    }

    # Image test payload - for models that support vision
    pixel_jpg_b64 = (
        '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8U'
        'HRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgN'
        'DRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy'
        'MjL/wAARCAABAAEDASIAAhEBAxEB/8QAFgABAQEAAAAAAAAAAAAAAAAABgUE/8QAIhAAAgIB'
        'BAMAAAAAAAAAAAAAAQIDBAUREiExQf/EABQBAQAAAAAAAAAAAAAAAAAAAAD/xAAUEQEAAAAA'
        'AAAAAAAAAAAAAP/aAAwDAQACEQMRAD8Apuz1ma4rRWnVJJXjVVZiWYgAAckkn3oA/9k='
    )
    payload_vision = {
        'contents': [{'parts': [
            {'inlineData': {'mimeType': 'image/jpeg', 'data': pixel_jpg_b64}},
            {'text': 'Return only this JSON: {"test": "ok"}'}
        ]}],
        'generationConfig': {'maxOutputTokens': 50}
    }

    test_results = {}
    for model_name in ['gemini-2.5-flash', 'gemini-3.5-flash', 'gemini-3.1-flash-lite', 'gemini-2.0-flash']:
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}'
        for label, payload in [('text_test', payload_text_only), ('vision_test', payload_vision)]:
            key_name = f"{model_name}_{label}"
            try:
                body = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(url, data=body, method='POST')
                req.add_header('Content-Type', 'application/json')
                with urllib.request.urlopen(req, context=ctx, timeout=10) as r:
                    resp = json.loads(r.read().decode('utf-8'))
                text = resp.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                test_results[key_name] = {'status': 'SUCCESS', 'response': text}
            except urllib.error.HTTPError as e:
                err_body = e.read().decode('utf-8', errors='replace')
                test_results[key_name] = {'status': f'HTTP_ERROR_{e.code}', 'error': err_body}
                if e.code == 429:
                    break  # quota exhausted, stop testing this model
            except Exception as e:
                test_results[key_name] = {'status': 'EXCEPTION', 'error': f'{type(e).__name__}: {str(e)}'}

    return jsonify({
        'gemini_key_set': True,
        'key_prefix': api_key[:8] + '...',
        'models_list': models_data,
        'models_error': models_error,
        'test_results': test_results
    }), 200


@app.route('/api/debug-vision-models', methods=['GET'])
def debug_vision_models() -> Response:
    """Test all Gemini models with a real image from the disk to see their vision response/errors."""
    import os, json, urllib.request, urllib.error, ssl, traceback, base64
    
    api_key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not api_key:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500
        
    img_url = "https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?w=640" # public photo of a cow
    try:
        ctx_no_ssl = ssl._create_unverified_context()
        with urllib.request.urlopen(img_url, context=ctx_no_ssl, timeout=15) as r:
            img_bytes = r.read()
        img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        return jsonify({'error': f'Failed to download test image from {img_url}: {e}'}), 500
        
    payload = {
        'contents': [{
            'parts': [
                {'inlineData': {'mimeType': 'image/jpeg', 'data': img_b64}},
                {'text': 'Identify the animal in this image. Return a JSON object: {"animal": "<name>"}'}
            ]
        }],
        'generationConfig': {'maxOutputTokens': 50}
    }
    
    ctx = ssl.create_default_context()
    results = {}
    models = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-3.1-flash-lite", "gemini-2.0-flash"]
    
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        body = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=20) as r:
                resp = json.loads(r.read().decode('utf-8'))
                text = resp.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '').strip()
                results[model] = {'status': 'SUCCESS', 'response': text}
        except urllib.error.HTTPError as e:
            err_body = e.read().decode('utf-8', errors='replace')
            results[model] = {'status': f'HTTP_{e.code}', 'error': err_body}
        except Exception as e:
            results[model] = {'status': 'EXCEPTION', 'error': f'{type(e).__name__}: {str(e)}'}
            
    return jsonify({
        'img_url': img_url,
        'img_size': len(img_bytes),
        'results': results
    }), 200






@app.route('/api/debug-enrich', methods=['POST'])
def debug_enrich() -> Response:
    import os, json, urllib.request, urllib.error, ssl, traceback
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files['image']
    file.stream.seek(0)
    file_bytes = file.read()
    image = _read_image_from_bytes(file_bytes)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400
    
    # Replicate _gemini_enrich call with raw try-except block returning detail
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY not set"}), 500
        
    try:
        success, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            return jsonify({"error": "cv2 encode failed"}), 500
        img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        
        # Test calling gemini-2.5-flash with prompt
        prompt = "Look at this image. Identify the breed and color of the animal if any, else identify the object. Return JSON: {\"breed\": \"name\", \"color\": \"color\"}"
        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "maxOutputTokens": 200
            }
        }
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            raw_response = resp.read().decode("utf-8")
            data = json.loads(raw_response)
            
        return jsonify({
            "status": "success",
            "gemini_response": data,
            "raw_text": data["candidates"][0]["content"]["parts"][0]["text"].strip()
        }), 200
        
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        return jsonify({
            "status": "http_error",
            "code": e.code,
            "error_body": err_body,
            "traceback": traceback.format_exc()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "exception",
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 200


@app.route('/api/debug-processor-gate-and-enrich', methods=['POST'])
def debug_processor_gate_and_enrich() -> Response:
    """Call _gemini_gate and _gemini_enrich directly, capturing raw responses."""
    import traceback, base64, cv2, os
    if 'image' not in request.files:
        return jsonify({'error': 'No image in request.files'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img_np = np.frombuffer(img_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return jsonify({'error': 'Invalid image'}), 400

    processor = AnimalProcessor()
    
    # 1. Test gate prompt
    gate_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not gate_api_key:
        return jsonify({'error': 'GEMINI_API_KEY not set'}), 500

    gate_payload = None
    gate_raw_text = None
    gate_full_response = None
    gate_error = None
    try:
        success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if success:
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            prompt = (
                "You are a livestock detection gatekeeper for a farming application.\n"
                "Look at this image and determine if it contains a livestock animal.\n\n"
                "Supported livestock species: cattle (cow/bull/calf), pig, goat, sheep, donkey, poultry (chicken/duck/turkey).\n\n"
                "Return ONLY a valid JSON object with exactly these keys:\n"
                "{\n"
                "  \"is_livestock\": <true or false>,\n"
                "  \"detected_species\": \"<what you actually see, e.g. pig, cow, dog, cat, mouse, phone, person, website, gadget>\",\n"
                "  \"confidence\": \"<high, medium, or low>\",\n"
                "  \"rejection_reason\": \"<if is_livestock is false: one clear sentence explaining what is in the image and why it cannot be weighed>\"\n"
                "}\n\n"
                "Be STRICT: a mouse, rat, dog, cat, wild animal, person, screenshot, website, or any non-livestock image must return is_livestock: false.\n"
                "Return ONLY the JSON object. No other text."
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 150
                }
            }
            gate_payload = payload
            resp = processor._call_gemini_api(payload, timeout=20)
            gate_full_response = resp
            if resp:
                gate_raw_text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                gate_error = "No response from _call_gemini_api"
    except Exception as e:
        gate_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    # 2. Test enrich prompt
    enrich_payload = None
    enrich_raw_text = None
    enrich_full_response = None
    enrich_error = None
    try:
        success, buf = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
            prompt = (
                f"You are a professional livestock veterinarian and expert body-condition scorer.\n"
                f"The farmer has submitted a photo they believe shows a dairy_cow.\n"
                f"Analyze this livestock image carefully and return ONLY a valid JSON object with these exact keys:\n\n"
                "{\n"
                "  \"breed\": \"<specific breed name, e.g. Large Black Pig, Duroc, Berkshire, Holstein, Boer Goat, or Mixed/Unknown>\",\n"
                "  \"sex\": \"<male, female, or unknown>\",\n"
                "  \"estimated_age_months\": <integer — your best estimate of the animal's age in months>,\n"
                "  \"body_condition\": \"<exactly one of: thin, fair, good, excellent>\",\n"
                "  \"body_condition_score\": <integer 1 to 5>,\n"
                "  \"posture\": \"<e.g. standing alert, grazing, resting, walking>\",\n"
                "  \"activity_level\": \"<calm, moderate, or active>\",\n"
                "  \"visible_health_concerns\": \"<any visible issues like wounds, lameness, distended belly, mud coverage, or None observed>\",\n"
                "  \"estimated_weight_range_kg\": [<min_integer>, <max_integer>],\n"
                "  \"photo_quality_note\": \"<one short sentence on photo quality and positioning>\",\n"
                "  \"gemini_explanation\": \"<A warm, clear, 3-5 sentence explanation written directly TO the farmer, in first person as Gemini. Start with what you visually observed about the animal. Explain WHY you estimated the weight you did — mentioning the breed, body condition, age, and any visual cues like belly size or muscle development. Mention what would make the estimate more accurate. End with a helpful farming insight.>\"\n"
                "}\n\n"
                "Body condition scoring guide:\n"
                "  1 = Emaciated (thin)\n"
                "  2 = Very thin (thin)\n"
                "  3 = Moderate (fair)\n"
                "  4 = Good (good)\n"
                "  5 = Excellent/Obese (excellent)\n\n"
                "Weight estimation guide:\n"
                "  - Adult pigs typically weigh 80–250 kg. A large mature sow can be 180–300 kg.\n"
                "  - Piglets (under 3 months) weigh 5–30 kg.\n"
                "  - Cattle: dairy cows 400–700 kg, beef bulls 500–800 kg, calves 50–200 kg.\n"
                "  - Goats: 20–90 kg. Sheep: 30–120 kg. Donkeys: 80–300 kg. Poultry: 0.5–8 kg.\n"
                "  - NEVER estimate above biologically plausible maximums.\n\n"
                "Be specific about the breed. Use your visual assessment of body size, muscle definition, and fat coverage.\n"
                "Return ONLY the JSON object, no other text."
            )
            payload = {
                "contents": [{
                    "parts": [
                        {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                        {"text": prompt}
                    ]
                }],
                "generationConfig": {
                    "temperature": 0.15,
                    "maxOutputTokens": 600
                }
            }
            enrich_payload = payload
            resp = processor._call_gemini_api(payload, timeout=30)
            enrich_full_response = resp
            if resp:
                enrich_raw_text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                enrich_error = "No response from _call_gemini_api"
    except Exception as e:
        enrich_error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    return jsonify({
        'gate_error': gate_error,
        'gate_raw_text': gate_raw_text,
        'gate_full_response': gate_full_response,
        'enrich_error': enrich_error,
        'enrich_raw_text': enrich_raw_text,
        'enrich_full_response': enrich_full_response
    }), 200


@app.route('/api/estimate-weight', methods=['POST', 'OPTIONS'])
def estimate_weight() -> Response:
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
        
    """Handle POST image uploads and return livestock weight estimation.

    Query parameters:
        - animal_type: Type of animal ('dairy_cow', 'beef_cattle', 'young_cattle', 'goat', 'sheep', 'donkey', 'pig', 'poultry')
        - session_id: Optional session ID for tracking calibration

    Form fields (optional):
        - pixel_ratio: direct cm/pixel conversion for this image
        - reference_cm: known size in centimeters of an object in the photo
        - reference_pixels: measured size in pixels of that reference object
        - animal_name: Name of the animal being scanned
        - farm_name: Name of the farm
    """
    animal_type = request.args.get('animal_type', 'dairy_cow')
    SUPPORTED_TYPES = list(AnimalProcessor.LIVESTOCK_CALIBRATION.keys())
    if animal_type not in SUPPORTED_TYPES:
        return jsonify({
            'status': 'error',
            'message': f"Animal type '{animal_type}' is not supported. Supported types: {SUPPORTED_TYPES}",
            'error_type': 'unsupported_animal_type',
            'supported_types': SUPPORTED_TYPES
        }), 400

    session_id = request.args.get('session_id', 'default')
    pixel_ratio = request.form.get('pixel_ratio')
    reference_cm = request.form.get('reference_cm')
    reference_pixels = request.form.get('reference_pixels')
    animal_name = request.form.get('animal_name', 'Unknown')
    farm_name = request.form.get('farm_name', 'Unknown Farm')

    # ── Build processor (ALL animal types go through AnimalProcessor / MediaPipe) ──
    try:
        processor = AnimalProcessor(animal_type=animal_type)
    except (ValueError, ImportError) as e:
        return jsonify({'status': 'error', 'error': str(e), 'error_type': 'processor_init_failed'}), 500

    original_pixel_ratio = processor.pixel_to_cm_ratio

    try:
        # Apply session calibration
        if session_id in session_calibration:
            processor.pixel_to_cm_ratio = session_calibration[session_id]
            processor._pixel_ratio_user_set = True

        # Apply per-request pixel_ratio override
        if pixel_ratio:
            try:
                val = float(pixel_ratio)
                if val > 0:
                    processor.pixel_to_cm_ratio = val
                    processor._pixel_ratio_user_set = True
                    session_calibration[session_id] = val
            except (ValueError, TypeError):
                pass

        # Apply reference-object calibration
        if reference_cm and reference_pixels:
            try:
                ref_cm_val = float(reference_cm)
                ref_px_val = float(reference_pixels)
                if ref_cm_val > 0 and ref_px_val > 0:
                    processor.calibrate_pixel_ratio(ref_cm_val, ref_px_val)
                    processor._pixel_ratio_user_set = True
                    session_calibration[session_id] = processor.pixel_to_cm_ratio
            except (ValueError, TypeError):
                pass

        # Read image
        if 'image' not in request.files or request.files['image'].filename == '':
            return jsonify({'status': 'error', 'message': 'No image provided', 'error_type': 'no_image'}), 400

        image_file = request.files['image']
        image_file.stream.seek(0)
        file_bytes = image_file.stream.read()

        if not file_bytes:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image data. Please upload a valid JPEG or PNG.',
                'error_type': 'invalid_image',
                'file_size': 0
            }), 400

        image = _read_image_from_bytes(file_bytes)
        if image is None:
            return jsonify({
                'status': 'error',
                'message': 'Invalid image data. Please upload a valid JPEG or PNG.',
                'error_type': 'invalid_image',
                'file_size': len(file_bytes),
                'signature': list(file_bytes[:8])
            }), 400

        quality_info = _assess_image_quality(image)
        result = processor.process(image)

    finally:
        processor.pixel_to_cm_ratio = original_pixel_ratio

    if result is None or result.get('method') == 'rejected':
        error_msg = result.get('error_message') if result else 'Could not detect animal pose or bounding box.'
        return jsonify({
            'status': 'error',
            'success': False,
            'message': error_msg,
            'error_type': 'detection_failed',
            'guidance': [
                "Ensure the animal is standing in a clear side profile.",
                "Make sure the full body from shoulder to heel is visible.",
                "Avoid extreme angles or partially visible animals.",
                "Try taking a new photo with better lighting."
            ],
            'image_quality': quality_info
        }), 422

    annotated_b64 = _encode_image_to_base64(result['annotated_image'])
    method_used = result.get('method', 'unknown')

    guidance = [
        f"Animal identified as {result['animal_type']} (using {method_used.upper()}).",
        f"Confidence: {round(result['confidence_score'] * 100, 1)}% - {'High' if result['confidence_score'] > 0.8 else 'Moderate'}",
        "For best accuracy, take a side-profile photo of the animal.",
        "Include a reference object (ruler/tape) in future photos."
    ]
    if not result.get('within_expected_range', True):
        exp_range = result.get('expected_weight_range')
        if exp_range:
            guidance.append(
                f"Estimated weight {result['weight']} kg is outside the expected range "
                f"for {result['animal_type']} ({exp_range[0]}–{exp_range[1]} kg)."
            )
        guidance.append("Check that the selected animal type matches the photo and review calibration.")

    # Use the DETECTED species key (not the user-button label) so that
    # a pig scanned on the 'Cattle' button records under 'pig', not 'cattle'.
    detected_species_key = result.get('detected_species_key', animal_type)
    detected_display_name = result['animal_type']  # human-readable (e.g. "Pig")

    scan_record = {
        'timestamp': datetime.now().isoformat(),
        'animal_name': animal_name,
        'animal_type': detected_display_name,          # display name for UI
        'animal_type_key': detected_species_key,       # internal key for filtering
        'requested_animal_type': animal_type,          # what the user selected
        'farm_name': farm_name,
        'weight': result['weight'],
        'body_length': result['body_length'],
        'body_height': result['body_height'],
        'confidence_score': result['confidence_score'],
        'breed': result.get('breed', 'Unknown'),
        'sex': result.get('sex', 'Unknown'),
        'estimated_age_months': result.get('estimated_age_months'),
        'body_condition': result.get('body_condition', 'Not assessed'),
        'within_expected_range': result.get('within_expected_range', True),
        'method': method_used
    }
    scan_history.append(scan_record)

    return jsonify({
        # ── Core weight result ────────────────────────────────────────────────
        'status': 'success',
        'success': True,
        'weight_kg': result['weight'],           # primary field
        'weight': result['weight'],              # legacy alias
        'body_length_cm': result['body_length'], # primary field
        'body_length': result['body_length'],    # legacy alias
        'body_height_cm': result['body_height'], # primary field
        'body_height': result['body_height'],    # legacy alias
        'estimated_girth': result['estimated_girth'],
        # ── Species info — detected (corrected) vs. requested ────────────
        'animal_type': result['animal_type'],              # display name of DETECTED species
        'detected_species_key': detected_species_key,      # internal key of DETECTED species
        'requested_animal_type': animal_type,              # what the user originally selected
        'species_corrected': detected_species_key != animal_type,  # true if AI corrected the species
        'confidence_score': result['confidence_score'],
        'pixel_to_cm_ratio': result.get('pixel_to_cm_ratio', processor.pixel_to_cm_ratio),
        'image_quality': quality_info,
        'expected_weight_range': result.get('expected_weight_range'),
        'within_expected_range': result.get('within_expected_range'),
        'guidance': guidance,
        'annotated_image': annotated_b64,
        'filename': image_file.filename,
        'method': method_used,
        # ── Gemini enrichment — breed, body condition, detailed attributes ─
        'breed': result.get('breed', 'Unknown'),
        'sex': result.get('sex', 'Unknown'),
        'estimated_age_months': result.get('estimated_age_months'),
        'posture': result.get('posture', ''),
        'activity_level': result.get('activity_level', ''),
        'visible_health_concerns': result.get('visible_health_concerns', 'None observed'),
        'body_condition': result.get('body_condition', 'Not assessed'),
        'body_condition_score': result.get('body_condition_score'),
        'health_notes': result.get('health_notes', ''),
        'photo_quality_note': result.get('photo_quality_note', ''),
        'gemini_cross_check': result.get('gemini_cross_check'),
        # ── Gemini conversational explanation (show this to the user) ─────
        'gemini_explanation': result.get('gemini_explanation', ''),
        # ── AI attribution (displayed in frontend as toast) ───────────────
        'ai_attribution': result.get('ai_attribution', {
            'detection_model': 'YOLOv8n (Ultralytics)',
            'classification_model': 'CLIP ViT-B/32 (OpenAI)',
            'enrichment_model': None,
            'weight_formula': 'Schoorl Girth Formula + Gemini Visual Intelligence',
        }),
        # ── Photo guidance (displayed as toast to help users get better shots)
        'photo_guidance': result.get('photo_guidance', {}),
    }), 200


@app.route('/api/live-cam/status', methods=['GET'])
def live_cam_status() -> Response:
    """Quick heartbeat for the live cam feature."""
    import os
    gemini_configured = bool(os.environ.get('GEMINI_API_KEY', '').strip())
    return jsonify({
        'success': True,
        'live_cam_available': True,
        'gemini_active': gemini_configured,
        'self_critique_enabled': gemini_configured,
        'supported_animal_types': list(AnimalProcessor.LIVESTOCK_CALIBRATION.keys()),
        'message': (
            'Live cam is ready. Send base64 JPEG frames to /api/live-cam/frame'
            if gemini_configured
            else 'Live cam works but Gemini enrichment is not configured (GEMINI_API_KEY missing).'
        ),
        'tips': [
            'Hold camera 2–3 metres from the animal.',
            'Ensure the full side profile is visible.',
            'Good lighting significantly improves accuracy.',
            'Keep the camera steady — motion blur reduces detection quality.',
        ]
    }), 200


@app.route('/api/live-cam/frame', methods=['POST', 'OPTIONS'])
def live_cam_frame() -> Response:
    """
    Process a single live camera frame for livestock weight estimation.

    Accepts JSON body:
    {
      "image_data": "<base64-encoded JPEG or PNG string>",
      "animal_type": "pig",                 (optional, default: dairy_cow)
      "animal_name": "Bessie",              (optional)
      "farm_name": "Green Acres",           (optional)
      "session_id": "user-xyz",             (optional, for history)
      "run_self_critique": true,            (optional, default true — Gemini validates its own answer)
      "save_to_history": false              (optional, default false for live frames)
    }

    Returns the full enrichment response including:
    - weight_kg, breed, sex, estimated_age_months
    - gemini_explanation  (Gemini speaking directly to the farmer)
    - frame_analysis      (step-by-step AI reasoning chain)
    - was_adjusted        (did the self-critique change the weight?)
    - critique_notes      (what Gemini reconsidered)
    - confidence_level    (high / medium / low)
    - visible_health_concerns
    - annotated_image_b64 (frame with bounding box drawn)
    """
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200

    data = request.get_json(silent=True)
    if not data:
        # Also support multipart/form-data (for compatibility)
        image_b64   = request.form.get('image_data', '')
        animal_type = request.args.get('animal_type', request.form.get('animal_type', 'dairy_cow'))
        animal_name = request.form.get('animal_name', 'Live Scan')
        farm_name   = request.form.get('farm_name', 'Unknown Farm')
        session_id  = request.form.get('session_id', 'live_default')
        run_critique = request.form.get('run_self_critique', 'true').lower() == 'true'
        save_history = request.form.get('save_to_history', 'false').lower() == 'true'
    else:
        image_b64   = data.get('image_data', '')
        animal_type = data.get('animal_type', 'dairy_cow')
        animal_name = data.get('animal_name', 'Live Scan')
        farm_name   = data.get('farm_name', 'Unknown Farm')
        session_id  = data.get('session_id', 'live_default')
        run_critique = data.get('run_self_critique', True)
        save_history = data.get('save_to_history', False)

    # ── Validate image data ────────────────────────────────────────────────
    if not image_b64:
        return jsonify({
            'status': 'error',
            'success': False,
            'message': 'No image_data provided. Send a base64-encoded JPEG or PNG string.',
            'error_type': 'no_image',
        }), 400

    # Strip data URL prefix if present (e.g. "data:image/jpeg;base64,...")
    if ',' in image_b64:
        image_b64 = image_b64.split(',', 1)[1]

    try:
        file_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({
            'status': 'error',
            'success': False,
            'message': 'Invalid base64 image data.',
            'error_type': 'invalid_base64',
        }), 400

    image = _read_image_from_bytes(file_bytes)
    if image is None:
        return jsonify({
            'status': 'error',
            'success': False,
            'message': 'Could not decode image. Ensure the frame is a valid JPEG or PNG.',
            'error_type': 'invalid_image',
        }), 400

    # ── Validate animal type ──────────────────────────────────────────────
    SUPPORTED = list(AnimalProcessor.LIVESTOCK_CALIBRATION.keys())
    if animal_type not in SUPPORTED:
        animal_type = 'dairy_cow'

    # ── Build processor ───────────────────────────────────────────────────
    try:
        processor = AnimalProcessor(animal_type=animal_type)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'success': False,
            'message': str(e),
            'error_type': 'processor_init_failed',
        }), 500

    # Apply any session calibration
    if session_id in session_calibration:
        processor.pixel_to_cm_ratio = session_calibration[session_id]
        processor._pixel_ratio_user_set = True

    # ── Image quality pre-check ───────────────────────────────────────────
    quality_info = _assess_image_quality(image)

    # ── Run full live-cam pipeline (with self-critique) ───────────────────
    try:
        result = processor.process_live_frame(image, run_self_critique=bool(run_critique))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'success': False,
            'message': f'Processing error: {str(e)}',
            'error_type': 'processing_failed',
        }), 500

    # ── Handle rejection / no detection ──────────────────────────────────
    if result is None or result.get('method') == 'rejected':
        error_msg = result.get('error_message') if result else 'No animal detected in frame.'
        detected_as = result.get('detected_as', '') if result else ''
        return jsonify({
            'status': 'no_detection',
            'success': False,
            'message': error_msg,
            'detected_as': detected_as,
            'error_type': 'detection_failed',
            'image_quality': quality_info,
            'live_cam_tip': (
                'Point the camera at a livestock animal standing in clear side profile. '
                'Ensure good lighting and hold the camera steady.'
            ),
        }), 200  # 200 so the frontend doesn't treat it as a crash

    # ── Encode annotated frame ────────────────────────────────────────────
    try:
        annotated_b64 = _encode_image_to_base64(result['annotated_image'])
    except Exception:
        annotated_b64 = ''

    method_used = result.get('method', 'unknown')
    detected_species_key  = result.get('detected_species_key', animal_type)
    detected_display_name = result.get('animal_type', 'Unknown')

    # ── Optionally save to history ────────────────────────────────────────
    if save_history:
        scan_record = {
            'timestamp': datetime.now().isoformat(),
            'animal_name': animal_name,
            'animal_type': detected_display_name,
            'animal_type_key': detected_species_key,
            'requested_animal_type': animal_type,
            'farm_name': farm_name,
            'weight': result['weight'],
            'body_length': result.get('body_length', 0),
            'body_height': result.get('body_height', 0),
            'confidence_score': result.get('confidence_score', 0),
            'breed': result.get('breed', 'Unknown'),
            'sex': result.get('sex', 'Unknown'),
            'estimated_age_months': result.get('estimated_age_months'),
            'body_condition': result.get('body_condition', 'Not assessed'),
            'within_expected_range': result.get('within_expected_range', True),
            'method': method_used,
            'source': 'live_cam',
            'was_adjusted': result.get('was_adjusted', False),
            'confidence_level': result.get('confidence_level', 'medium'),
        }
        scan_history.append(scan_record)

    return jsonify({
        'status': 'success',
        'success': True,
        'live_cam_mode': True,
        # ── Core weight (critique-confirmed) ─────────────────────────────
        'weight_kg': result['weight'],
        'weight': result['weight'],
        'proposed_weight_kg': result.get('proposed_weight_kg', result['weight']),
        'was_adjusted': result.get('was_adjusted', False),
        'adjustment_reason': result.get('adjustment_reason', ''),
        'confidence_level': result.get('confidence_level', 'medium'),
        'critique_notes': result.get('critique_notes', ''),
        # ── Dimensions ───────────────────────────────────────────────────
        'body_length_cm': result.get('body_length', 0),
        'body_height_cm': result.get('body_height', 0),
        'estimated_girth': result.get('estimated_girth', 0),
        # ── Species info ─────────────────────────────────────────────────
        'animal_type': detected_display_name,
        'detected_species_key': detected_species_key,
        'requested_animal_type': animal_type,
        'species_corrected': detected_species_key != animal_type,
        'confidence_score': result.get('confidence_score', 0),
        # ── Gemini enrichment attributes ─────────────────────────────────
        'breed': result.get('breed', 'Unknown'),
        'sex': result.get('sex', 'Unknown'),
        'estimated_age_months': result.get('estimated_age_months'),
        'posture': result.get('posture', ''),
        'activity_level': result.get('activity_level', ''),
        'visible_health_concerns': result.get('visible_health_concerns', 'None observed'),
        'body_condition': result.get('body_condition', 'Not assessed'),
        'body_condition_score': result.get('body_condition_score'),
        # ── Gemini conversational explanation ────────────────────────────
        'gemini_explanation': result.get('gemini_explanation', ''),
        'health_notes': result.get('health_notes', ''),
        'photo_quality_note': result.get('photo_quality_note', ''),
        # ── Step-by-step AI reasoning chain ─────────────────────────────
        'frame_analysis': result.get('frame_analysis', {}),
        # ── Range and quality metadata ───────────────────────────────────
        'expected_weight_range': result.get('expected_weight_range'),
        'within_expected_range': result.get('within_expected_range'),
        'gemini_cross_check': result.get('gemini_cross_check'),
        'image_quality': quality_info,
        'method': method_used,
        # ── Annotated frame with bounding box ────────────────────────────
        'annotated_image': annotated_b64,
        # ── AI attribution ────────────────────────────────────────────────
        'ai_attribution': result.get('ai_attribution', {}),
        'photo_guidance': result.get('photo_guidance', {}),
    }), 200


@app.route('/api/animal-types', methods=['GET'])
def get_animal_types() -> Response:
    """Get available animal types and their calibration info."""
    types = AnimalProcessor.get_available_types()
    return jsonify({'success': True, 'animal_types': types, 'count': len(types)}), 200


@app.route('/api/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint."""
    import os
    gemini_configured = bool(os.environ.get('GEMINI_API_KEY', '').strip())
    return jsonify({
        'status': 'healthy',
        'version': '3.4.0',
        'deploy_version': '2026-06-14-v7-ai-self-review',
        'service': 'LivestockAI Weight Estimation API',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'weight_estimation': True,
            'calibration': True,
            'species_detection_clip': True,
            'gemini_enrichment': gemini_configured,
            'gemini_gatekeeper': gemini_configured,
            'gemini_self_critique': gemini_configured,
            'bcs_weight_correction': gemini_configured,
            'breed_identification': gemini_configured,
            'sex_age_detection': gemini_configured,
            'health_assessment': gemini_configured,
            'live_cam': True,
            'hard_weight_caps': True,
            'photo_guidance': True,
            'ai_attribution': True,
            'session_tracking': True,
            'image_quality_assessment': True,
        },
        'ai_models': {
            'detection': 'YOLOv8n (Ultralytics)',
            'classification': 'CLIP ViT-B/32 (OpenAI)',
            'enrichment': 'Google Gemini 2.5 Flash' if gemini_configured else 'Not configured',
            'self_critique': 'Google Gemini 2.5 Flash (validates its own answers)' if gemini_configured else 'Not configured',
            'fallback_chain': 'gemini-2.5-flash → gemini-3.5-flash → gemini-3.1-flash-lite → gemini-2.0-flash',
        }
    }), 200



@app.route('/api/scan-history', methods=['GET'])
def get_scan_history() -> Response:
    animal_type_filter = request.args.get('animal_type')
    limit = int(request.args.get('limit', 100))
    filtered_scans = scan_history
    if animal_type_filter:
        filtered_scans = [s for s in scan_history if animal_type_filter.lower() in s['animal_type'].lower()]
    return jsonify({'success': True, 'total_scans': len(filtered_scans), 'scans': filtered_scans[-limit:]}), 200


@app.route('/api/scan-history', methods=['DELETE'])
def clear_scan_history() -> Response:
    global scan_history
    scan_history = []
    return jsonify({'success': True, 'message': 'Scan history cleared.'}), 200


@app.route('/api/session/calibration', methods=['GET'])
def get_session_calibration() -> Response:
    session_id = request.args.get('session_id', 'default')
    calibration_value = session_calibration.get(session_id)
    return jsonify({
        'success': True,
        'session_id': session_id,
        'pixel_to_cm_ratio': calibration_value,
        'is_calibrated': calibration_value is not None
    }), 200


@app.route('/api/session/calibration', methods=['POST'])
def set_session_calibration() -> Response:
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided.', 'error_type': 'no_json'}), 400
    session_id = data.get('session_id', 'default')
    pixel_ratio = data.get('pixel_to_cm_ratio')
    if pixel_ratio is None:
        return jsonify({'error': 'pixel_to_cm_ratio is required.', 'error_type': 'missing_ratio'}), 400
    try:
        session_calibration[session_id] = float(pixel_ratio)
        return jsonify({
            'success': True,
            'message': f'Calibration set for session {session_id}',
            'session_id': session_id,
            'pixel_to_cm_ratio': float(pixel_ratio)
        }), 200
    except (ValueError, TypeError):
        return jsonify({'error': 'pixel_to_cm_ratio must be a number.', 'error_type': 'invalid_ratio'}), 400


@app.route('/api/calibrate', methods=['POST'])
def calibrate() -> Response:
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided.', 'error_type': 'no_json'}), 400

    action = data.get('action')

    if action == 'pixel_ratio':
        known_cm = data.get('known_cm')
        measured_pixels = data.get('measured_pixels')
        if known_cm is None or measured_pixels is None:
            return jsonify({'error': 'Missing known_cm or measured_pixels.', 'error_type': 'missing_fields'}), 400
        try:
            processor = AnimalProcessor()
            processor.calibrate_pixel_ratio(float(known_cm), float(measured_pixels))
            return jsonify({
                'success': True,
                'message': f'Calibrated pixel ratio to {processor.pixel_to_cm_ratio:.6f} cm/pixel',
                'pixel_to_cm_ratio': processor.pixel_to_cm_ratio
            }), 200
        except (ValueError, ZeroDivisionError) as e:
            return jsonify({'error': str(e), 'error_type': 'calibration_failed'}), 400

    elif action == 'weight_formula':
        animal_type = data.get('animal_type', 'dairy_cow')
        divisor = data.get('divisor')
        girth_multiplier = data.get('girth_multiplier')
        if divisor is None and girth_multiplier is None:
            return jsonify({'error': 'Provide at least divisor or girth_multiplier.', 'error_type': 'missing_calibration'}), 400
        try:
            processor = AnimalProcessor()
            processor.adjust_weight_calibration(animal_type, divisor, girth_multiplier)
            return jsonify({
                'success': True,
                'message': f'Calibrated {animal_type} weight formula',
                'calibration': processor.LIVESTOCK_CALIBRATION[animal_type]
            }), 200
        except ValueError as e:
            return jsonify({'error': str(e), 'error_type': 'invalid_animal_type'}), 400

    else:
        return jsonify({'error': 'Unknown action. Use pixel_ratio or weight_formula.', 'error_type': 'invalid_action'}), 400


@app.route('/api/guidelines', methods=['GET'])
def get_guidelines() -> Response:
    """Returns comprehensive photo-taking guidelines and reference object specs.
    Frontend can display these as an onboarding toast or help modal."""
    from services.processor import AnimalProcessor
    return jsonify({
        'success': True,
        'guidelines': {
            'photo_steps': [
                "📐  Stand exactly 2–3 metres (6–10 feet) away from the animal.",
                "🐄  Capture a FULL SIDE PROFILE — head to tail, all four legs visible.",
                "📷  Hold the camera at the animal's mid-body height (not from above or below).",
                "☀️  Use good, even lighting — avoid harsh shadows across the body.",
                "🔲  Place an A4 sheet of paper (29.7 × 21 cm) flat on the ground beside the animal.",
                "✋  Keep the animal standing still — a moving animal blurs measurements.",
                "🚫  Remove other animals, people, or vehicles from the background.",
            ],
            'distance_guide': '2–3 metres (6–10 feet) gives the best balance of full-body visibility and pixel density.',
            'reference_objects': [
                {'name': 'A4 Paper (recommended)', 'width_cm': 21.0,  'height_cm': 29.7,  'note': 'Place flat on ground beside animal'},
                {'name': 'Credit Card',             'width_cm': 8.56,  'height_cm': 5.40,  'note': 'Tape to fence rail beside animal'},
                {'name': 'Standard Ruler',          'width_cm': 30.0,  'height_cm': 3.0,   'note': 'Hold vertically against animal shoulder'},
                {'name': 'Fence Rail (typical)',     'width_cm': None,  'height_cm': 120.0, 'note': 'Standard cattle fence rail is ~1.2 m high'},
            ],
            'accuracy_table': [
                {'method': 'No reference object, random distance', 'typical_error': '±20–30%'},
                {'method': 'Correct distance (2–3 m), no reference',  'typical_error': '±10–15%'},
                {'method': 'Reference object in frame',               'typical_error': '±5–10%'},
                {'method': 'Reference object + correct distance',     'typical_error': '±3–8%'},
            ],
            'calibration_tips': [
                "Include a known reference object (A4 paper, ruler, or credit card) in the photo.",
                "A4 paper is ideal — place it flat on the ground beside the animal.",
                "Measure the A4 paper's pixel width in your photo, then send reference_cm=21 and reference_pixels=<your measurement> with the scan request.",
                "Consistent distance (2–3 m) between shots improves trending accuracy over time.",
            ],
            'weight_caps': AnimalProcessor.WEIGHT_HARD_CAPS,
            'ai_pipeline': {
                'step1': 'YOLOv8n detects bounding box (WHERE the animal is)',
                'step2': 'CLIP ViT-B/32 classifies species (WHAT the animal is)',
                'step3': 'Schoorl Girth Formula converts pixels → estimated weight',
                'step4': 'Google Gemini 2.5 Flash enriches with breed, body condition score, and health notes',
                'step5': 'BCS correction factor adjusts weight (thin animals weigh less than their frame suggests)',
                'step6': 'Hard species cap ensures no biologically impossible weight is returned',
            }
        }
    }), 200


@app.errorhandler(404)
def handle_not_found(error):
    return jsonify({'success': False, 'error': 'not_found', 'message': 'The requested URL was not found.', 'path': request.path}), 404


@app.errorhandler(405)
def handle_method_not_allowed(error):
    return jsonify({'success': False, 'error': 'method_not_allowed', 'message': 'The requested method is not allowed for this URL.'}), 405


@app.errorhandler(500)
def handle_internal_error(error):
    return jsonify({'success': False, 'error': 'internal_server_error', 'message': 'An unexpected error occurred.'}), 500


@app.errorhandler(Exception)
def handle_unhandled_exception(error):
    import traceback
    traceback.print_exc()
    return jsonify({'success': False, 'error': 'unhandled_exception', 'message': str(error)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)