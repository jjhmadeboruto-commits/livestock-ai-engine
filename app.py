import base64
from io import BytesIO
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from processor import AnimalProcessor

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Session storage for calibration and scan history
session_calibration = {}
scan_history = []


def _read_image_from_file(file_storage) -> np.ndarray:
    """Read an uploaded file into an OpenCV BGR image."""
    file_stream = BytesIO(file_storage.read())
    file_bytes = np.frombuffer(file_stream.getvalue(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def _encode_image_to_base64(image: np.ndarray) -> str:
    """Encode a BGR image to a Base64 PNG string."""
    success, encoded_image = cv2.imencode('.png', image)
    if not success:
        raise ValueError('Could not encode annotated image.')

    return base64.b64encode(encoded_image.tobytes()).decode('utf-8')


def _assess_image_quality(image: np.ndarray) -> dict:
    """Assess image quality for livestock weight estimation.
    
    Returns quality metrics and guidance for improvement.
    """
    height, width = image.shape[:2]
    
    # Check image dimensions
    aspect_ratio = width / height if height > 0 else 0
    
    # Check brightness (avoid overexposed/underexposed)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    brightness_ok = 50 < mean_brightness < 200
    
    # Check contrast (Laplacian variance indicates focus quality)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    focus_ok = laplacian_var > 100
    
    quality_score = 0.0
    issues = []
    
    if brightness_ok:
        quality_score += 0.4
    else:
        if mean_brightness <= 50:
            issues.append("Image too dark. Increase lighting.")
        else:
            issues.append("Image too bright. Reduce glare.")
    
    if focus_ok:
        quality_score += 0.3
    else:
        issues.append("Image appears blurry. Take a clearer photo.")
    
    if 0.5 < aspect_ratio < 2.0:
        quality_score += 0.3
    else:
        issues.append("Unusual aspect ratio. Take a side-profile photo.")
    
    return {
        'quality_score': round(quality_score, 2),
        'brightness': round(mean_brightness, 1),
        'focus_quality': round(laplacian_var, 1),
        'aspect_ratio': round(aspect_ratio, 2),
        'issues': issues,
        'is_good_quality': quality_score >= 0.7
    }


@app.route('/health')
def health():
    return {"status": "ok"}, 200


@app.route('/api/estimate-weight', methods=['POST'])
def estimate_weight() -> Response:
    """Handle POST image uploads and return livestock weight estimation.
    
    Query parameters:
        - animal_type: Type of animal ('dairy_cow', 'beef_cattle', 'young_cattle', 'goat', 'sheep', 'donkey', 'poultry')
        - session_id: Optional session ID for tracking calibration

    Form fields (optional):
        - pixel_ratio: direct cm/pixel conversion for this image
        - reference_cm: known size in centimeters of an object in the photo
        - reference_pixels: measured size in pixels of that reference object
        - animal_name: Name of the animal being scanned
        - farm_name: Name of the farm
    """
    animal_type = request.args.get('animal_type', 'dairy_cow')
    session_id = request.args.get('session_id', 'default')
    pixel_ratio = request.form.get('pixel_ratio')
    reference_cm = request.form.get('reference_cm')
    reference_pixels = request.form.get('reference_pixels')
    animal_name = request.form.get('animal_name', 'Unknown')
    farm_name = request.form.get('farm_name', 'Unknown Farm')

    try:
        processor = AnimalProcessor(animal_type=animal_type)
    except ValueError as e:
        return jsonify({'error': str(e), 'error_type': 'invalid_animal_type'}), 400

    # Apply session calibration if available
    if session_id in session_calibration:
        processor.pixel_to_cm_ratio = session_calibration[session_id]

    if pixel_ratio:
        try:
            processor.pixel_to_cm_ratio = float(pixel_ratio)
            session_calibration[session_id] = float(pixel_ratio)
        except (ValueError, TypeError):
            return jsonify({'error': 'pixel_ratio must be a number.', 'error_type': 'invalid_pixel_ratio'}), 400

    if reference_cm and reference_pixels:
        try:
            ref_cm_val = float(reference_cm)
            ref_px_val = float(reference_pixels)
            processor.calibrate_pixel_ratio(ref_cm_val, ref_px_val)
            session_calibration[session_id] = processor.pixel_to_cm_ratio
        except (ValueError, TypeError):
            return jsonify({'error': 'reference_cm and reference_pixels must be numbers.', 'error_type': 'invalid_reference'}), 400

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.', 'error_type': 'no_image'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image file provided.', 'error_type': 'empty_filename'}), 400

    image = _read_image_from_file(image_file)
    if image is None:
        return jsonify({'error': 'Invalid image data. Please upload a valid JPEG or PNG.', 'error_type': 'invalid_image'}), 400

    # Assess image quality
    quality_info = _assess_image_quality(image)

    result = processor.process(image)
    if result is None:
        guidance = [
            "Ensure the animal is standing in a clear side profile.",
            "Make sure the full body from shoulder to heel is visible.",
            "Avoid extreme angles or partially visible animals.",
            "Try taking a new photo with better lighting."
        ]
        return jsonify({
            'error': 'Livestock posture not recognized. Could not detect animal pose.',
            'error_type': 'pose_detection_failed',
            'guidance': guidance,
            'image_quality': quality_info
        }), 400

    annotated_b64 = _encode_image_to_base64(result['annotated_image'])

    # Add weight sanity warnings if the computed value falls outside the expected range.
    guidance = [
        f"Animal identified as {result['animal_type']}.",
        f"Confidence: {round(result['confidence_score']*100, 1)}% - {'High' if result['confidence_score'] > 0.8 else 'Moderate'}",
        "For best accuracy, take a side-profile photo of the animal.",
        "Include a reference object (ruler/tape) in future photos."
    ]
    if not result.get('within_expected_range', True):
        exp_range = result.get('expected_weight_range')
        if exp_range:
            guidance.append(
                f"Estimated weight {result['weight']} kg is outside the expected range for {result['animal_type']} ({exp_range[0]}-{exp_range[1]} kg)."
            )
        guidance.append(
            "Check that the selected animal type matches the photo and review calibration measurements."
        )

    # Store in scan history
    scan_record = {
        'timestamp': datetime.now().isoformat(),
        'animal_name': animal_name,
        'animal_type': result['animal_type'],
        'farm_name': farm_name,
        'weight': result['weight'],
        'body_length': result['body_length'],
        'body_height': result['body_height'],
        'confidence_score': result['confidence_score'],
        'within_expected_range': result.get('within_expected_range', True)
    }
    scan_history.append(scan_record)

    return jsonify({
        'success': True,
        'weight': result['weight'],
        'body_length': result['body_length'],
        'body_height': result['body_height'],
        'estimated_girth': result['estimated_girth'],
        'animal_type': result['animal_type'],
        'confidence_score': result['confidence_score'],
        'pixel_to_cm_ratio': processor.pixel_to_cm_ratio,
        'image_quality': quality_info,
        'expected_weight_range': result.get('expected_weight_range'),
        'within_expected_range': result.get('within_expected_range'),
        'guidance': guidance,
        'annotated_image': annotated_b64,
    }), 200


@app.route('/api/animal-types', methods=['GET'])
def get_animal_types() -> Response:
    """Get available animal types and their calibration info."""
    processor = AnimalProcessor()
    types = processor.get_available_types()
    return jsonify({
        'success': True,
        'animal_types': types,
        'count': len(types)
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check() -> Response:
    """Health check endpoint for Base44 frontend."""
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0',
        'service': 'LivestockAI Weight Estimation API',
        'timestamp': datetime.now().isoformat(),
        'features': {
            'weight_estimation': True,
            'calibration': True,
            'reference_object_support': True,
            'session_tracking': True,
            'image_quality_assessment': True
        }
    }), 200


@app.route('/api/scan-history', methods=['GET'])
def get_scan_history() -> Response:
    """Get all scans from current session.
    
    Query parameters:
        - animal_type: Filter by animal type
        - limit: Maximum number of records to return (default: 100)
    """
    animal_type_filter = request.args.get('animal_type')
    limit = int(request.args.get('limit', 100))
    
    filtered_scans = scan_history
    if animal_type_filter:
        filtered_scans = [s for s in scan_history if animal_type_filter.lower() in s['animal_type'].lower()]
    
    return jsonify({
        'success': True,
        'total_scans': len(filtered_scans),
        'scans': filtered_scans[-limit:]
    }), 200


@app.route('/api/scan-history', methods=['DELETE'])
def clear_scan_history() -> Response:
    """Clear scan history."""
    global scan_history
    scan_history = []
    return jsonify({
        'success': True,
        'message': 'Scan history cleared.'
    }), 200


@app.route('/api/session/calibration', methods=['GET'])
def get_session_calibration() -> Response:
    """Get current calibration for a session.
    
    Query parameters:
        - session_id: Session ID (default: 'default')
    """
    session_id = request.args.get('session_id', 'default')
    calibration_value = session_calibration.get(session_id)
    
    return jsonify({
        'success': True,
        'session_id': session_id,
        'pixel_to_cm_ratio': calibration_value if calibration_value else None,
        'is_calibrated': calibration_value is not None
    }), 200


@app.route('/api/session/calibration', methods=['POST'])
def set_session_calibration() -> Response:
    """Set calibration for a session.
    
    JSON body:
        {
            "session_id": "my_session",
            "pixel_to_cm_ratio": 0.1234
        }
    """
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
    """Calibrate the pixel-to-cm ratio or adjust weight formula.
    
    JSON body:
        {
            "action": "pixel_ratio" | "weight_formula",
            "animal_type": "dairy_cow" (for weight_formula action),
            "known_cm": 10.5 (for pixel_ratio action),
            "measured_pixels": 40 (for pixel_ratio action),
            "divisor": 11.88 (for weight_formula action, optional),
            "girth_multiplier": 2.8 (for weight_formula action, optional)
        }
    """
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
    """Get best practice guidelines for accurate weight estimation."""
    return jsonify({
        'success': True,
        'guidelines': {
            'photo_tips': [
                "Take a clear SIDE-PROFILE photo of the animal.",
                "Ensure the entire body is visible (shoulder to heel).",
                "Use good natural or artificial lighting.",
                "Avoid shadows across the animal's body.",
                "Keep the camera at roughly hip height."
            ],
            'calibration_tips': [
                "Include a known reference object (ruler, measuring tape, or credit card) in the photo.",
                "Credit card width: 8.56 cm (standard)",
                "A4 paper width: 21 cm, height: 29.7 cm",
                "Measure the object's pixel width in your photo.",
                "Send both measurements to the API for accurate scaling."
            ],
            'accuracy_factors': [
                "Confidence score > 80% = High accuracy",
                "Confidence score 50-80% = Moderate accuracy",
                "Confidence score < 50% = Re-take photo with better pose",
                "Proper calibration can improve accuracy by 10-20%",
                "Multiple scans help establish baseline for trending."
            ],
            'reference_objects': [
                {'name': 'Credit Card', 'width_cm': 8.56, 'height_cm': 5.398},
                {'name': 'A4 Paper', 'width_cm': 21, 'height_cm': 29.7},
                {'name': 'US Dollar Bill', 'width_cm': 16.66, 'height_cm': 6.63},
                {'name': 'Standard Ruler', 'width_cm': 30, 'height_cm': None},
                {'name': 'Measuring Tape', 'width_cm': 'Variable', 'height_cm': None}
            ]
        }
    }), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
