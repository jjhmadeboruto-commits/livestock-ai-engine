"""
Test script for the calibrated livestock weight estimation API.
Demonstrates different animal types and calibration methods.
"""
import requests
import json

BASE_URL = "http://localhost:5000"

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")

def test_get_animal_types():
    """Retrieve available animal types and calibration info."""
    print_section("AVAILABLE ANIMAL TYPES")
    response = requests.get(f"{BASE_URL}/api/animal-types")
    if response.status_code == 200:
        types = response.json().get('animal_types', {})
        for animal_type, config in types.items():
            print(f"Type: {animal_type}")
            print(f"  Name: {config['name']}")
            print(f"  Divisor: {config['divisor']}")
            print(f"  Girth Multiplier: {config['girth_multiplier']}")
            print()
    else:
        print(f"Error: {response.text}")

def test_estimate_weight(image_path, animal_type="dairy_cow", reference_cm=None, reference_pixels=None, session_id='default'):
    """Test weight estimation for a specific animal type."""
    print_section(f"WEIGHT ESTIMATION - {animal_type.upper()}")
    
    try:
        with open(image_path, 'rb') as img:
            files = {'image': img}
            params = {'animal_type': animal_type, 'session_id': session_id}
            data = {}
            if reference_cm is not None and reference_pixels is not None:
                data['reference_cm'] = reference_cm
                data['reference_pixels'] = reference_pixels

            response = requests.post(
                f"{BASE_URL}/api/estimate-weight",
                files=files,
                params=params,
                data=data
            )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Animal Type: {result['animal_type']}")
            print(f"Weight: {result['weight']} kg")
            print(f"Body Length: {result['body_length']} cm")
            print(f"Body Height: {result['body_height']} cm")
            print(f"Estimated Girth: {result['estimated_girth']} cm")
            print(f"Confidence Score: {result['confidence_score']}")
            print(f"Pixel Ratio: {result.get('pixel_to_cm_ratio')}")
            if result.get('image_quality'):
                print(f"Image Quality: {result['image_quality']}")
            return result
        else:
            print(f"Error: {response.status_code} {response.text}")
            return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None

def test_session_calibration(session_id='default', known_cm=8.56, measured_pixels=150):
    """Set and verify session calibration via POST /api/session/calibration."""
    print_section(f"SESSION CALIBRATION - {session_id}")
    payload = {
        'session_id': session_id,
        'pixel_to_cm_ratio': known_cm / measured_pixels
    }
    response = requests.post(f"{BASE_URL}/api/session/calibration", json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Calibration set: {result['pixel_to_cm_ratio']:.6f} cm/pixel")
    else:
        print(f"Error: {response.text}")


def test_calibrate_pixel_ratio(known_cm=10.0, measured_pixels=37.8):
    """Calibrate the pixel-to-cm conversion ratio."""
    print_section("CALIBRATE PIXEL RATIO")
    
    data = {
        "action": "pixel_ratio",
        "known_cm": known_cm,
        "measured_pixels": measured_pixels
    }
    
    response = requests.post(f"{BASE_URL}/api/calibrate", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Message: {result['message']}")
        print(f"New Pixel-to-CM Ratio: {result['pixel_to_cm_ratio']:.6f}")
    else:
        print(f"Error: {response.text}")

def test_calibrate_weight_formula(animal_type="dairy_cow", divisor=None, girth_multiplier=None):
    """Calibrate the weight estimation formula for a specific animal type."""
    print_section(f"CALIBRATE WEIGHT FORMULA - {animal_type.upper()}")
    
    data = {
        "action": "weight_formula",
        "animal_type": animal_type
    }
    
    if divisor is not None:
        data["divisor"] = divisor
    if girth_multiplier is not None:
        data["girth_multiplier"] = girth_multiplier
    
    response = requests.post(f"{BASE_URL}/api/calibrate", json=data)
    if response.status_code == 200:
        result = response.json()
        print(f"Message: {result['message']}")
        print(f"Updated Calibration:")
        for key, value in result['calibration'].items():
            print(f"  {key}: {value}")
    else:
        print(f"Error: {response.text}")

def main():
    """Run comprehensive calibration tests."""
    print("\n" + "="*60)
    print("  LIVESTOCK WEIGHT ESTIMATION API - CALIBRATION TEST")
    print("="*60)
    
    image_path = "C:\\Users\\User 2\\Downloads\\istockphoto-496397741-612x612.jpg"
    
    test_get_animal_types()
    test_session_calibration(session_id='default', known_cm=8.56, measured_pixels=150)

    animal_types = [
        'dairy_cow',
        'beef_cattle',
        'young_cattle',
        'goat',
        'sheep',
        'donkey',
        'poultry'
    ]

    results = {}
    for animal_type in animal_types:
        results[animal_type] = test_estimate_weight(
            image_path,
            animal_type=animal_type,
            reference_cm=8.56,
            reference_pixels=150,
            session_id='default'
        )

    print("\n" + "="*60)
    print("  WEIGHT COMPARISON BY ANIMAL TYPE")
    print("="*60 + "\n")

    for animal_type in animal_types:
        result = results.get(animal_type)
        if result:
            print(f"{animal_type.replace('_', ' ').title()}: {result['weight']} kg")
        else:
            print(f"{animal_type.replace('_', ' ').title()}: FAILED")

    print("\nNote: Different coefficients are used for each animal type.")
    print("Use the same Base44 payload contract for all animal types.")

if __name__ == "__main__":
    main()
