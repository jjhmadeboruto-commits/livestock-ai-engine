import urllib.request
import urllib.error
import ssl
import json
import base64

ctx = ssl.create_default_context()

# Download a public photo of an animal to use for live camera simulation
img_url = "https://images.unsplash.com/photo-1548550023-2bdb3c5beed7?w=640" # public photo of an animal (chicken)
print(f"Downloading test image from {img_url}...")
try:
    ctx_no_ssl = ssl._create_unverified_context()
    with urllib.request.urlopen(img_url, context=ctx_no_ssl, timeout=15) as r:
        img_bytes = r.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
except Exception as e:
    print(f"Failed to download test image: {e}")
    exit(1)

# Build JSON request payload
payload = {
    "image_data": img_b64,
    "animal_type": "goat",
    "animal_name": "Test Goat",
    "farm_name": "Test Farm",
    "run_self_critique": True,
    "save_to_history": False
}

url = "https://jjhboruto-livestock-ai-engine.hf.space/api/live-cam/frame"
req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST")
req.add_header("Content-Type", "application/json")

print("Sending request to /api/live-cam/frame...")
try:
    with urllib.request.urlopen(req, context=ctx, timeout=90) as r:
        data = json.loads(r.read().decode("utf-8"))
        print("\n=== LIVE CAM API RESPONSE SUCCESS ===")
        print(f"Status:             {data.get('status')}")
        print(f"Weight (Confirmed):  {data.get('weight_kg')} kg")
        print(f"Proposed Weight:    {data.get('proposed_weight_kg')} kg")
        print(f"Was Adjusted:       {data.get('was_adjusted')}")
        print(f"Confidence Level:   {data.get('confidence_level')}")
        print(f"Critique Notes:     {data.get('critique_notes')}")
        print(f"Adjustment Reason:  {data.get('adjustment_reason')}")
        print(f"Gemini Explanation: {data.get('gemini_explanation')}")
        print("\n=== FRAME ANALYSIS STEPS ===")
        frame_analysis = data.get("frame_analysis", {})
        for step_key, step_data in frame_analysis.items():
            print(f"  [{step_key}] {step_data.get('label')}: {step_data.get('status')} -> {step_data.get('detail')}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}:")
    try:
        body = e.read().decode()
        print("Body:", body)
    except Exception as ex:
        print("Could not read body:", ex)
except Exception as e:
    print("Error calling endpoint:", e)
