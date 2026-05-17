# Copy-paste fixes for Base44 editor

Use these if you edit inside Base44 when credits return. Search for the **Before** text and replace with **After**.

---

## 1. Loading message (scan screen)

**Before:** `MediaPipe pose detection running`  
**After:** `Analyzing image`

**Before:** `Analysing morphology…`  
**After:** `Analyzing image…`

**Before:** `Detecting skeletal keypoints`  
**After:** `Detecting animal in frame`

---

## 2. Homepage feature text

**Before:** `MediaPipe pose detection estimates body length and height instantly.`  
**After:** `Computer vision estimates body length and height from your photo.`

---

## 3. API call — pass animal type + 90s timeout

Find the function that POSTs to `/api/estimate-weight` (often named like `analyzeImage` or contains `FormData` + `fetch`).

**Before (conceptually):**

```javascript
const formData = new FormData();
formData.append("image", file);
const response = await fetch(`${API_URL}/api/estimate-weight`, {
  method: "POST",
  body: formData,
});
```

**After:**

```javascript
const ANIMAL_MAP = {
  Cattle: "dairy_cow",
  Pig: "pig",
  Poultry: "poultry",
  Goat: "goat",
  Sheep: "sheep",
  Donkey: "donkey",
};

const formData = new FormData();
formData.append("image", file);
const animalKey = ANIMAL_MAP[selectedAnimal] || "dairy_cow";

const response = await fetch(
  `${API_URL}/api/estimate-weight?animal_type=${animalKey}`,
  {
    method: "POST",
    body: formData,
    signal: AbortSignal.timeout(90_000),
  }
);
```

Make sure the scan handler passes `selectedAnimal` (e.g. `"Cattle"`, `"Pig"`) into this function.

---

## 4. Response fields (backend already fixed)

The API now returns both naming styles. Your UI should read:

- `estimated_weight_kg` or `weight`
- `confidence_interval` or `confidence_score`
- `body_length_cm` or `body_length`
- `annotated_image_base64` or `annotated_image`
