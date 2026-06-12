---
title: Livestock AI Engine
emoji: 🐮
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# LiveStock AI Engine

AI-powered livestock weight estimation platform using:
- YOLOv8
- OpenCV
- Flask
- React + Vite

## New Features

- **Robust species detection**: Integrated OpenAI **CLIP** zero‑shot classification to accurately identify cattle, pig, donkey, sheep, goat, and poultry, even when YOLO’s COCO labels miss them.
- **Cross‑button auto‑correction**: The backend now validates the animal in the image regardless of the button pressed and automatically corrects the species.
- **Invalid target rejection**: Images of non‑livestock (dogs, humans, vehicles) return **HTTP 422** with a clear error message.
- **Improved YOLO handling**: Considers all COCO animal classes and tests the top three candidate boxes through CLIP.
- **Resolution Auto-Calibration**: Automatically scales the default pixel-to-cm ratios based on the upload's resolution (relative to a 640px baseline). This guarantees realistic physical dimension and weight estimates even for ultra-high-resolution photos from modern phones and iPads (such as 4032x3024).

## Deployment

The app is containerised; the Dockerfile now pre‑downloads the CLIP model to avoid cold‑start latency.

# Build the Docker image
docker build -t livestock-ai .
# Run the container
docker run -p 8080:8080 livestock-ai

The live Hugging Face Space is available at:
`https://jjhboruto-livestock-ai-engine.hf.space`
