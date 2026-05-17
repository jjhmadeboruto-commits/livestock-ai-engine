# Use official Python image with slim footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies (this will install Torch CPU and Ultralytics)
# We add the extra index url for CPU-only PyTorch to save space, but on HF Spaces you have 16GB so it handles it fine.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics

# Copy the rest of the application
COPY . .

# Expose port 7860 (Hugging Face Spaces requires this port)
EXPOSE 7860

# Command to run the application using Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
