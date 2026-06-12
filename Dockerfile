# Use official Python image with slim footprint
FROM python:3.11-slim

# Set working directory
WORKDIR /app



# Copy requirements
COPY requirements.txt .

# Install Python dependencies (this will install Torch CPU and Ultralytics)
# We add the extra index url for CPU-only PyTorch to save space, but on HF Spaces you have 16GB so it handles it fine.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ultralytics "numpy<2.0.0"

# Ultralytics forcefully installs the GUI version of OpenCV (opencv-python).
# We must uninstall it and reinstall the headless version to prevent libxcb/libgl runtime errors.
RUN pip uninstall -y opencv-python opencv-python-headless && \
    pip install --no-cache-dir opencv-python-headless==4.8.1.78 "numpy<2.0.0"

# Download YOLOv8n weights at build time (avoids runtime download issues on free tier)
RUN mkdir -p /app/models && \
    python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt')" && \
    cp -f yolov8n.pt /app/models/yolov8n.pt 2>/dev/null || true && \
    find / -name 'yolov8n.pt' -type f 2>/dev/null | head -1 | xargs -I{} cp {} /app/models/yolov8n.pt 2>/dev/null || true

# Copy the rest of the application
COPY . .

# Expose port 7860 (Hugging Face Spaces requires this port)
EXPOSE 7860

# Command to run the application using Gunicorn on port 7860
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "2", "--timeout", "120", "app:app"]
