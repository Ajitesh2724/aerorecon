# Multi-stage Dockerfile for AeroRecon (FastAPI backend + Vite React frontend)
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app

# Install system dependencies: OpenCV headless dependencies, libgl, git, curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy built frontend assets to static dist
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Create storage directories
RUN mkdir -p /app/data/jobs /app/data/uploads

ENV HOST=0.0.0.0
ENV PORT=8000
ENV DATA_DIR=/app/data

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
