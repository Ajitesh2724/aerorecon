<div align="center">

# ✈️ AeroRecon

### Single-Pass Drone Video → Accurate 3D Model

**Smart India Hackathon 2026 · Problem SIH26158**
*NTRO · Robotics & Drones*

</div>

---

## Problem Statement

Convert a continuous single-pass drone video into a metrically meaningful 3D reconstruction with confidence scoring, measurement tools, and optional georeferencing.

## Motivation

Current 3D reconstruction workflows typically require multi-pass planned flights, heavy manual intervention, and expensive photogrammetry software. AeroRecon converts a single continuous drone flight pass into a metrically meaningful, dense 3D model combining classical Structure-from-Motion, learned depth and optical-flow priors, 3D Gaussian Splatting, and georeferenced metric scaling.

## Solution Overview

AeroRecon implements an end-to-end 9-stage modular pipeline:

| Stage | Method | Purpose |
|-------|--------|---------|
| **1. Preprocessing** | OpenCV / FFmpeg | Video validation, Laplacian blur/motion scoring, keyframe extraction, thumbnail generation |
| **2. Dynamic Masking** | YOLOv8 / Frame Differencing | Motion-compensated segmentation to mask moving vehicles, people, and transient artifacts |
| **3. SfM Baseline** | COLMAP / OpenCV SIFT fallback | Feature matching, 6-DoF camera pose estimation, sparse point cloud generation |
| **4. Monocular Depth** | PyTorch MiDaS / Multi-scale Laplacian | Relative depth map prediction per keyframe with edge-preserving filtering |
| **5. Optical Flow** | DISOpticalFlow / Forward-Backward Check | Sub-pixel motion tracking and cross-frame geometric consistency verification |
| **6. Dense 3D & Splatting** | Multi-view Back-projection + 3DGS + Mesh | Dense point cloud fusion, official binary 3D Gaussian Splatting PLY, continuous Delaunay surface OBJ |
| **7. Georeferencing** | 7-DoF Sim(3) Umeyama Alignment | Correlate camera trajectory with WGS84 GPS telemetry, ENU plane projection, metric scale factor |
| **8. Confidence Mapping** | Multi-factor Uncertainty Model | Heuristic quality evaluation: view redundancy, reprojection precision, motion consistency (4 tiers) |
| **9. Package Export** | Automated Deliverable Archiving | Standalone ZIP packaging with all PLY point clouds, 3DGS models, OBJ meshes, GeoJSON flight paths, and audit reports |

### Key Differentiators

- **Hybrid Reconstruction**: Classical SfM baseline fused with learned depth priors for uniform density across low-texture regions.
- **Interactive 3D Measurements**: Raycasted Euclidean distance and elevation delta calculation directly inside the browser with metric (meters) and imperial (feet) conversions.
- **Single-Pass Aerial Operation**: Operates on linear, continuous drone video passes without requiring cross-grid flights.
- **Transparent Confidence**: Multi-factor scoring (High / Medium / Low / Unseen) with per-factor weighting and visual audit breakdown.
- **Graceful Hardware Fallback**: Runs on budget GPUs (GTX 1650 4GB) and CPU-only environments with native OpenCV/SciPy fallbacks.

## Architecture

```
┌────────────────────────────────────────────────────────┐
│             React 19 + Three.js Client                 │
│  - Multi-mode 3D Viewer (Sparse, Dense, Mesh)         │
│  - 3D Distance & Elevation Measurement Raycaster       │
│  - Confidence Distribution & Factor Audit Breakdown     │
│  - Real-time Stage Progress & WebSocket Telemetry      │
└───────────────────────────▲────────────────────────────┘
                            │ REST + WebSocket
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI Backend                     │
│  - Async SQLite Job Store (JobCRUD)                    │
│  - Multi-stage Worker Orchestrator (9 Stages)          │
│  - Artifact Streaming & ZIP Packaging                  │
└───────────────────────────┬────────────────────────────┘
                            │
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ SfM / COLMAP │     │ Learned Depth│     │ 3D Gaussians │
│ + SIFT Fallb.│     │ + DIS Flow   │     │ & Mesh Gen   │
└──────────────┘     └──────────────┘     └──────────────┘
```

## Installation

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **NVIDIA GPU** (recommended, GTX 1650+ with 4 GB+ VRAM)
- **COLMAP** ([install guide](https://colmap.github.io/install.html)) — optional, required for SfM
- **FFmpeg** — optional, OpenCV fallback available

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

For GPU support (added in Milestone 4):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Configuration

```bash
cp .env.example .env
# Edit .env to set paths, GPU config, etc.
```

### COLMAP Installation (Windows)

Download the pre-built binary from [COLMAP releases](https://github.com/colmap/colmap/releases) and set `COLMAP_PATH` in your `.env` file.

### Running the Application

**Option A: Local Development**

1. **Backend** (from project root):
   ```bash
   cd backend
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Frontend** (separate terminal):
   ```bash
   cd frontend
   npm run dev
   ```
   Open **http://localhost:5173** in your browser.

**Option B: Docker Compose**

```bash
docker compose up --build
```
Open **http://localhost:8000** for the containerized application.

### Demo Dataset Generation

Generate synthetic drone video footage and telemetry for immediate demonstration and testing:
```bash
python backend/scripts/generate_demo_dataset.py
```
This produces:
- `data/demo/demo_flight.mp4`: Synthetic aerial footage with textured structures.
- `data/demo/demo_telemetry.json` & `demo_telemetry.csv`: Synchronized GPS & IMU telemetry.

### Running Automated Verification Tests

```bash
cd backend
python -m pytest -q tests/test_api.py tests/test_processing.py tests/test_sfm.py tests/test_learned_priors.py tests/test_dense_reconstruction.py tests/test_operational.py tests/test_e2e_pipeline.py
```

## API Documentation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health and tool availability |
| `/api/jobs` | POST | Upload video & create reconstruction job |
| `/api/jobs` | GET | List all jobs |
| `/api/jobs/{id}` | GET | Get job details |
| `/api/jobs/{id}/process` | POST | Start reconstruction pipeline |
| `/api/jobs/{id}/status` | GET | Lightweight status poll |
| `/api/jobs/{id}/results` | GET | Reconstruction results & artifacts |
| `/api/jobs/{id}/confidence` | GET | Confidence scoring breakdown |
| `/api/jobs/{id}/download/{artifact}` | GET | Download output file |
| `/api/jobs/{id}` | DELETE | Delete job and files |
| `/ws` | WebSocket | Real-time updates (all jobs) |
| `/ws/{id}` | WebSocket | Real-time updates (specific job) |

## Confidence Interpretation

AeroRecon assigns confidence tiers to reconstructed regions:

| Tier | Meaning |
|------|---------|
| **HIGH** | Strong multi-view coverage, consistent depth, low reprojection error |
| **MEDIUM** | Partial coverage or moderate consistency |
| **LOW** | Sparse observations, significant uncertainty |
| **UNSEEN** | Insufficient direct evidence |

> ⚠️ Confidence scores are **estimated heuristics**, not certified accuracy metrics.

## Measurement Limitations

- Measurements are only as accurate as the reconstruction
- Without GPS data, measurements are in arbitrary (relative) units
- With GPS, metric scale is estimated but **not surveying-grade**
- All measurements display calibration status and uncertainty warnings

## Known Limitations

- GTX 1650 (4 GB VRAM) limits Gaussian Splatting to small scenes
- COLMAP required for full SfM pipeline (fallback is basic)
- Single-pass flights may have low overlap → reduced reconstruction quality
- No real-time processing — pipeline runs asynchronously
- Depth estimation is monocular (not stereo) — metric scale relies on GPS

## Future Improvements

- [ ] Multi-GPU support for larger scenes
- [ ] Real-time SLAM integration
- [ ] Cloud processing backend
- [ ] Point cloud editing tools
- [ ] Ortho-mosaic generation
- [ ] LiDAR fusion pipeline
- [ ] Mobile-optimized viewer

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Backend | Python 3.11, FastAPI, SQLite, OpenCV, PyTorch |
| Frontend | React 19, TypeScript, Vite, Three.js, Tailwind CSS |
| 3D Engine | React Three Fiber, Drei |
| SfM | COLMAP |
| Depth | Depth Anything V2 (ViT-S) |
| Flow | RAFT (Small) |
| Detection | YOLOv8n-seg |
| Dense 3D | 3D Gaussian Splatting, Open3D |
| Georef | pyproj |

## License

This project was developed as an original solution for SIH 2026 Problem 26158.

---

<div align="center">
<sub>Built with ❤️ for Smart India Hackathon 2026</sub>
</div>
