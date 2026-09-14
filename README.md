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

Current 3D reconstruction workflows typically require multi-pass planned flights, heavy manual intervention, and expensive photogrammetry software. AeroRecon aims to make 3D reconstruction accessible from a single drone flight pass, combining classical Structure-from-Motion with modern AI depth estimation for improved coverage in challenging conditions.

## Solution Overview

AeroRecon implements a 7-stage modular pipeline:

| Stage | Method | Purpose |
|-------|--------|---------|
| 1. Preprocessing | OpenCV / FFmpeg | Keyframe selection, blur/motion scoring |
| 2. Dynamic Masking | YOLOv8n-seg | Remove moving objects from reconstruction |
| 3. Structure from Motion | COLMAP | Camera poses, sparse point cloud |
| 4. Monocular Depth | Depth Anything V2 | Dense depth priors per keyframe |
| 5. Optical Flow | RAFT | Inter-frame motion consistency |
| 6. Dense Reconstruction | Depth back-projection + Gaussian Splatting | Dense 3D model |
| 7. Post-processing | pyproj, confidence scoring | Georeferencing, quality assessment |

### Key Differentiators

- **Hybrid reconstruction**: Classical SfM + learned depth priors for improved density
- **Single-pass operation**: No multi-flight planning required
- **Transparent confidence**: Per-point quality scoring, never fabricated
- **Hardware-aware**: Adapts to available GPU, degrades gracefully
- **Modular design**: Each pipeline stage is independently testable

## Architecture

```
┌─────────────────┐     REST + WebSocket     ┌──────────────────┐
│    React UI      │ ◄─────────────────────► │   FastAPI         │
│  Three.js 3D     │                          │   Pipeline        │
│  Tailwind CSS    │                          │   SQLite          │
└─────────────────┘                          └──────┬───────────┘
                                                     │
                    ┌────────────────────────────────┤
                    ▼              ▼           ▼     ▼
              ┌──────────┐  ┌──────────┐  ┌──────┐ ┌──────────┐
              │ COLMAP   │  │ Depth    │  │ RAFT │ │ Gaussian │
              │ SfM      │  │ Anything │  │      │ │ Splatting│
              └──────────┘  └──────────┘  └──────┘ └──────────┘
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

**Backend** (from project root):
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend** (separate terminal):
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** in your browser.

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
