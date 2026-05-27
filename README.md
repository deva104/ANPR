# 🚗 ANPR Gate Access Control System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E?style=for-the-badge&logo=supabase)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv)

**Real-time Automatic Number Plate Recognition for intelligent gate access control**

[Features](#-features) • [Demo](#-demo) • [Architecture](#-architecture) • [Performance](#-performance) • [Setup](#-setup) • [Usage](#-usage)

</div>

---

## 📌 Overview

A complete end-to-end ANPR system that detects Indian vehicle licence plates from an **ESP32-CAM live stream**, verifies them against a **cloud database**, and controls gate access — all with a **real-time web dashboard** accessible from any device.

Built specifically for Indian licence plate formats using a two-stage deep learning pipeline:
- **Stage 1** — FCOS + HRNet backbone for plate localization
- **Stage 2** — LPRNet with CTC loss for text recognition

Achieved **35 FPS on CPU** (10.4× speedup over baseline) through Phase 1 pipeline optimizations — no GPU required.

---

## ✨ Features

- 🎯 **Real-time plate detection** at 35 FPS on CPU hardware
- 📷 **ESP32-CAM integration** — low-cost wireless camera streaming over WiFi
- 🔐 **Verify-once mechanism** — 5-second cooldown prevents repeated database writes
- 🗳️ **Temporal voting** — filters OCR misreads using rolling buffer consensus
- 👥 **4 vehicle categories** — Owner, Renter, Visitor, Relative
- ⏰ **Automatic pass expiry** — Visitor/Relative passes auto-denied after valid date
- 🌐 **Real-time web dashboard** — live WebSocket updates to any device on network
- 📊 **Access logs** — every detection event logged to Supabase with export to CSV
- 🚨 **Unknown vehicle popup** — instant register prompt for unrecognized plates
- 📱 **Mobile responsive** — dashboard works on phone browser

---

## 🎬 Demo

> Dashboard screenshot — add your screenshot here

| Detection | Dashboard | Vehicle Management |
|-----------|-----------|-------------------|
| ![Detection](assets/detection.png) | ![Dashboard](assets/dashboard.png) | ![Vehicles](assets/vehicles.png) |

---

## 🏗️ Architecture

```
ESP32-CAM ──WiFi (MJPEG)──► live_anpr.py
                              │
                     FCOS + HRNet Detector
                     LPRNet OCR
                     PlateTracker (voting)
                              │
                    HTTP POST /api/verify
                              │
                         backend.py (FastAPI)
                         │              │
                    Supabase        WebSocket
                  PostgreSQL     ──────────────►  Browser Dashboard
                  (cloud DB)                      any device on WiFi
```

### Key Components

| File | Purpose |
|------|---------|
| `live_anpr.py` | Main detection loop — reads ESP32 stream, runs ANPR, sends to backend |
| `src/plate_tracker.py` | PlateTracker — IoU matching, temporal voting, cooldown logic |
| `src/infer_video_utils.py` | Phase 1 optimized video pipeline |
| `backend.py` | FastAPI server — REST API + WebSocket |
| `database.py` | Supabase client — all DB functions including verify_plate() |
| `frontend/index.html` | Dashboard — live detections, access logs, stats |
| `frontend/vehicles.html` | Vehicle management — add/edit/delete, filter by type |

---

## ⚡ Performance

Phase 1 optimizations achieved **10.4× speedup** on CPU without any GPU or model changes:

| Configuration | FPS | Detector % | Notes |
|--------------|-----|-----------|-------|
| Baseline (no optimization) | 3.4 | ~90% | Every frame, full resolution |
| + MIL Tracker | 16.3 | 42.4% | Frame skip helped, tracker killed gain |
| + Remove MIL Tracker | 20.7 | 67.3% | Velocity extrapolation instead |
| **Run A — Final** | **35.4** | **61.5%** | detect_every=5, detect_scale=0.50 |
| Run B (aggressive) | 45.7 | 53.6% | detect_every=8, quality tradeoff |

### Time Breakdown (Run A, 900 frames)

```
FCOS Detector:    15.64s  (61.5%)  ← 180 runs only (every 5th frame)
VideoWriter:       4.03s  (15.9%)
LPRNet OCR:        3.41s  (13.4%)  ← 292 runs (67% skipped by gating)
cap.read():        1.55s  ( 6.1%)
Other:             0.71s  ( 2.8%)
Tracker update:    0.01s  ( 0.0%)  ← velocity extrapolation, near zero
```

### Pipeline Optimizations Applied

- **Frame skipping** — FCOS runs every 5th frame only (80% reduction in detection calls)
- **Detect scale 0.50** — detector input at 50% resolution (75% fewer pixels)
- **Velocity extrapolation** — boxes advance between detections using estimated velocity
- **OCR gating** — size + sharpness checks before running LPRNet
- **Temporal voting** — 12-frame buffer, majority vote, sanity filter (4-13 chars)
- **Verify-once cooldown** — 5-second real-time cooldown prevents re-verification

---

## 🔧 Setup

### Prerequisites

- Python 3.10
- ESP32-CAM (AI-Thinker) with MB programmer board
- Arduino IDE 2.x (for ESP32 firmware)
- Supabase account (free tier)

### 1. Clone the repository

```bash
git clone https://github.com/deva104/ANPR.git
cd ANPR
```

### 2. Create virtual environment

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -r requirements_backend.txt
```

### 4. Download pretrained weights

Place these files in the `weights/` folder:
```
weights/
├── best_od.pth        # FCOS object detector
├── best_lprnet.pth    # LPRNet OCR
└── best_semantic.pth  # Semantic segmentation (alternative)
```

> Weights are available from the [Indian_LPR](https://github.com/xuebinqin/Indian_LPR) upstream repository.

### 5. Configure Supabase

Create a free project at [supabase.com](https://supabase.com) and run these SQL queries:

```sql
CREATE TABLE vehicles (
    id BIGSERIAL PRIMARY KEY,
    plate_number TEXT UNIQUE NOT NULL,
    owner_name TEXT NOT NULL,
    vehicle_type TEXT NOT NULL
        CHECK (vehicle_type IN ('owner', 'renter', 'visitor', 'relative')),
    valid_from TIMESTAMPTZ NULL,
    valid_until TIMESTAMPTZ NULL,
    purpose TEXT NULL,
    added_on TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE access_logs (
    id BIGSERIAL PRIMARY KEY,
    plate_number TEXT NOT NULL,
    owner_name TEXT DEFAULT 'Unknown',
    vehicle_type TEXT NULL,
    detected_on TIMESTAMPTZ DEFAULT NOW(),
    access_granted BOOLEAN NOT NULL,
    denial_reason TEXT NULL
);

-- Allow anon access
CREATE POLICY "allow_all_vehicles" ON vehicles FOR ALL TO anon USING (true) WITH CHECK (true);
CREATE POLICY "allow_all_logs" ON access_logs FOR ALL TO anon USING (true) WITH CHECK (true);
```

### 6. Configure environment

Create `.env` file in project root:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
LAPTOP_IP=localhost
BACKEND_PORT=8000
ESP32_STREAM_URL=http://YOUR_ESP32_IP:81/stream
```

### 7. Flash ESP32-CAM

1. Open Arduino IDE → File → Examples → ESP32 → Camera → CameraWebServer
2. Set your WiFi credentials and uncomment `#define CAMERA_MODEL_AI_THINKER`
3. Hold IO0 button + press RST → Upload
4. After upload: remove IO0 wire → press RST → note IP from Serial Monitor

---

## 🚀 Usage

### Start the backend

```bash
python backend.py
```

Backend runs at `http://localhost:8000`

### Start live ANPR detection

```bash
python live_anpr.py --source "http://YOUR_ESP32_IP:81/stream"
```

### Open dashboard

```
http://localhost:8000              # same machine
http://YOUR_LAPTOP_IP:8000         # any device on same WiFi
```

### Optional flags

```bash
python live_anpr.py \
  --source "http://ESP32_IP:81/stream" \
  --backend "http://localhost:8000" \
  --mjpeg              # use FFMPEG backend for stream
```

### Process video files

```bash
python infer_objectdet.py \
  --source "demo_videos/test.mp4" \
  --output_path "out_videos/objectdet" \
  --detect_every 5 \
  --detect_scale 0.50 \
  --live_preview
```

---

## 📁 Project Structure

```
ANPR/
├── live_anpr.py                    # Live ESP32 stream detection
├── backend.py                      # FastAPI REST + WebSocket server
├── database.py                     # Supabase database functions
├── infer_objectdet.py              # Video/image inference (object detection)
├── infer_semanticseg.py            # Video/image inference (segmentation)
├── test_phase1.py                  # Phase 1 benchmark script
├── requirements.txt                # ANPR dependencies
├── requirements_backend.txt        # Backend dependencies
├── .env                            # Credentials (not committed)
├── frontend/
│   ├── index.html                  # Dashboard
│   ├── vehicles.html               # Vehicle management
│   ├── style.css                   # Dark theme CSS
│   ├── app.js                      # Dashboard JS
│   └── vehicles.js                 # Vehicles page JS
├── src/
│   ├── plate_tracker.py            # PlateTracker class
│   ├── infer_video_utils.py        # Phase 1 video pipeline
│   ├── object_detection/           # FCOS + HRNet model
│   ├── semantic_segmentation/      # HRNet segmentation model
│   └── License_Plate_Recognition/  # LPRNet model
├── weights/                        # Pretrained model weights
├── demo_videos/                    # Test videos (not committed)
└── out_videos/                     # Output results (not committed)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Camera | ESP32-CAM AI-Thinker (OV2640) |
| Detection | FCOS + HRNet backbone (PyTorch) |
| OCR | LPRNet with CTC loss (PyTorch) |
| Tracking | Custom PlateTracker (IoU + velocity) |
| Backend | FastAPI + Uvicorn |
| Database | Supabase (PostgreSQL) |
| Realtime | WebSocket |
| Frontend | Plain HTML + CSS + JS |
| Hardware | ASUS S14 (Ryzen AI 9 HX 370) |

---

## 📋 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/vehicles` | List all registered vehicles |
| POST | `/api/vehicles` | Add new vehicle |
| PUT | `/api/vehicles/{plate}` | Update vehicle |
| DELETE | `/api/vehicles/{plate}` | Remove vehicle |
| GET | `/api/logs` | Get access logs (last 50) |
| GET | `/api/logs/search/{plate}` | Search logs by plate |
| GET | `/api/stats` | Today's stats |
| POST | `/api/verify` | Verify plate + log + WebSocket push |
| WS | `/ws` | WebSocket for live updates |

---

## 🗺️ Roadmap

- [ ] ONNX export + DirectML for AMD GPU acceleration
- [ ] Servo motor integration for physical barrier control
- [ ] Night vision with IR illumination support
- [ ] Multi-camera support
- [ ] Mobile push notifications for unknown vehicles
- [ ] Cloud deployment (Railway/Fly.io)
- [ ] LPRNet fine-tuning on larger Indian plate dataset

---

## 🙏 Acknowledgements

- [Indian_LPR](https://github.com/xuebinqin/Indian_LPR) — upstream FCOS + LPRNet implementation for Indian plates
- [LPRNet Paper](https://arxiv.org/abs/1806.10447) — Zherzdev et al., 2018
- [FCOS Paper](https://arxiv.org/abs/1904.01355) — Tian et al., 2019
- [HRNet](https://arxiv.org/abs/1908.07919) — Sun et al., 2019

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built by <a href="https://github.com/deva104">Devendra Harale</a> • MIT AOE, Alandi • 2025-26
</div>
