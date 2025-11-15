# Facetrack - Intelligent Face Tracking System

This project performs intelligent face detection, tracking, and recognition for entry/exit logging using YOLO, ByteTrack, and InsightFace.

Loom - project explaination
https://www.loom.com/share/879944448b1e4b9baee2128351a5cb4e

## Features
- Real-time face detection and tracking
- Face recognition and re-identification
- Entry/exit event logging with timestamps
- PostgreSQL database integration for visitor tracking
- Configurable similarity thresholds and timeouts
- Support for webcam and video file inputs

## Setup Instructions (Windows)

### 1. Prerequisites
- Python 3.10 or later
- PostgreSQL 14+ with the `vector` extension installed
- CUDA-capable GPU (optional, speeds up detection)

### 2. Environment Setup (PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Option A: Activate venv (requires execution policy change)
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned -Force
.\.venv\Scripts\Activate.ps1

# Option B: Or use venv Python directly without activation
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Install dependencies
python -m pip install -r requirements.txt

# Install InsightFace wheel (if present in repo)
python -m pip install .\insightface-0.7.3-cp310-cp310-win_amd64.whl
```

### 3. Database Setup
1. Copy `config.example.json` to `config.json` and update database credentials
2. Initialize the database:
```powershell
.\.venv\Scripts\python.exe innit_db.py
```

### 4. Input Video Setup
Place your input videos in a folder (e.g., `videos/`) and update `config.json`:
```json
{
    "video_source": "video.mp4",  # or 0 for webcam
    "frame_skip": 3,  # process every Nth frame
    "similarity_threshold": 0.6,
    "use_gpu": false  # set true if using CUDA GPU
}
```

### 5. Run the System

Full system with database:
```powershell
.\.venv\Scripts\python.exe main.py
```

Test video processing only (no DB):
```powershell
.\.venv\Scripts\python.exe test_video.py
```

## Controls
- Press 'q' to quit the video display
- The window shows:
  - Current face detections (green boxes)
  - Unique visitor count
  - Entry/exit events are logged to the database and `logs/entries/`

## Troubleshooting
- VS Code: Select interpreter (Ctrl+Shift+P → Python: Select Interpreter) → choose `.venv\Scripts\python.exe`
- If imports fail: `python -m pip install --force-reinstall -r requirements.txt`
- PostgreSQL connection issues: Check `config.json` credentials and ensure PostgreSQL is running
- CUDA/GPU issues: Set `use_gpu": false` in config.json to force CPU mode

## Project Structure
- `main.py` - Main application entry point
- `face_embedder.py` - InsightFace model wrapper
- `database.py` - PostgreSQL database operations
- `state_tracker.py` - Visitor state management
- `test_video.py` - Standalone video processing test
- `innit_db.py` - Database initialization
- `config.example.json` - Configuration template

---

Hackathon notes
- This project is a part of a hackathon run by https://katomaran.com

Architecture (high level)

  [Video Source/RTSP] --> [YOLOv8 Detector + ByteTrack] --> [VisitorTracker]
                                      |                         |
                                      v                         v
                               [Face crops saved]        [Database (Visitors, Events)]


folders of the model and training videos are not added here in the repo




