# Integrated Emotion + Posture Detection

Real-time comfort-aware perception system for human-robot handover scenarios. Combines GPU-based emotion/gaze detection with CPU-based posture analysis on Intel RealSense D435 `.bag` recordings, producing a unified comfort score.

## Architecture

```
RealSense .bag (color + depth)
        │
        ├──► Face Detection (RetinaFace)
        │       ├──► Emotion Detection (EfficientNet-B0, AffectNet)
        │       └──► Gaze Estimation (L2CS-Net, Gaze360)
        │
        ├──► Pose Detection (MediaPipe Pose)
        │       ├──► Open Posture Scoring
        │       ├──► Mouth Covering Detection
        │       └──► Depth-based Withdrawal Detection
        │
        └──► Integrated Comfort Scorer
                ├──► Emotion Comfort (55%)
                ├──► Posture Comfort (45%)
                └──► Unified Score (0-100, EMA-smoothed)
```

## Setup

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (for emotion/gaze models)
- Intel RealSense SDK 2.0

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Models

Download pre-trained models using the provided script:

```bash
python scripts/download_models.py
```

Or symlink from an existing emotion detection repo:

```bash
ln -s /path/to/mmai-emotion-detection/models models
```

### Data

Place `.bag` files in `data/` organized by scenario and lighting:

```
data/
├── sc02_comfortable/
│   ├── Bright/
│   │   └── recording.bag
│   └── Dark/
│       └── recording.bag
├── sc04_sudden_withdrawal/
│   └── ...
```

Or symlink from an existing data directory:

```bash
ln -s /path/to/mmai-emotion-detection/data data
```

## Usage

### Interactive Playback

```bash
python scripts/run_bag.py
```

Select files interactively. Controls:
- `q` — quit
- `n` — skip to next file

### Process Specific File

```bash
python scripts/run_bag.py data/sc02_comfortable/Dark/recording.bag
```

### Save Annotated Videos

```bash
python scripts/run_bag.py --save
python scripts/run_bag.py --save --save-dir rendering_output/
```

### Headless Mode

```bash
python scripts/run_bag.py --headless
python scripts/run_bag.py --headless --save --save-dir rendering_output/
```

### Multiple File Selection

At the interactive prompt, enter:
- `3` — single file
- `1,4,7` — comma-separated
- `2-5` — range
- `0` — all files

## Visualization Layout

```
+--------------------------------------------------+
| [Comfort: 72/100]                      FPS: 28   |
|                                                   |
| happy (V:+0.45 A:-0.10)       Posture: 0.75      |
| Emotion Score: 68              Depth Z: 0.85m     |
| Gaze: Looking at camera       Posture Score: 82   |
|                                                   |
|        [face bbox]         [depth marker]         |
|                                                   |
|         STATE: SCARED (mouth/face covered)        |
| sc02/Bright  t=5.2s  frame=156                    |
+--------------------------------------------------+
```

## Configuration

All parameters are in `config/default.yaml`. Key sections:

| Section | Description |
|---------|-------------|
| `face_detector` | RetinaFace confidence threshold, bbox expansion |
| `emotion_detector` | EfficientNet model, input size |
| `gaze_detector` | L2CS-Net yaw/pitch thresholds |
| `pose_detector` | MediaPipe settings, posture thresholds, depth/withdrawal params |
| `comfort` | Scoring weights, EMA smoothing, decay rates |
| `visualization` | Toggle individual overlay elements |

## Comfort Scoring

**Emotion comfort** (0-100): Based on valence, arousal, and gaze engagement.

**Posture comfort** (0-100): Based on open posture score, with penalties for mouth covering (-30) and sudden withdrawal (-25).

**Integrated comfort**: Weighted blend — 55% emotion, 45% posture — with independent EMA smoothing per component.

## Dependencies

- **GPU**: PyTorch, timm, EmotiEffLib, L2CS-Net
- **CPU**: MediaPipe Pose
- **Shared**: OpenCV, NumPy, pyrealsense2
