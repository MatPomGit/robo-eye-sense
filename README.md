# robo-vision

Lightweight real-time visual marker detection for mobile robots.

**robo-eye-sense** combines three complementary detection technologies —
AprilTag fiducial markers, QR codes, and laser-pointer spots — into a single,
unified detection pipeline.  Every detected object is assigned a *stable
track ID* that persists across frames, even through brief occlusions.  The
system ships with a full Tkinter GUI for interactive tuning and a headless CLI
for deployment on embedded hardware.

## Features

| Capability | Technology |
|---|---|
| **AprilTag detection & tracking** | [pupil-apriltags](https://github.com/pupil-labs/apriltags) |
| **QR-code detection & decoding** | [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar) *(preferred)* / OpenCV fallback |
| **Laser-spot detection** | OpenCV brightness thresholding + circularity filter |
| **Multi-object tracking** | Centroid tracker with persistent IDs (Kalman-filter mode available) |
| **Interactive GUI** | Tkinter control panel – change mode, toggle detectors, tune laser parameters live |
| **Three operating modes** | Normal / Fast (half-resolution) / Robust (sharpening + Kalman tracking) |

All detectors run on every frame and their results are unified through a
single `CentroidTracker` that assigns stable track IDs.  Labeled objects
(AprilTags, QR codes) are matched by their semantic identity so the same
physical marker always keeps the same track ID even after a brief occlusion.
Unlabeled objects (laser spots) are matched by nearest-centroid distance, with
optional Kalman-filter prediction in ROBUST mode.

---

## Quick start

### 1 – Install system dependencies

```bash
# Ubuntu / Debian (required for QR-code detection via pyzbar)
sudo apt-get install libzbar0
```

### 2 – Install Python packages

```bash
pip install -r requirements.txt
```

> **Headless deployments** (no display, e.g. a Raspberry Pi running
> without a monitor): replace `opencv-python` with
> `opencv-python-headless` to save ~50 MB and skip GUI dependencies.

### 3 – Run

#### CLI (OpenCV window)

```bash
# Default camera, display window, all detectors enabled
python main.py

# Specific camera index
python main.py --source 1

# Video file
python main.py --source path/to/video.mp4

# Headless mode (print detections to stdout)
python main.py --headless
```

Press **q** in the display window to quit.

#### Full GUI mode (Tkinter)

```bash
# Launch the full control panel GUI
python main.py --gui
```

In GUI mode you can:
- Switch between Normal / Fast / Robust detection modes via the combobox
  (or keyboard shortcuts Ctrl+1 / Ctrl+2 / Ctrl+3).
- Toggle individual detectors (AprilTag, QR Code, Laser Spot) on and off.
- Adjust laser-detection parameters (brightness threshold, target area,
  sensitivity) with live sliders.
- Enable a **threshold overlay** to visualise which pixels are above the
  laser brightness threshold in real time.

If tkinter is missing, install it (e.g. `sudo apt install python3-tk`).

#### Operating modes

```bash
# Balanced default
python main.py --mode normal

# Faster on weaker hardware
python main.py --mode fast

# More robust for motion blur / rapid movement
python main.py --mode robust
```

#### Example: speed-oriented profile

```bash
python main.py --mode fast --no-apriltag --no-qr --width 320 --height 240
```

---

## Architecture

```
robo_eye_sense/
├── __init__.py          # Public surface: RoboEyeDetector, Detection, DetectionType
├── results.py           # Detection dataclass + DetectionType / DetectionMode enums
├── tracker.py           # CentroidTracker – assigns stable track IDs (Kalman optional)
├── april_tag_detector.py  # Wraps pupil-apriltags
├── qr_detector.py       # pyzbar (preferred) or cv2.QRCodeDetector fallback
├── laser_detector.py    # Brightness threshold + size/circularity filters
├── detector.py          # RoboEyeDetector – orchestrates all sub-detectors
├── camera.py            # VideoCapture wrapper (context manager)
└── gui.py               # Tkinter control-panel GUI (launched via --gui)
main.py                  # CLI entry point
```

### Detection pipeline (per frame)

```
Camera.read()
    │
    ▼
RoboEyeDetector.process_frame(frame)
    ├─► AprilTagDetector.detect(gray)    → List[Detection]
    ├─► QRCodeDetector.detect(frame)     → List[Detection]
    └─► LaserSpotDetector.detect(frame)  → List[Detection]
                │
                ▼
        CentroidTracker.update(detections)
                │
                ▼
        List[Detection]  (each with a stable track_id)
```

---

## Tuning for performance

| Parameter | Default | Effect |
|---|---|---|
| `--width 320 --height 240` | 640×480 | Smaller frame → faster for all detectors |
| `--laser-threshold 250` | `240` | Higher value → fewer false positives from lamps |
| `--no-apriltag` / `--no-qr` / `--no-laser` | all on | Disable unused detectors |
| `--mode fast` | `normal` | Downscales input by 50 % before detection (~4× fewer pixels) |
| `--mode robust` | `normal` | Applies unsharp-mask sharpening + Kalman-filter tracking |

---

## Programmatic usage

```python
import cv2
from robo_eye_sense import RoboEyeDetector
from robo_eye_sense.camera import Camera

detector = RoboEyeDetector(
    enable_apriltag=True,
    enable_qr=True,
    enable_laser=True,
    laser_brightness_threshold=240,
)

with Camera(source=0, width=640, height=480) as cam:
    while True:
        frame = cam.read()
        if frame is None:
            break

        detections = detector.process_frame(frame)

        for d in detections:
            print(f"{d.detection_type.value}: id={d.identifier} "
                  f"center={d.center} track={d.track_id}")

        annotated = detector.draw_detections(frame, detections)
        cv2.imshow("robo-eye-sense", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
```

---

## Running tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

> **Note:** GUI tests (``tests/test_gui.py``) require a display and the
> ``python3-tk`` package.  On headless CI machines run them via
> ``xvfb-run pytest tests/ -v``.

---

## License

See [LICENSE](LICENSE).
