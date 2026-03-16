# robo-eye-sense

Lightweight real-time visual marker detection for mobile robots.

## Features

| Capability | Technology |
|---|---|
| **AprilTag detection & tracking** | [pupil-apriltags](https://github.com/pupil-labs/apriltags) |
| **QR-code detection & decoding** | [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar) *(preferred)* / OpenCV fallback |
| **Laser-spot detection** | OpenCV brightness thresholding + circularity filter |
| **Multi-object tracking** | Centroid tracker with persistent IDs |

All detectors run on every frame and their results are unified through a
single `CentroidTracker` that assigns stable track IDs.  Labeled objects
(AprilTags, QR codes) are matched by their semantic identity so the same
physical marker always keeps the same track ID even after a brief occlusion.

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

In GUI mode you can change detection mode and toggle detectors at runtime.
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
├── results.py           # Detection dataclass + DetectionType enum
├── tracker.py           # CentroidTracker – assigns stable track IDs
├── april_tag_detector.py  # Wraps pupil-apriltags
├── qr_detector.py       # pyzbar (preferred) or cv2.QRCodeDetector fallback
├── laser_detector.py    # Brightness threshold + size/circularity filters
├── detector.py          # RoboEyeDetector – orchestrates all sub-detectors
└── camera.py            # VideoCapture wrapper (context manager)
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

---

## License

See [LICENSE](LICENSE).