# robo-vision

Lekkie wykrywanie wizualnych markerów w czasie rzeczywistym dla robotów mobilnych.

**robo-eye-sense** łączy trzy uzupełniające się technologie detekcji —
znaczniki fiducjalne AprilTag, kody QR oraz punkty lasera — w jeden,
zunifikowany potok wykrywania. Każdemu wykrytemu obiektowi nadawane jest
*stabilne ID śledzenia*, które utrzymuje się przez klatki, nawet przy krótkich
zasłonięciach. System zawiera pełne GUI Tkinter do interaktywnego strojenia
oraz bezgłowy interfejs CLI do wdrożeń na sprzęcie wbudowanym.

## Funkcje

| Możliwość | Technologia |
|---|---|
| **Detekcja i śledzenie AprilTag** | [pupil-apriltags](https://github.com/pupil-labs/apriltags) |
| **Detekcja i dekodowanie kodów QR** | [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar) *(preferowane)* / alternatywa OpenCV |
| **Detekcja punktu lasera** | Progowanie jasności OpenCV + filtr kołowości |
| **Śledzenie wielu obiektów** | Tracker centroidów z trwałymi ID (tryb filtru Kalmana dostępny) |
| **Interaktywne GUI** | Panel sterowania Tkinter – zmiana trybu, przełączanie detektorów, strojenie parametrów lasera na żywo |
| **Trzy tryby pracy** | Normal / Fast (połowa rozdzielczości) / Robust (wyostrzanie + śledzenie Kalmana) |

Wszystkie detektory działają na każdej klatce, a ich wyniki są ujednolicane przez
pojedynczy `CentroidTracker` przydzielający stabilne ID śledzenia. Obiekty
oznaczone etykietą (AprilTags, kody QR) są dopasowywane według ich tożsamości
semantycznej, dzięki czemu ten sam fizyczny marker zawsze zachowuje to samo ID,
nawet po krótkim zasłonięciu. Obiekty bez etykiety (punkty lasera) są
dopasowywane według odległości najbliższego centroidu, z opcjonalną predykcją
filtru Kalmana w trybie ROBUST.

---

## Szybki start

### 1 – Instalacja zależności systemowych

```bash
# Ubuntu / Debian (required for QR-code detection via pyzbar)
sudo apt-get install libzbar0
```

### 2 – Instalacja pakietów Python

```bash
pip install -r requirements.txt
```

> **Wdrożenia bezgłowe** (bez wyświetlacza, np. Raspberry Pi bez monitora):
> zamień `opencv-python` na `opencv-python-headless`, aby zaoszczędzić ~50 MB
> i pominąć zależności GUI.

### 3 – Uruchomienie

#### CLI (okno OpenCV)

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

Naciśnij **q** w oknie podglądu, aby zamknąć program.

#### Pełny tryb GUI (Tkinter)

```bash
# Launch the full control panel GUI
python main.py --gui
```

W trybie GUI możesz:
- Przełączać się między trybami detekcji Normal / Fast / Robust za pomocą listy
  rozwijanej (lub skrótów klawiszowych Ctrl+1 / Ctrl+2 / Ctrl+3).
- Włączać i wyłączać poszczególne detektory (AprilTag, QR Code, Laser Spot).
- Dostosowywać parametry detekcji lasera (próg jasności, docelowy obszar,
  czułość) za pomocą suwaków na żywo.
- Włączyć **nakładkę progową**, aby wizualizować w czasie rzeczywistym, które
  piksele przekraczają próg jasności lasera.

Jeśli brakuje tkinter, zainstaluj go (np. `sudo apt install python3-tk`).

#### Tryby pracy

```bash
# Balanced default
python main.py --mode normal

# Faster on weaker hardware
python main.py --mode fast

# More robust for motion blur / rapid movement
python main.py --mode robust
```

#### Przykład: profil zorientowany na szybkość

```bash
python main.py --mode fast --no-apriltag --no-qr --width 320 --height 240
```

---

## Architektura

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

### Potok detekcji (na klatkę)

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

## Strojenie wydajności

| Parametr | Domyślnie | Efekt |
|---|---|---|
| `--width 320 --height 240` | 640×480 | Mniejsza klatka → szybsze działanie wszystkich detektorów |
| `--laser-threshold 250` | `240` | Wyższa wartość → mniej fałszywych alarmów od lamp |
| `--no-apriltag` / `--no-qr` / `--no-laser` | wszystkie włączone | Wyłączenie nieużywanych detektorów |
| `--mode fast` | `normal` | Skaluje wejście o 50% przed detekcją (~4× mniej pikseli) |
| `--mode robust` | `normal` | Stosuje wyostrzanie unsharp-mask + śledzenie filtrem Kalmana |

---

## Użycie programistyczne

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

## Uruchamianie testów

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

> **Uwaga:** Testy GUI (`tests/test_gui.py`) wymagają wyświetlacza i pakietu
> `python3-tk`. Na maszynach CI bez wyświetlacza uruchom je przez
> `xvfb-run pytest tests/ -v`.

---

## Licencja

Zobacz [LICENSE](LICENSE).
