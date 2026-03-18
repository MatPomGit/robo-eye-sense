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
| **Scenariusz Offset** | Kalibracja przesunięcia kamery – porównanie pozycji AprilTagów z estymacją odległości |
| **SLAM – budowanie mapy markerów** | Inkrementalna mapa 3-D z estymacją pozy 6-DoF robota (`cv2.solvePnP`) |
| **Estymacja odległości** | Model kamery otworkowej (pinhole) – odległość do każdego tagu i do pozycji referencyjnej |
| **Nagrywanie wideo** | Zapis strumienia do pliku MP4 (`cv2.VideoWriter`) – CLI, GUI i tryb bezgłowy |
| **Tryb bezgłowy (headless)** | Praca bez wyświetlacza – wyniki na stdout; idealne dla urządzeń wbudowanych |
| **Podsumowanie konfiguracji** | Automatyczne wyświetlanie parametrów startowych w terminalu |

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
python main.py --mode fast --no-apriltag --width 320 --height 240
```

#### Scenariusz Offset (kalibracja przesunięcia kamery)

```bash
# Interaktywny – ręczne ustawienie odniesienia, przesunięcie, obliczenie wektora
python main.py --scenario offset

# Bezgłowy – automatyczne przechwycenie i obliczenie
python main.py --scenario offset --headless

# Z nagrywaniem
python main.py --scenario offset --record offset_session.mp4
```

#### Scenariusz SLAM (budowanie mapy markerów)

```bash
# Interaktywny – okno OpenCV z nakładką SLAM
python main.py --scenario slam

# Bezgłowy – tylko logi pozycji robota w stdout
python main.py --scenario slam --headless

# Zapis mapy do niestandardowego pliku
python main.py --scenario slam --map-file my_map.json

# Z nagrywaniem
python main.py --scenario slam --record slam_session.mp4
```

W trybie GUI (``--gui``) scenariusze Offset i SLAM dostępne są jako
zakładki w panelu informacyjnym po prawej stronie. Zakładka **SLAM**
wyświetla:

- **Pozycję i orientację robota** (6-DoF) w czasie rzeczywistym.
- **Listę wykrytych markerów** z ich pozycjami, orientacjami oraz liczbą
  obserwacji.
- **Wizualizację 3-D** zrekonstruowanej przestrzeni (widok z góry) wraz
  z pozycją kamery.

#### Estymacja odległości (scenariusz Offset)

Scenariusz Offset automatycznie szacuje odległość od kamery do każdego
widocznego AprilTaga oraz odległość przesunięcia kamery względem pozycji
referencyjnej. Wykorzystywany jest model kamery otworkowej (pinhole):

```
distance = (tag_size_cm × focal_length_px) / apparent_size_px
```

Domyślne parametry:

| Parametr | Wartość |
|---|---|
| Rozmiar fizyczny tagu | 5,0 cm |
| Poziomy kąt widzenia (HFOV) | 60° |

Wynik (`OffsetResult`) zawiera:

- `per_tag_distances_cm` – odległość (cm) do każdego widocznego tagu.
- `distance_to_reference_cm` – szacowana odległość (cm) między bieżącą
  a referencyjną pozycją kamery.

#### Podsumowanie konfiguracji w terminalu

Przy każdym uruchomieniu program wyświetla podsumowanie aktywnej
konfiguracji w terminalu:

```
robo-eye-sense 0.3.0
Display mode      : display
Detection mode    : normal
Detectors enabled : AprilTag
Source            : 0
Camera opened     : 640x480 @ 30.0 FPS
Starting detection loop...
```

Podsumowanie obejmuje tryb wyświetlania, tryb detekcji, listę włączonych
detektorów, źródło wideo, aktywny scenariusz (jeśli ustawiony) oraz
ścieżkę nagrywania (jeśli aktywna).

---

## Konfiguracje środowiska

Poniższa tabela przedstawia wszystkie dostępne parametry wiersza poleceń
wraz z ich wartościami domyślnymi i opisem.

| Parametr | Domyślnie | Opis |
|---|---|---|
| `--source INDEX\|PATH` | `0` | Indeks kamery (0, 1, …) lub ścieżka do pliku/strumienia wideo. |
| `--width W` | `640` | Szerokość klatki w pikselach. |
| `--height H` | `480` | Wysokość klatki w pikselach. |
| `--mode normal\|fast\|robust` | `normal` | Tryb detekcji: *normal* – zrównoważony; *fast* – 50% rozdzielczości; *robust* – wyostrzanie + Kalman. |
| `--no-apriltag` | *(włączony)* | Wyłącza detekcję AprilTag. |
| `--qr` | *(wyłączony)* | Włącza detekcję kodów QR. |
| `--laser` | *(wyłączony)* | Włącza detekcję punktu lasera. |
| `--laser-threshold 0–255` | `240` | Próg jasności dla detekcji lasera. |
| `--headless` | *(wył.)* | Praca bez okna wyświetlania – wynik na stdout. |
| `--gui` | *(wył.)* | Uruchomienie pełnego GUI Tkinter (wymaga `python3-tk`). |
| `--record FILE` | *(brak)* | Nagrywanie strumienia do pliku MP4. |
| `--scenario offset\|slam` | *(brak)* | Uruchomienie scenariusza: *offset* – kalibracja przesunięcia; *slam* – budowanie mapy markerów. |
| `--map-file FILE` | `marker_map.json` | Ścieżka do pliku JSON z mapą markerów (zapis/odczyt). |

### Przykładowe konfiguracje

```bash
# 1. Detekcja domyślna – kamera 0, AprilTag włączony, okno OpenCV
python main.py

# 2. Tryb szybki, detekcja lasera, niska rozdzielczość
python main.py --mode fast --laser --width 320 --height 240

# 3. Tryb robuśny, detekcja QR + AprilTag, nagrywanie
python main.py --mode robust --qr --record session.mp4

# 4. GUI ze wszystkimi detektorami
python main.py --gui --qr --laser

# 5. SLAM bezgłowy z plikiem wideo
python main.py --scenario slam --source video.mp4 --headless --map-file map.json

# 6. Offset w trybie headless
python main.py --scenario offset --headless

# 7. Kamera USB #2, tryb robuśny, GUI
python main.py --source 2 --mode robust --gui
```

---

## Architektura

```
robo_eye_sense/
├── __init__.py            # Public surface: RoboEyeDetector, Detection, DetectionType,
│                          #   DetectionMode, SlamCalibrator, MarkerMap, MarkerPose3D,
│                          #   RobotPose3D, APP_NAME, __version__
├── results.py             # Detection dataclass + DetectionType / DetectionMode enums
├── tracker.py             # CentroidTracker – assigns stable track IDs (Kalman optional)
├── april_tag_detector.py  # Wraps pupil-apriltags (4 rodziny tagów jednocześnie)
├── qr_detector.py         # pyzbar (preferred) or cv2.QRCodeDetector fallback
├── laser_detector.py      # Brightness threshold + size/circularity filters
├── detector.py            # RoboEyeDetector – orchestrates all sub-detectors
│                          #   Public API: enable/disable_april(), enable/disable_qr(),
│                          #   enable/disable_laser(), mode property (NORMAL/FAST/ROBUST)
├── camera.py              # VideoCapture wrapper (context manager)
├── offset_scenario.py     # Camera-offset calibration + distance estimation (pinhole model)
├── marker_map.py          # SLAM marker map building + robot localisation (6-DoF)
├── recorder.py            # Video recording utility (MP4, context manager)
└── gui.py                 # Tkinter GUI (Offset + SLAM tabs, 3-D visualisation, recording)
main.py                    # CLI entry point
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
| `--no-apriltag` | *(włączony)* | Wyłączenie detektora AprilTag oszczędza czas przetwarzania |
| `--mode fast` | `normal` | Skaluje wejście o 50% przed detekcją (~4× mniej pikseli) |
| `--mode robust` | `normal` | Stosuje wyostrzanie unsharp-mask + śledzenie filtrem Kalmana |

---

## Użycie programistyczne

### Podstawowe wykrywanie

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

### Nagrywanie wideo

```python
from robo_eye_sense.recorder import VideoRecorder

with VideoRecorder("output.mp4", width=640, height=480, fps=30.0) as rec:
    for frame in frames:
        rec.write_frame(frame)
```

### Kalibracja Offset z estymacją odległości

```python
from robo_eye_sense.offset_scenario import compute_offset

result = compute_offset(reference_detections, current_detections,
                        frame_width=640, tag_size_cm=5.0)

print(f"Przesunięcie: {result.offset}")
print(f"Odległość do ref.: {result.distance_to_reference_cm} cm")
for tag_id, dist in result.per_tag_distances_cm.items():
    print(f"  Tag {tag_id}: {dist:.1f} cm")
```

### SLAM – budowanie mapy i lokalizacja

```python
from robo_eye_sense import RoboEyeDetector
from robo_eye_sense.camera import Camera
from robo_eye_sense.marker_map import SlamCalibrator, MarkerMap

detector = RoboEyeDetector(enable_apriltag=True)
calibrator = SlamCalibrator(tag_size_cm=5.0)

with Camera(source=0, width=640, height=480) as cam:
    for _ in range(300):
        frame = cam.read()
        if frame is None:
            break
        detections = detector.process_frame(frame)
        robot_pose = calibrator.process_detections(detections)
        print(f"Robot: {robot_pose.position}, mapa: {len(calibrator.marker_map)} markerów")

calibrator.marker_map.save("mapa.json")

# Późniejsza lokalizacja z istniejącą mapą
marker_map = MarkerMap.load("mapa.json")
pose = marker_map.estimate_robot_pose(detections, tag_size_cm=5.0)
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

## Dodatkowa dokumentacja

| Dokument | Opis |
|---|---|
| [slam_marker_map.md](slam_marker_map.md) | Szczegółowy opis algorytmu SLAM, budowania mapy markerów i lokalizacji robota |
| [kalibracja_kamery.md](kalibracja_kamery.md) | Kompletna instrukcja kalibracji kamery (intrinsyki, dystorsja, szachownica) |
| [markery_wizyjne_nawigacja.md](markery_wizyjne_nawigacja.md) | Przewodnik po markerach wizyjnych w nawigacji robotów |
| [camera-spatial-orientation.md](camera-spatial-orientation.md) | Orientacja przestrzenna kamery |
| [github_releases_packages.md](github_releases_packages.md) | Przewodnik po GitHub Releases i Packages |
| [CHANGELOG.md](CHANGELOG.md) | Historia zmian i wydań |

---

## Licencja

Zobacz [LICENSE](LICENSE).
