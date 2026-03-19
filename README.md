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
| **Kalibracja kamery** | Wyznaczanie parametrów wewnętrznych kamery na podstawie wzorca szachownicy (`cv2.calibrateCamera`) |
| **Detekcja pudełek (box)** | Wykrywanie prostokątnych obiektów (pudełka, prostopadłościany) w czasie rzeczywistym |
| **Estymacja pozy (pose)** | Estymacja pozy 6-DoF każdego znacznika AprilTag (`cv2.solvePnP`) |
| **Śledzenie markera (follow)** | Aktywne śledzenie AprilTag / pudełka z generowaniem sygnałów sterowania robota |
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
python main.py --quality normal

# Faster on weaker hardware
python main.py --quality low

# More robust for motion blur / rapid movement
python main.py --quality high
```

#### Przykład: profil zorientowany na szybkość

```bash
python main.py --quality low --no-apriltag --width 320 --height 240
```

#### Scenariusz Offset (kalibracja przesunięcia kamery)

```bash
# Interaktywny – ręczne ustawienie odniesienia, przesunięcie, obliczenie wektora
python main.py --mode offset

# Bezgłowy – automatyczne przechwycenie i obliczenie
python main.py --mode offset --headless

# Z nagrywaniem
python main.py --mode offset --record offset_session.mp4
```

#### Scenariusz SLAM (budowanie mapy markerów)

```bash
# Interaktywny – okno OpenCV z nakładką SLAM
python main.py --mode slam

# Bezgłowy – tylko logi pozycji robota w stdout
python main.py --mode slam --headless

# Zapis mapy do niestandardowego pliku
python main.py --mode slam --map-file my_map.json

# Z nagrywaniem
python main.py --mode slam --record slam_session.mp4

# Z plikiem kalibracji kamery (lepsza dokładność SLAM)
python main.py --mode slam --calib-output calibration.npz --tag-size 0.05
```

W trybie GUI (``--gui``) scenariusze Offset i SLAM dostępne są jako
zakładki w panelu informacyjnym po prawej stronie. Zakładka **SLAM**
wyświetla:

- **Pozycję i orientację robota** (6-DoF) w czasie rzeczywistym.
- **Listę wykrytych markerów** z ich pozycjami, orientacjami oraz liczbą
  obserwacji.
- **Wizualizację 3-D** zrekonstruowanej przestrzeni (widok z góry) wraz
  z pozycją kamery.

#### Tryb kalibracji kamery (calibration)

Tryb *calibration* pozwala wyznaczyć parametry wewnętrzne kamery
(macierz kamery, współczynniki dystorsji) na podstawie wzorca szachownicy.

```bash
# Domyślna szachownica 9×6, zapis do calibration.npz
python main.py --mode calibration

# Niestandardowy rozmiar szachownicy i ścieżka wyjściowa
python main.py --mode calibration --chessboard-size 7x5 --calib-output my_calib.npz

# Kalibracja na pliku wideo
python main.py --mode calibration --source calibration_video.mp4
```

W trybie kalibracji naciśnij **spację**, aby przechwycić klatkę z wykrytym
wzorcem. Program wymaga 15–25 poprawnych ujęć, po czym automatycznie
oblicza parametry kamery i zapisuje je do pliku `.npz`.

#### Tryb detekcji pudełek (box)

Tryb *box* wykrywa prostokątne obiekty (pudełka, prostopadłościany)
w czasie rzeczywistym przy użyciu potoku detekcji krawędziowej.

```bash
# Detekcja pudełek z domyślnej kamery
python main.py --mode box

# Bezgłowy – wyniki na stdout
python main.py --mode box --headless

# Z nagrywaniem sesji
python main.py --mode box --record box_session.mp4
```

#### Tryb estymacji pozy (pose)

Tryb *pose* szacuje pozę 6-DoF (pozycja + orientacja) każdego widocznego
znacznika AprilTag za pomocą `cv2.solvePnP`.

```bash
# Estymacja pozy z domyślnym rozmiarem tagu (5 cm)
python main.py --mode pose

# Z niestandardowym rozmiarem tagu i plikiem kalibracji
python main.py --mode pose --tag-size 0.10 --calib-output calibration.npz

# Bezgłowy
python main.py --mode pose --headless --tag-size 0.05
```

#### Tryb śledzenia (follow)

Tryb *follow* aktywnie śledzi wybrany znacznik AprilTag (lub pudełko
jako awaryjny cel) i generuje sygnały sterowania (prędkość liniowa
i kątowa) dla robota mobilnego.

```bash
# Śledzenie pierwszego widocznego markera
python main.py --mode follow

# Śledzenie konkretnego markera (ID = 5)
python main.py --mode follow --follow-marker 5

# Z rezerwowym śledzeniem pudełka, gdy brak tagów
python main.py --mode follow --follow-box

# Z ustawioną odległością docelową (1 m) i rozmiarem tagu
python main.py --mode follow --target-distance 1.0 --tag-size 0.05

# Bezgłowy z kalibracją kamery
python main.py --mode follow --headless --calib-output calibration.npz
```

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
robo-eye-sense 0.4.0
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
| `--quality low\|normal\|high` | `normal` | Jakość detekcji: *low* – 50% rozdzielczości (szybkie); *normal* – zrównoważony; *high* – wyostrzanie + Kalman (odporny). |
| `--mode basic\|offset\|slam\|calibration\|box\|pose\|follow` | `basic` | Tryb pracy: *basic* – standardowa pętla detekcji; *offset* – kalibracja przesunięcia kamery; *slam* – budowanie mapy markerów; *calibration* – kalibracja intrinsik kamery; *box* – detekcja pudełek; *pose* – estymacja pozy 6-DoF tagów; *follow* – śledzenie markera z generowaniem sygnałów sterowania. |
| `--no-apriltag` | *(włączony)* | Wyłącza detekcję AprilTag. |
| `--qr` | *(wyłączony)* | Włącza detekcję kodów QR. |
| `--laser` | *(wyłączony)* | Włącza detekcję punktu lasera. |
| `--laser-threshold 0–255` | `240` | Dolny próg jasności dla detekcji lasera. |
| `--laser-threshold-max 0–255` | `255` | Górny próg jasności dla detekcji lasera (zakres min–max). |
| `--laser-channels r\|g\|b\|rgb` | `rgb` | Kanały kolorów do analizy detekcji lasera (np. `r` – tylko czerwony). |
| `--headless` | *(wył.)* | Praca bez okna wyświetlania – wynik na stdout. |
| `--gui` | *(wył.)* | Uruchomienie pełnego GUI Tkinter (wymaga `python3-tk`). |
| `--record FILE` | *(brak)* | Nagrywanie strumienia do pliku MP4. |
| `--map-file FILE` | `marker_map.json` | Ścieżka do pliku JSON z mapą markerów SLAM (zapis/odczyt). |
| `--tag-size METRES` | `0.05` | Fizyczny rozmiar boku znacznika AprilTag w metrach (tryby slam/pose/follow). |
| `--tag-names ID=NAME …` | *(brak)* | Nazwy czytelne dla człowieka dla ID tagów, np. `--tag-names 1=box 2=table`. |
| `--follow-marker ID` | *(brak)* | W trybie *follow*: ID tagu AprilTag do śledzenia (domyślnie pierwszy widoczny). |
| `--follow-box` | *(wył.)* | W trybie *follow*: cofnięcie do śledzenia pudełka gdy nie ma tagów. |
| `--target-distance M` | `0.5` | Żądana odległość do celu w metrach (tryb follow). |
| `--chessboard-size COLSxROWS` | `9x6` | Wymiary wewnętrznych narożników szachownicy kalibracyjnej. |
| `--calib-output FILE` | `calibration.npz` | Plik NPZ z danymi kalibracji kamery (zapis/odczyt). |
| `--info` | *(wył.)* | Wyświetla informacje o kamerze (rozdzielczość, FPS, backend) i kończy działanie. |

### Parametry trybu headless

W trybie `--headless` program nie otwiera żadnego okna graficznego i wypisuje
wszystkie informacje na standardowe wyjście (stdout). Możesz używać
dowolnego parametru z tabeli powyżej. Parametry szczególnie przydatne
w trybie headless:

| Parametr | Typowe użycie headless |
|---|---|
| `--source` | Wskaż plik wideo lub strumień (np. RTSP), bo kamera fizyczna może nie być dostępna. |
| `--mode slam` | Budowanie mapy markerów – logi pozycji robota per klatka w stdout. |
| `--mode offset` | Kalibracja przesunięcia – wyniki wypisywane po przetworzeniu. |
| `--quality low` | Szybsze przetwarzanie na słabym sprzęcie wbudowanym. |
| `--record FILE` | Zapis wideo nawet w trybie headless (bez podglądu). |
| `--map-file FILE` | Zapis/wczytanie mapy SLAM do/z pliku JSON. |
| `--calib-output FILE` | Użycie pliku kalibracji w trybie SLAM dla lepszej dokładności. |
| `--tag-size METRES` | Dokładny rozmiar fizyczny tagu – wpływa na estymację odległości i SLAM. |
| `--tag-names ID=NAME` | Czytelne etykiety w logach (np. `--tag-names 1=robot 2=goal`). |
| `--no-apriltag` / `--qr` / `--laser` | Włączanie/wyłączanie detektorów stosownie do sceny. |

Przykład kompleksowego uruchomienia headless (SLAM z nagrywaniem i kalibracją):

```bash
python main.py \
  --headless \
  --mode slam \
  --source /dev/video0 \
  --quality normal \
  --tag-size 0.05 \
  --calib-output calibration.npz \
  --map-file my_map.json \
  --record slam_session.mp4
```

### Przykładowe konfiguracje

```bash
# 1. Detekcja domyślna – kamera 0, AprilTag włączony, okno OpenCV
python main.py

# 2. Tryb szybki, detekcja lasera, niska rozdzielczość
python main.py --quality low --laser --width 320 --height 240

# 3. Tryb robuśny, detekcja QR + AprilTag, nagrywanie
python main.py --quality high --qr --record session.mp4

# 4. GUI ze wszystkimi detektorami
python main.py --gui --qr --laser

# 5. SLAM bezgłowy z plikiem wideo
python main.py --mode slam --source video.mp4 --headless --map-file map.json

# 6. Offset w trybie headless
python main.py --mode offset --headless

# 7. Kamera USB #2, tryb robuśny, GUI
python main.py --source 2 --quality high --gui

# 8. Kalibracja kamery szachownicą
python main.py --mode calibration --chessboard-size 9x6 --calib-output cam.npz

# 9. Detekcja pudełek z nagrywaniem
python main.py --mode box --record boxes.mp4

# 10. Estymacja pozy tagów z plikiem kalibracji
python main.py --mode pose --tag-size 0.10 --calib-output cam.npz

# 11. Śledzenie markera ID=3 z rezerwą na pudełko
python main.py --mode follow --follow-marker 3 --follow-box --target-distance 0.8
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
├── auto_scenario.py       # Auto-follow scenario – śledzenie markera z wektorem sterowania
├── recorder.py            # Video recording utility (MP4, context manager)
└── gui.py                 # Tkinter GUI (Offset + SLAM tabs, 3-D visualisation, recording)
modes/
├── __init__.py            # Exportuje BaseMode, CalibrationMode, BoxMode, PoseMode, FollowMode
├── base.py                # BaseMode – abstrakcyjna klasa bazowa (run(frame, context))
├── calibration_mode.py    # CalibrationMode – kalibracja kamery szachownicą
├── box_mode.py            # BoxMode – detekcja prostokątnych obiektów
├── pose_mode.py           # PoseMode – estymacja pozy 6-DoF tagów AprilTag
└── follow_mode.py         # FollowMode – śledzenie markera z sygnałami sterowania
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
| `--quality low` | `normal` | Skaluje wejście o 50% przed detekcją (~4× mniej pikseli) |
| `--quality high` | `normal` | Stosuje wyostrzanie unsharp-mask + śledzenie filtrem Kalmana |

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
