# Automatyczna kalibracja SLAM i budowanie mapy markerów

Dokument opisuje, w jaki sposób przeprowadzić automatyczną kalibrację za pomocą
algorytmów SLAM (Simultaneous Localisation and Mapping) w systemie
**robo-eye-sense** oraz jak zbudować i wykorzystać mapę markerów (ang. *marker
map* / *tag map*), która definiuje położenie i orientację każdego markera
w globalnym układzie współrzędnych, a także pozycję samego robota
w trójwymiarowej przestrzeni.

---

## Spis treści

1. [Czym jest SLAM oparty na markerach](#1-czym-jest-slam-oparty-na-markerach)
2. [Architektura modułu `marker_map`](#2-architektura-modułu-marker_map)
   1. [Struktury danych](#21-struktury-danych)
   2. [Klasa `MarkerMap`](#22-klasa-markermap)
   3. [Klasa `SlamCalibrator`](#23-klasa-slamcalibrator)
3. [Estymacja pozy markera — `cv2.solvePnP`](#3-estymacja-pozy-markera--cv2solvepnp)
4. [Algorytm budowania mapy](#4-algorytm-budowania-mapy)
   1. [Inicjalizacja — pierwsza klatka](#41-inicjalizacja--pierwsza-klatka)
   2. [Lokalizacja robota](#42-lokalizacja-robota)
   3. [Dodawanie nowych markerów](#43-dodawanie-nowych-markerów)
   4. [Uśrednianie obserwacji](#44-uśrednianie-obserwacji)
5. [Lokalizacja robota na podstawie mapy](#5-lokalizacja-robota-na-podstawie-mapy)
6. [Serializacja mapy — JSON](#6-serializacja-mapy--json)
7. [Krok po kroku — tryb CLI](#7-krok-po-kroku--tryb-cli)
   1. [Budowanie mapy (scenariusz `slam`)](#71-budowanie-mapy-scenariusz-slam)
   2. [Korzystanie z gotowej mapy](#72-korzystanie-z-gotowej-mapy)
8. [Krok po kroku — użycie programistyczne (API)](#8-krok-po-kroku--użycie-programistyczne-api)
   1. [Budowanie mapy z kodu Python](#81-budowanie-mapy-z-kodu-python)
   2. [Lokalizacja robota z istniejącą mapą](#82-lokalizacja-robota-z-istniejącą-mapą)
9. [Rola kalibracji kamery](#9-rola-kalibracji-kamery)
10. [Dobre praktyki i ograniczenia](#10-dobre-praktyki-i-ograniczenia)
11. [Podsumowanie](#11-podsumowanie)

---

## 1. Czym jest SLAM oparty na markerach

SLAM (*Simultaneous Localisation and Mapping*) to rodzina algorytmów, które
pozwalają robotowi jednocześnie:

- **budować mapę otoczenia** — rejestrować położenie istotnych punktów
  odniesienia (w naszym przypadku: markerów fiducjalnych),
- **lokalizować się w tej mapie** — wyznaczać swoją bieżącą pozycję
  i orientację (6 stopni swobody) na podstawie obserwowanych markerów.

W klasycznym Visual SLAM punktami odniesienia są cechy naturalne obrazu
(narożniki, krawędzie, deskryptory ORB/SIFT). W przypadku **marker-based
SLAM** rolę tę pełnią sztuczne znaczniki — AprilTagi, ArUco lub inne markery
fiducjalne — które mają trzy kluczowe zalety:

| Cecha | Korzyść |
|---|---|
| Unikalny identyfikator | Natychmiastowe skojarzenie danych (ang. *data association*) |
| Znana geometria i rozmiar fizyczny | Dokładna estymacja pozy 6-DoF bez dodatkowej kalibracji skali |
| Wysoka powtarzalność detekcji | Mniejsza liczba fałszywych dopasowań niż w przypadku cech naturalnych |

System **robo-eye-sense** wykorzystuje bibliotekę `pupil-apriltags` do
detekcji AprilTagów. Moduł `marker_map` rozszerza tę funkcjonalność o pełny
pipeline SLAM-owy: budowanie mapy i lokalizację robota w 3-D.

---

## 2. Architektura modułu `marker_map`

Moduł `robo_eye_sense/marker_map.py` składa się z następujących elementów:

```
marker_map.py
├── MarkerPose3D      – pozycja i orientacja jednego markera (dataclass)
├── RobotPose3D       – pozycja i orientacja robota (dataclass)
├── MarkerMap          – kolekcja MarkerPose3D + serializacja JSON
├── SlamCalibrator     – inkrementalny builder mapy (główny algorytm)
└── funkcje pomocnicze
    ├── _solve_marker_pose()    – estymacja pozy markera (solvePnP)
    ├── _euler_to_rotation_matrix()
    ├── _rotation_matrix_to_euler()
    ├── _angle_average()
    └── _mean_angles()
```

### 2.1. Struktury danych

**`MarkerPose3D`** przechowuje pozycję i orientację jednego markera
w globalnym układzie współrzędnych:

```python
@dataclass
class MarkerPose3D:
    marker_id: str                              # np. "42"
    position: Tuple[float, float, float]        # (x, y, z) w cm
    orientation: Tuple[float, float, float]     # (roll, pitch, yaw) w stopniach
    observations: int                           # ile obserwacji złożyło się na tę estymację
```

**`RobotPose3D`** przechowuje estymowaną pozycję robota (kamery):

```python
@dataclass
class RobotPose3D:
    position: Tuple[float, float, float]        # (x, y, z) w cm
    orientation: Tuple[float, float, float]     # (roll, pitch, yaw) w stopniach
    visible_markers: int                        # ile markerów z mapy było widocznych
    reprojection_error: Optional[float]         # błąd reprojekcji w pikselach
```

Obie struktury **nie zależą od OpenCV** — można je importować i używać nawet
w środowiskach bez `cv2`.

### 2.2. Klasa `MarkerMap`

`MarkerMap` to kolekcja wpisów `MarkerPose3D` indeksowana po `marker_id`.
Oferuje:

| Metoda | Opis |
|---|---|
| `add(pose)` | Dodaj lub nadpisz wpis markera |
| `get(marker_id)` | Pobierz wpis (lub `None`) |
| `remove(marker_id)` | Usuń marker z mapy |
| `merge_observation(id, pos, ori)` | Dodaj obserwację ze średnią kroczącą |
| `save(path)` / `load(path)` | Zapis / odczyt pliku JSON |
| `to_dict()` / `from_dict(data)` | Serializacja do/z słownika |
| `estimate_robot_pose(detections, …)` | Estymuj pozycję robota |

### 2.3. Klasa `SlamCalibrator`

`SlamCalibrator` implementuje pełny pipeline SLAM-owy. Wystarczy podawać mu
kolejne zestawy detekcji (wynik `RoboEyeDetector.process_frame()`), a on
sam buduje mapę i zwraca bieżącą pozycję robota:

```python
calibrator = SlamCalibrator(tag_size_cm=5.0)

for frame in camera_stream:
    detections = detector.process_frame(frame)
    robot_pose = calibrator.process_detections(detections)
    # robot_pose.position → (x, y, z) w cm
    # calibrator.marker_map → bieżąca mapa markerów

calibrator.marker_map.save("mapa.json")
```

---

## 3. Estymacja pozy markera — `cv2.solvePnP`

Kluczowym krokiem algorytmu jest wyznaczenie **6-DoF pozy** (3 translacje +
3 rotacje) każdego widocznego markera względem kamery. Wykorzystujemy do tego
funkcję OpenCV `solvePnP`, która rozwiązuje problem Perspektywy-n-Punktów
(PnP):

1. **Punkty 3-D w układzie markera** — znane z fizycznego rozmiaru tagu.
   Dla markera o boku `s` cm definiujemy cztery narożniki:
   ```
   (-s/2, -s/2, 0),  (s/2, -s/2, 0),
   (s/2,  s/2,  0),  (-s/2, s/2,  0)
   ```

2. **Punkty 2-D w obrazie** — współrzędne pikseli narożników zwrócone
   przez detektor AprilTag (`Detection.corners`).

3. **Macierz kamery (intrinsics)** — parametry wewnętrzne kamery
   (ogniskowa, punkt główny). Jeśli nie podano kalibracji, moduł
   konstruuje przybliżoną macierz na podstawie rozmiaru obrazu.

4. **Współczynniki dystorsji** — opcjonalne; domyślnie zerowe.

Wynikiem jest wektor rotacji `rvec` (Rodrigues) i wektor translacji `tvec`,
opisujące pozycję markera **w układzie kamery**.

```
             kamera (początek układu)
                 |
                 | tvec (translacja)
                 ▼
            ┌─────────┐
            │ AprilTag │  ← rvec opisuje obrót markera
            └─────────┘
```

Moduł automatycznie oblicza również **błąd reprojekcji** — średnią odległość
między rzutowanymi punktami 3-D a rzeczywistymi współrzędnymi narożników
w obrazie. Niski błąd reprojekcji (< 2 px) świadczy o dobrej estymacji.

---

## 4. Algorytm budowania mapy

### 4.1. Inicjalizacja — pierwsza klatka

Algorytm przyjmuje, że w pierwszej klatce kamera znajduje się w **początku
układu współrzędnych świata** `(0, 0, 0)` i jest skierowana wzdłuż osi +Z.

Każdy marker wykryty w pierwszej klatce jest umieszczany w mapie bezpośrednio
na podstawie pozy kamery-marker (`tvec`, `rvec`) zwróconej przez `solvePnP`:

```
Świat:  marker_world_pos = tvec_camera_marker
        marker_world_ori = R_camera_marker
```

### 4.2. Lokalizacja robota

W kolejnych klatkach kamera mogła się przemieścić. Aby wyznaczyć jej nową
pozycję, wykorzystujemy markery **już obecne w mapie**:

1. Dla każdego widocznego markera `m`, który istnieje w mapie, wyznaczamy
   jego pozę `(rvec_m, tvec_m)` względem kamery.
2. Odwracamy tę transformację, aby uzyskać pozycję kamery w układzie markera:
   ```
   R_mc = R_cm^T
   t_mc = -R_mc · tvec_m
   ```
3. Przechodzimy do układu świata, wykorzystując znaną pozycję markera
   w mapie:
   ```
   cam_world = R_mw · t_mc + marker_world_position
   ```
4. Jeśli widocznych jest wiele markerów, wyniki są **uśredniane** (z użyciem
   średniej kołowej dla kątów), co zwiększa odporność na szum pomiarowy.

### 4.3. Dodawanie nowych markerów

Markery widziane po raz pierwszy są umieszczane w mapie na podstawie
aktualnej estymowanej pozycji robota:

```
marker_world_pos = R_rw · tvec_camera_marker + robot_world_position
marker_world_ori = R_rw · R_camera_marker
```

Dzięki temu mapa rośnie inkrementalnie — robot odkrywa nowe markery
w miarę eksploracji przestrzeni.

### 4.4. Uśrednianie obserwacji

Wielokrotne obserwacje tego samego markera są łączone za pomocą **średniej
kroczącej** (ang. *running average*). Każda nowa obserwacja z wagą 1 jest
uśredniana z dotychczasową estymacją mającą wagę równą liczbie
dotychczasowych obserwacji:

```
position_new = (position_old * n + observation) / (n + 1)
```

Dla kątów stosowana jest **średnia kołowa** (uwzględnia zawijanie 360°→0°),
aby uniknąć artefaktów na granicy 0°/360°.

Im więcej obserwacji danego markera, tym bardziej stabilna i dokładna jest
jego estymowana pozycja w mapie.

---

## 5. Lokalizacja robota na podstawie mapy

Gdy mapa markerów jest już zbudowana (lub wczytana z pliku JSON), można ją
wykorzystać do **ciągłej lokalizacji robota** w czasie rzeczywistym:

```python
from robo_eye_sense.marker_map import MarkerMap

marker_map = MarkerMap.load("mapa.json")

# W pętli głównej:
robot_pose = marker_map.estimate_robot_pose(
    detections,
    camera_matrix=camera_intrinsics,   # opcjonalne
    tag_size_cm=5.0,
)

print(robot_pose.position)          # (x, y, z) w cm
print(robot_pose.orientation)       # (roll, pitch, yaw) w stopniach
print(robot_pose.visible_markers)   # ile markerów z mapy było widocznych
print(robot_pose.reprojection_error) # błąd reprojekcji
```

Metoda `estimate_robot_pose()`:

1. Filtruje detekcje — bierze pod uwagę tylko AprilTagi z ≥ 4 narożnikami,
   których identyfikatory istnieją w mapie.
2. Dla każdego takiego markera wyznacza pozę kamery w układzie świata
   (odwracając transformację marker→kamera i łącząc ze znaną pozycją
   markera w mapie).
3. Uśrednia wyniki ze wszystkich widocznych markerów.
4. Zwraca `RobotPose3D` z pozycją, orientacją, liczbą użytych markerów
   i błędem reprojekcji.

---

## 6. Serializacja mapy — JSON

Mapa markerów może być zapisana i odczytana w formacie JSON:

```python
# Zapis
marker_map.save("mapa.json")

# Odczyt
marker_map = MarkerMap.load("mapa.json")
```

Przykładowa zawartość pliku:

```json
{
  "markers": [
    {
      "marker_id": "1",
      "position": [0.0, 0.0, 25.3],
      "orientation": [0.5, -1.2, 3.7],
      "observations": 15
    },
    {
      "marker_id": "5",
      "position": [30.0, 0.0, 25.0],
      "orientation": [0.0, 0.0, 0.0],
      "observations": 8
    }
  ]
}
```

Serializacja pozwala:
- **zachować wyniki kalibracji** — nie trzeba powtarzać budowania mapy
  przy każdym starcie robota,
- **przenosić mapę** między maszynami — np. skalibrować na stacji
  roboczej, a wgrać na robota,
- **edytować ręcznie** — poprawić współrzędne konkretnych markerów,
  jeśli ich fizyczne położenie uległo zmianie.

---

## 7. Krok po kroku — tryb CLI

### 7.1. Budowanie mapy (scenariusz `slam`)

Aby zbudować mapę markerów z linii poleceń:

```bash
# Kamera na żywo — robot powoli ogląda otoczenie z markerami
python main.py --scenario slam --headless --map-file mapa.json

# Z pliku wideo
python main.py --scenario slam --headless --source nagranie.mp4 --map-file mapa.json

# Z podglądem na ekranie (bez --headless)
python main.py --scenario slam --map-file mapa.json

# Z jednoczesnym nagrywaniem
python main.py --scenario slam --map-file mapa.json --record kalibracja.mp4
```

Po zakończeniu (klawisz `q` lub koniec wideo) program wyświetla podsumowanie:

```
==================================================
SLAM calibration result
==================================================
Frames processed  : 245
Markers in map    : 6
  tag    1: pos=(+0.0, +0.0, +25.3) cm  ori=(+0.5, -1.2, +3.7)°  obs=15
  tag    2: pos=(+30.0, +0.0, +25.0) cm  ori=(+0.0, +0.0, +0.0)°  obs=12
  tag    5: pos=(-15.0, +10.0, +24.8) cm  ori=(+0.1, -0.5, +1.2)°  obs=8
  ...
Map saved to      : mapa.json
==================================================
```

### 7.2. Korzystanie z gotowej mapy

Po zbudowaniu mapy można ją wykorzystać w kodzie Python (zob. [sekcja 8.2](#82-lokalizacja-robota-z-istniejącą-mapą))
lub wczytać programowo:

```bash
python -c "
from robo_eye_sense.marker_map import MarkerMap
m = MarkerMap.load('mapa.json')
print(f'Markers: {len(m)}')
for mp in m.markers():
    print(f'  {mp.marker_id}: pos={mp.position}')
"
```

---

## 8. Krok po kroku — użycie programistyczne (API)

### 8.1. Budowanie mapy z kodu Python

```python
from robo_eye_sense import RoboEyeDetector
from robo_eye_sense.camera import Camera
from robo_eye_sense.marker_map import SlamCalibrator

# Inicjalizacja
detector = RoboEyeDetector(enable_apriltag=True)
camera = Camera(source=0, width=640, height=480)
calibrator = SlamCalibrator(tag_size_cm=5.0)

# Opcjonalnie: macierz kamery z kalibracji
# calibrator = SlamCalibrator(
#     tag_size_cm=5.0,
#     camera_matrix=[
#         [fx,  0, cx],
#         [ 0, fy, cy],
#         [ 0,  0,  1],
#     ],
# )

with camera:
    for _ in range(300):          # 300 klatek ≈ 10 s przy 30 FPS
        frame = camera.read()
        if frame is None:
            break
        detections = detector.process_frame(frame)
        robot_pose = calibrator.process_detections(detections)
        print(f"Robot: {robot_pose.position}, Map size: {len(calibrator.marker_map)}")

# Zapisz mapę
calibrator.marker_map.save("mapa.json")
print(f"Gotowe — {len(calibrator.marker_map)} markerów w mapie.")
```

### 8.2. Lokalizacja robota z istniejącą mapą

```python
from robo_eye_sense import RoboEyeDetector
from robo_eye_sense.camera import Camera
from robo_eye_sense.marker_map import MarkerMap

# Wczytaj wcześniej zbudowaną mapę
marker_map = MarkerMap.load("mapa.json")
detector = RoboEyeDetector(enable_apriltag=True)
camera = Camera(source=0, width=640, height=480)

with camera:
    while True:
        frame = camera.read()
        if frame is None:
            break
        detections = detector.process_frame(frame)
        pose = marker_map.estimate_robot_pose(detections, tag_size_cm=5.0)

        if pose.visible_markers > 0:
            x, y, z = pose.position
            roll, pitch, yaw = pose.orientation
            print(f"Pozycja: ({x:.1f}, {y:.1f}, {z:.1f}) cm")
            print(f"Orientacja: roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°")
            print(f"Widoczne markery: {pose.visible_markers}")
            print(f"Błąd reprojekcji: {pose.reprojection_error:.2f} px")
        else:
            print("Brak widocznych markerów z mapy.")
```

---

## 9. Rola kalibracji kamery

Dokładność estymacji pozy (zarówno markerów, jak i robota) **znacząco zależy**
od jakości parametrów wewnętrznych kamery (macierz intrinsyków):

```
        ┌ fx   0   cx ┐
K  =    │  0  fy   cy │
        └  0   0    1 ┘
```

| Parametr | Znaczenie |
|---|---|
| `fx`, `fy` | Ogniskowa w pikselach (oś X i Y) |
| `cx`, `cy` | Punkt główny (środek optyczny) |

**Bez kalibracji** moduł stosuje przybliżenie oparte na rozmiarze obrazu,
co działa akceptowalnie dla standardowych kamer internetowych, ale wprowadza
błędy rzędu kilku–kilkunastu procent.

**Z kalibracją** (np. wyznaczoną za pomocą szachownicy — zob. dokument
`kalibracja_kamery.md`) dokładność estymacji pozy rośnie do poziomu
pojedynczych milimetrów dla markerów w odległości do 1 m.

Aby przekazać macierz kamery:

```python
cam_mtx = [
    [554.26,   0.0, 320.0],
    [  0.0, 554.26, 240.0],
    [  0.0,   0.0,   1.0],
]
calibrator = SlamCalibrator(tag_size_cm=5.0, camera_matrix=cam_mtx)
```

---

## 10. Dobre praktyki i ograniczenia

### Dobre praktyki

1. **Rozmieść markery w widocznych miejscach** — markery powinny być widoczne
   z typowych pozycji robota, najlepiej na ścianach lub stałych obiektach.

2. **Używaj wielu markerów jednocześnie** — im więcej markerów jest widocznych
   w jednej klatce, tym dokładniejsza jest estymacja pozy robota (efekt
   uśredniania).

3. **Poruszaj się powoli podczas kalibracji** — gwałtowne ruchy powodują
   rozmycie obrazu (*motion blur*), co obniża jakość detekcji. W trybie
   `robust` włączane jest wyostrzanie, które częściowo kompensuje ten efekt.

4. **Zapewnij nachodzenie widoków** — podczas przejazdu kalibracyjnego
   w każdej klatce powinien być widoczny co najmniej jeden marker z mapy,
   aby algorytm mógł wyznaczyć pozycję kamery.

5. **Skalibruj kamerę** — pełna kalibracja intrinsyków kamery
   (zob. `kalibracja_kamery.md`) znacząco poprawia dokładność.

6. **Użyj markerów o różnych ID** — każdy marker powinien mieć unikalny
   identyfikator, aby uniknąć pomyłek w budowaniu mapy.

### Ograniczenia

| Ograniczenie | Opis |
|---|---|
| Brak optymalizacji grafu (bundle adjustment) | Mapa jest budowana metodą inkrementalną ze średnią kroczącą; pełna optymalizacja grafowa zwiększyłaby dokładność, ale wykracza poza zakres lekkiego systemu. |
| Wymagany co najmniej 1 marker w mapie | Lokalizacja robota wymaga widoczności przynajmniej jednego markera już istniejącego w mapie. |
| Płaskie markery | AprilTagi są płaskie — ich estymacja pozy jest dokładna tylko, gdy obserwujemy je z przodu, nie pod bardzo ostrym kątem. |
| Brak korekty dystorsji w locie | Moduł nie prostuje obrazu; dla kamer z dużą dystorsją (szerokokątne, fisheye) należy najpierw zastosować `cv2.undistort()`. |

---

## 11. Podsumowanie

Moduł `marker_map` w systemie **robo-eye-sense** dostarcza praktyczne
narzędzia do:

- **automatycznego budowania mapy markerów** — `SlamCalibrator` analizuje
  kolejne klatki wideo, estymuje pozę 6-DoF każdego AprilTaga za pomocą
  `cv2.solvePnP` i umieszcza je w globalnym układzie współrzędnych,
- **lokalizacji robota** — `MarkerMap.estimate_robot_pose()` wyznacza
  pozycję i orientację kamery (robota) w trójwymiarowej przestrzeni na
  podstawie widocznych markerów i ich znanych pozycji w mapie,
- **persystencji mapy** — mapa może być zapisana do pliku JSON i wczytana
  przy następnym starcie, eliminując potrzebę ponownej kalibracji.

Cały pipeline jest dostępny zarówno przez **API Pythona** (klasy
`SlamCalibrator` i `MarkerMap`), jak i przez **linię poleceń** (flaga
`--scenario slam`), co pozwala na łatwą integrację z istniejącymi
systemami robotycznymi.
