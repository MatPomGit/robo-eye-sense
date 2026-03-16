# Kalibracja kamery robota — kompletna instrukcja

Dokument opisuje proces kalibracji kamery stosowanej w systemie **robo-eye-sense**.
Zawiera wyjaśnienie, dlaczego kalibracja jest konieczna, co oznaczają poszczególne
wartości pliku kalibracyjnego, jak przeprowadzić kalibrację krok po kroku oraz kiedy
należy ją powtórzyć.

---

## Spis treści

1. [Czym jest kalibracja kamery i dlaczego ma znaczenie](#1-czym-jest-kalibracja-kamery-i-dlaczego-ma-znaczenie)
2. [Wpływ kalibracji na jakość wykrywania markerów wizyjnych](#2-wpływ-kalibracji-na-jakość-wykrywania-markerów-wizyjnych)
3. [Plik kalibracyjny — struktura i znaczenie wartości](#3-plik-kalibracyjny--struktura-i-znaczenie-wartości)
4. [Przeprowadzenie kalibracji krok po kroku](#4-przeprowadzenie-kalibracji-krok-po-kroku)
5. [Przykładowe skrypty kalibracyjne](#5-przykładowe-skrypty-kalibracyjne)
6. [Wczytywanie kalibracji w robo-eye-sense](#6-wczytywanie-kalibracji-w-robo-eye-sense)
7. [Jak często powtarzać kalibrację](#7-jak-często-powtarzać-kalibrację)
8. [Typowe błędy i jak je uniknąć](#8-typowe-błędy-i-jak-je-uniknąć)

---

## 1. Czym jest kalibracja kamery i dlaczego ma znaczenie

Każda kamera wprowadza do obrazu zniekształcenia wynikające z właściwości fizycznych
obiektywu oraz geometrii sensorów. Kalibracja kamery to proces wyznaczenia parametrów
matematycznych, które opisują te zniekształcenia, tak by oprogramowanie mogło je
skompensować.

### Rodzaje zniekształceń

| Rodzaj | Opis | Typowy objaw |
|---|---|---|
| **Dystorsja radialna** | Promienie optyczne odchylają się tym bardziej, im dalej od środka obrazu | Linie proste wyginają się (efekt „beczki" lub „poduszki") |
| **Dystorsja tangencjalna** | Soczewka nie jest idealnie równoległa do płaszczyzny sensora | Obraz jest lekko przechylony lub asymetryczny |
| **Błąd ogniskowej** | Rzeczywista ogniskowa różni się od nominalnej | Obiekty wydają się większe lub mniejsze niż powinny |
| **Błąd punktu głównego** | Środek optyczny nie pokrywa się z geometrycznym środkiem obrazu | Przesunięcia pozycji wykrytych punktów |

Bez kalibracji współrzędne pikseli zwracane przez detektor nie odpowiadają
rzeczywistym pozycjom w przestrzeni. To z kolei przekłada się bezpośrednio na
obniżoną skuteczność wykrywania i śledzenia markerów przez **robo-eye-sense**.

---

## 2. Wpływ kalibracji na jakość wykrywania markerów wizyjnych

### AprilTagi

Detektor AprilTag (`AprilTagDetector`) szuka w obrazie charakterystycznych wzorów
binarnych. Dystorsja soczewki powoduje, że krawędzie markera na skraju obrazu są
zakrzywione. Skutki braku kalibracji:

- **Fałszywe odrzucenia (false negatives)** — zniekształcony marker nie przechodzi
  testu geometrycznej spójności krawędzi; detektor pomija go, mimo że jest fizycznie
  w zasięgu widzenia.
- **Błędna estymacja pozy** — bez korekcji dystorsji obliczone wektory rotacji i
  translacji znacząco odbiegają od rzeczywistości, co uniemożliwia precyzyjne
  pozycjonowanie robota względem markera.
- **Pogorszenie przy krawędziach kadru** — obiekty w rogach obrazu są zwykle bardziej
  zniekształcone niż w centrum; brak kalibracji sprawia, że obszary te są praktycznie
  martwe dla detekcji.

### Kody QR

Detektor QR (`QRCodeDetector`) również korzysta z geometrii krawędzi. Radialna
dystorsja może spowodować, że siatka modułów QR traci regularność, co utrudnia
dekodowanie. Po kalibracji:

- Wskaźnik poprawnych odczytów (*decode rate*) rośnie, szczególnie dla małych lub
  oddalonych kodów.
- Granica rozpoznawalności (minimalny rozmiar kodu w pikselach) jest niższa.

### Plamy laserowe

Detektor plam laserowych (`LaserSpotDetector`) opiera się głównie na progowaniu
jasności, więc jest mniej wrażliwy na dystorsję geometryczną niż detektory oparte
na wzorcach. Jednak kalibracja ma wpływ na **dokładność wyznaczania pozycji** plamy
w układzie współrzędnych robota — co jest krytyczne, gdy system używa plamy do
wskazywania celu.

### Tracker centroidów

`CentroidTracker` przypisuje identyfikatory śladów na podstawie odległości między
centroidami w kolejnych klatkach. Nieznana dystorsja może powodować pozorne skoki
pozycji obiektu, które tracker interpretuje jako zaginięcie istniejącego i pojawienie
się nowego obiektu. Po kalibracji ruch pikseli jest bardziej spójny z rzeczywistym
ruchem fizycznym, co zmniejsza liczbę fałszywych przypisań.

---

## 3. Plik kalibracyjny — struktura i znaczenie wartości

Wyniki kalibracji zapisuje się najczęściej w formacie **YAML** lub **JSON**.
Poniżej omówiono wszystkie standardowe pola.

### Przykładowy plik `calibration.yaml`

```yaml
# Plik kalibracyjny kamery — robo-eye-sense
# Wygenerowany: 2024-01-15  Kamera: USB-CAM-01  Rozdzielczość: 640x480

image_width: 640
image_height: 480

camera_matrix:
  rows: 3
  cols: 3
  data: [612.3, 0.0,   317.8,
         0.0,   611.7, 241.2,
         0.0,   0.0,   1.0]

distortion_coefficients:
  rows: 1
  cols: 5
  data: [-0.3512, 0.1423, 0.0008, -0.0003, -0.0421]

rectification_matrix:
  rows: 3
  cols: 3
  data: [1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0]

projection_matrix:
  rows: 3
  cols: 4
  data: [601.5, 0.0,   318.2, 0.0,
         0.0,   600.9, 241.7, 0.0,
         0.0,   0.0,   1.0,   0.0]

rms_reprojection_error: 0.412
calibration_image_count: 28
```

---

### 3.1 `image_width` / `image_height`

Rozdzielczość obrazu, dla której kalibracja jest ważna, wyrażona w pikselach.

> **Ważne:** Jeśli obraz przechwytywany przez `Camera` ma inną rozdzielczość niż
> ta zapisana w pliku, parametry kalibracyjne należy przeskalować lub kalibrację
> przeprowadzić ponownie w docelowej rozdzielczości.
>
> Tryb **FAST** w `RoboEyeDetector` zmniejsza obraz do 50 % przed detekcją.
> Przeskalowana macierz kamery dla połowy rozdzielczości wynosi:
> `fx' = fx/2, fy' = fy/2, cx' = cx/2, cy' = cy/2`.

---

### 3.2 `camera_matrix` (macierz wewnętrzna kamery, K)

```
K = | fx   0   cx |
    |  0  fy   cy |
    |  0   0    1 |
```

| Symbol | Nazwa | Jednostka | Znaczenie |
|---|---|---|---|
| `fx` | Ogniskowa pozioma | piksele | Ile pikseli odpowiada 1 metrowi odległości w kierunku X; zależy od fizycznej ogniskowej obiektywu i rozmiarze piksela sensora |
| `fy` | Ogniskowa pionowa | piksele | Analogicznie dla kierunku Y; w idealnej kamerze `fx ≈ fy` |
| `cx` | Środek optyczny X | piksele | Punkt przecięcia osi optycznej z płaszczyzną obrazu; w idealnej kamerze bliski `image_width / 2` |
| `cy` | Środek optyczny Y | piksele | Analogicznie; w idealnej kamerze bliski `image_height / 2` |

**Interpretacja praktyczna:**
- Duże wartości `fx`, `fy` → długa ogniskowa, wąskie pole widzenia (kamera „tele").
- Małe wartości → krótka ogniskowa, szerokie pole widzenia.
- Stosunek `fx / fy` bliski 1 oznacza piksele kwadratowe (typowe dla nowoczesnych
  kamer).
- `cx` i `cy` znacznie odbiegające od środka obrazu sugerują błędnie zamontowany
  obiektyw lub uszkodzony sensor.

---

### 3.3 `distortion_coefficients` (współczynniki dystorsji, D)

Wektor pięciu (lub więcej) wartości: `[k1, k2, p1, p2, k3]`

#### Dystorsja radialna: k1, k2, k3

Opisuje, jak bardzo promienie optyczne odchylają się od idealnego modelu pinhole w
zależności od odległości od środka obrazu (promienia `r`):

```
x_distorted = x * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y * (1 + k1*r² + k2*r⁴ + k3*r⁶)
```

| Wartość | Efekt |
|---|---|
| `k1 < 0` | Dystorsja beczkowa — krawędzie obrazu wyginają się do wewnątrz |
| `k1 > 0` | Dystorsja poduszkowa — krawędzie wyginają się na zewnątrz |
| Duże \|k1\| | Silna dystorsja (typowa dla obiektywów szerokokątnych, kamer rybie oko) |
| k2, k3 ≈ 0 | Wyższe rzędy dystorsji pomijalnie małe; typowe dla kamery z normalnym obiektywem |

#### Dystorsja tangencjalna: p1, p2

Wynika z nierównoległości soczewki i sensora:

```
x_distorted += 2*p1*x*y + p2*(r² + 2*x²)
y_distorted += p1*(r² + 2*y²) + 2*p2*x*y
```

| Wartość | Efekt |
|---|---|
| p1, p2 ≈ 0 | Soczewka dobrze wycentrowana (idealna sytuacja) |
| \|p1\| lub \|p2\| > 0.01 | Widoczna asymetria; warto sprawdzić mocowanie obiektywu |

> **Reguła kciuka:** W dobrze skalibrowanej kamerze przemysłowej wartości
> `|k1| < 0.5`, `|k2| < 0.2`, `|p1|, |p2| < 0.01`.
> Wartości spoza tych zakresów są możliwe, ale warto je zweryfikować.

---

### 3.4 `rectification_matrix` (macierz rektyfikacji, R)

Stosowana głównie w systemach stereo. Dla kamer mono jest to macierz jednostkowa 3×3.
Opisuje rotację, którą należy zastosować do obrazu z kamery, aby był wyrównany z
wirtualną płaszczyzną referencyjną.

---

### 3.5 `projection_matrix` (macierz projekcji, P)

Macierz 3×4 łącząca macierz wewnętrzną z przesunięciem linii bazowej (baseline)
w systemach stereo. Dla kamery mono:

```
P = | fx'   0   cx'  0 |
    |   0  fy'  cy'  0 |
    |   0   0    1   0 |
```

Wartości `fx'`, `fy'`, `cx'`, `cy'` mogą nieznacznie różnić się od wartości w
`camera_matrix` po zastosowaniu rektyfikacji.

---

### 3.6 `rms_reprojection_error` (błąd reprojekcji RMS)

Najważniejszy wskaźnik jakości kalibracji. Mierzy, jak bardzo punkty wzorca
kalibracyjnego, przetransformowane z powrotem do obrazu przy użyciu wyznaczonych
parametrów, różnią się od ich rzeczywistych pozycji w pikselach.

| Wartość RMS | Ocena jakości kalibracji |
|---|---|
| < 0.3 px | Doskonała — odpowiednia do precyzyjnej estymacji pozy |
| 0.3 – 0.5 px | Dobra — wystarczająca dla większości zastosowań robotycznych |
| 0.5 – 1.0 px | Akceptowalna — warto powtórzyć z lepszymi zdjęciami |
| > 1.0 px | Zła — kalibrację należy powtórzyć |

Wysoki błąd RMS może wynikać z:
- Zbyt małej liczby zdjęć kalibracyjnych (zalecane minimum: 20–30).
- Zdjęć tylko z jednej pozycji lub jednego kąta.
- Rozmytych lub słabo oświetlonych zdjęć wzorca.
- Źle wydrukowanego lub pofałdowanego wzorca szachownicy.

---

### 3.7 `calibration_image_count`

Liczba zdjęć użytych do kalibracji. Zalecane minimum to **20 zdjęć** pokrywających
różne pozycje i obroty wzorca. Więcej zdjęć z różnorodnym pokryciem kadru przekłada
się na lepszą dokładność — do około 40–50 zdjęć, po czym przyrost jest marginalny.

---

## 4. Przeprowadzenie kalibracji krok po kroku

### Wymagane materiały

| Element | Uwagi |
|---|---|
| Wzorzec szachownicy (chessboard) | Wydrukowany na sztywnym, płaskim podłożu; bez połysku; zalecany rozmiar siatki: 9×6 lub 8×6 |
| Miarka lub suwmiarka | Do zmierzenia rozmiaru jednego pola szachownicy w milimetrach |
| Stałe oświetlenie | Równomierne, bez odblasków na wzorcu |
| Python + OpenCV | `pip install opencv-python` |

### Krok 1 — Przygotowanie wzorca

Wydrukuj wzorzec szachownicy i zmierz rozmiar jednego pola (np. 25 mm).
Przyklej wydruk do twardej, płaskiej płyty (np. aluminium lub MDF).
Wzorzec musi być absolutnie płaski — fale lub zawinięcia drastycznie obniżają
jakość kalibracji.

### Krok 2 — Zbieranie zdjęć kalibracyjnych

Zbierz **co najmniej 20–30 zdjęć** wzorca w różnych pozycjach:

- Różne **odległości** od kamery (wzorzec bliski, średni, daleki).
- Różne **kąty** (tilty ±30–45°, obroty w płaszczyźnie).
- Wzorzec w **każdym rogu** kadru i na środku.
- **Unikaj** ruchu kamery lub wzorca podczas wykonywania zdjęcia (mogą być rozmyte).

> **Wskazówka:** Dbaj o to, aby wzorzec był widoczny w całości na każdym zdjęciu.
> Krawędzie szachownicy obcięte przez krawędź kadru spowodują błąd wykrycia.

### Krok 3 — Wykrycie narożników

OpenCV wykrywa automatycznie wewnętrzne narożniki pól szachownicy
(`cv2.findChessboardCorners`). Dla zwiększenia dokładności stosuje się
subpikselowy refining (`cv2.cornerSubPix`).

### Krok 4 — Kalibracja i ocena jakości

Funkcja `cv2.calibrateCamera` zwraca parametry kamery. Sprawdź błąd RMS —
powinien być < 0.5 px. Jeśli jest wyższy, przejrzyj zdjęcia i usuń te z wysokim
indywidualnym błędem reprojekcji.

### Krok 5 — Zapis pliku kalibracyjnego

Zapisz parametry do pliku YAML lub JSON — format opisany w sekcji 3.

---

## 5. Przykładowe skrypty kalibracyjne

### 5.1 Zbieranie zdjęć z kamery robota

```python
"""Zbieranie zdjęć kalibracyjnych na żywo z kamery."""

import cv2
from pathlib import Path
from robo_eye_sense.camera import Camera

SAVE_DIR = Path("calibration_images")
SAVE_DIR.mkdir(exist_ok=True)

CHESSBOARD = (9, 6)   # liczba wewnętrznych narożników (kolumny × wiersze)

count = 0
with Camera(source=0, width=640, height=480) as cam:
    while True:
        frame = cam.read()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(gray, CHESSBOARD, None)

        display = frame.copy()
        status = "Szachownica WYKRYTA — naciśnij SPACJA" if found else "Szachownica nie znaleziona"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, f"Zapisane: {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow("Kalibracja — zbieranie zdjęć", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            path = SAVE_DIR / f"calib_{count:03d}.png"
            cv2.imwrite(str(path), frame)
            count += 1
            print(f"Zapisano: {path}")
        elif key == ord("q"):
            break

cv2.destroyAllWindows()
print(f"\nZebrano {count} zdjęć kalibracyjnych w katalogu '{SAVE_DIR}'.")
```

### 5.2 Obliczanie parametrów i zapis pliku kalibracyjnego

```python
"""Kalibracja kamery na podstawie zebranych zdjęć szachownicy."""

import cv2
import numpy as np
import yaml
from pathlib import Path

CHESSBOARD = (9, 6)       # (kolumny, wiersze) wewnętrznych narożników
SQUARE_SIZE_MM = 25.0     # rozmiar jednego pola szachownicy w milimetrach
IMAGE_DIR = Path("calibration_images")
OUTPUT_FILE = Path("calibration.yaml")

# Przygotowanie punktów 3D w układzie wzorca
objp = np.zeros((CHESSBOARD[0] * CHESSBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD[0], 0:CHESSBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_points = []   # punkty 3D w przestrzeni wzorca
img_points = []   # punkty 2D w obrazie

image_paths = sorted(IMAGE_DIR.glob("*.png"))
image_size = None
good_images = 0

for img_path in image_paths:
    img = cv2.imread(str(img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if image_size is None:
        image_size = gray.shape[::-1]   # (width, height)

    found, corners = cv2.findChessboardCorners(gray, CHESSBOARD, None)
    if not found:
        print(f"  [POMINIĘTO] Nie znaleziono szachownicy: {img_path.name}")
        continue

    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    obj_points.append(objp)
    img_points.append(corners_refined)
    good_images += 1
    print(f"  [OK] {img_path.name}")

print(f"\nUżyto {good_images}/{len(image_paths)} zdjęć.")

if good_images < 10:
    raise RuntimeError("Za mało dobrych zdjęć (minimum 10). Zbierz więcej zdjęć.")

# Kalibracja
rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, image_size, None, None
)

print(f"\nBłąd RMS reprojekcji: {rms:.4f} px")
if rms > 1.0:
    print("OSTRZEŻENIE: Wysoki błąd RMS. Sprawdź jakość zdjęć i powtórz kalibrację.")
elif rms > 0.5:
    print("Kalibracja akceptowalna. Dla lepszych wyników zbierz więcej różnorodnych zdjęć.")
else:
    print("Kalibracja dobra.")

# Zapis do YAML
data = {
    "image_width":  image_size[0],
    "image_height": image_size[1],
    "camera_matrix": {
        "rows": 3, "cols": 3,
        "data": K.flatten().tolist(),
    },
    "distortion_coefficients": {
        "rows": 1, "cols": int(D.size),
        "data": D.flatten().tolist(),
    },
    "rectification_matrix": {
        "rows": 3, "cols": 3,
        "data": np.eye(3).flatten().tolist(),
    },
    "rms_reprojection_error": round(float(rms), 4),
    "calibration_image_count": good_images,
}

with open(OUTPUT_FILE, "w") as f:
    yaml.dump(data, f, default_flow_style=None, sort_keys=False)

print(f"\nPlik kalibracyjny zapisany: {OUTPUT_FILE}")
print(f"  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}")
print(f"  k1={D[0,0]:.4f}  k2={D[0,1]:.4f}  p1={D[0,2]:.4f}  p2={D[0,3]:.4f}")
```

---

## 6. Wczytywanie kalibracji w robo-eye-sense

Poniższy przykład pokazuje, jak załadować plik kalibracyjny i korzystać z niego
podczas pracy z `RoboEyeDetector` — w szczególności do undistortion klatek przed
detekcją oraz do obliczania pozy markera.

```python
"""Przykład użycia kalibracji z detektorem robo-eye-sense."""

import cv2
import numpy as np
import yaml
from robo_eye_sense import RoboEyeDetector
from robo_eye_sense.camera import Camera


def load_calibration(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Wczytaj macierz kamery K i współczynniki dystorsji D z pliku YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)

    K = np.array(data["camera_matrix"]["data"], dtype=np.float64).reshape(3, 3)
    D = np.array(data["distortion_coefficients"]["data"], dtype=np.float64)
    return K, D


def undistort_frame(frame: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """Usuń dystorsję z klatki na podstawie parametrów kalibracyjnych."""
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, K, D, None, new_K)
    x, y, rw, rh = roi
    return undistorted[y:y + rh, x:x + rw]


K, D = load_calibration("calibration.yaml")

detector = RoboEyeDetector(
    enable_apriltag=True,
    enable_qr=True,
    enable_laser=True,
)

with Camera(source=0, width=640, height=480) as cam:
    while True:
        frame = cam.read()
        if frame is None:
            break

        # Korekcja dystorsji przed detekcją
        corrected = undistort_frame(frame, K, D)

        detections = detector.process_frame(corrected)
        annotated = detector.draw_detections(corrected, detections)

        for d in detections:
            print(f"{d.detection_type.value}: id={d.identifier} center={d.center}")

        cv2.imshow("robo-eye-sense (skalibrowana)", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cv2.destroyAllWindows()
```

> **Uwaga dotycząca trybu FAST:** Tryb `DetectionMode.FAST` zmniejsza rozdzielczość
> klatki do 50% przed detekcją. Jeśli stosujesz undistortion przed przekazaniem
> klatki do detektora, korekcja powinna być wykonana w **oryginalnej** rozdzielczości.
> Detektor wewnętrznie skaluje ramkę, a następnie przywraca współrzędne.

---

## 7. Jak często powtarzać kalibrację

### Bezwzględnie wymagana ponowna kalibracja

| Sytuacja | Uzasadnienie |
|---|---|
| **Zmiana kamery** | Nowa kamera ma inne parametry obiektywu i sensora |
| **Zmiana obiektywu** | Inny obiektyw = inne `fx`, `fy`, `k1–k3` |
| **Zmiana rozdzielczości obrazu** | Parametry kalibracji są rozdzielczość-specyficzne |
| **Fizyczne uderzenie w kamerę** | Może zmienić ustawienie soczewki względem sensora |
| **Poluzowanie mocowania obiektywu** | Skutkuje zmianą tangencjalnej dystorsji |
| **Znacząca zmiana warunków temperaturowych** | Rozszerzalność termiczna elementów optycznych zmienia ogniskową |

### Zalecana ponowna kalibracja

| Sytuacja | Zalecany interwał |
|---|---|
| **Regularna eksploatacja robota** | Co 3–6 miesięcy |
| **Praca w zmiennym środowisku** (duże wahania temperatury, wibracje) | Co 4–8 tygodni |
| **Precyzyjne zastosowania** (np. montaż komponentów, pomiary) | Po każdej istotnej zmianie konfiguracji |
| **Raz na sezon w zastosowaniach przemysłowych** | Standardowe zalecenie dla środowisk produkcyjnych |

### Kiedy kalibracja jest opcjonalna

- Tymczasowe testy funkcjonalne (wystarczy kalibracja przybliżona).
- Aplikacje, gdzie liczy się tylko obecność markera, nie jego precyzyjna pozycja.
- Kamery z bardzo małą dystorsją (jasne obiektywy z małą aperturą).

### Monitorowanie jakości kalibracji w trakcie pracy

Dobry sposób na wykrycie utraty kalibracji to periodyczne sprawdzanie błędu
reprojekcji „w locie": umieść znany marker AprilTag w znanych współrzędnych
i mierz odchylenie od przewidywanej pozycji. Wzrost błędu powyżej progu (np. 5 px)
sygnalizuje konieczność ponownej kalibracji.

---

## 8. Typowe błędy i jak je uniknąć

| Błąd | Objaw | Rozwiązanie |
|---|---|---|
| **Pofałdowany wzorzec** | Wysokie RMS (> 1 px), niespójne `k1` | Użyj sztywnego podkładu, np. aluminium |
| **Za mało zdjęć** | Niestabilne wartości `k2`, `k3` | Zbierz co najmniej 20–30 zdjęć |
| **Zbyt jednorodne kąty** | Dobry RMS, ale błędy przy krawędziach kadru | Uwzględnij zdjęcia wzorca w rogach kadru i pod dużymi kątami |
| **Rozmyte zdjęcia** | Wysokie RMS, błąd `findChessboardCorners` | Dobre oświetlenie, krótki czas naświetlania, stałe ustawienie |
| **Zmienna ogniskowa (zoom)** | RMS dobry, ale błędy w eksploatacji | Zablokuj zoom na stałe przed kalibracją i nie zmieniaj go |
| **Kalibracja w złej rozdzielczości** | Błędna detekcja przy innych ustawieniach `Camera` | Kalibruj zawsze w docelowej rozdzielczości |
| **Nie uwzględniono undistortion** | Markery przy krawędziach pomijane | Zastosuj `cv2.undistort()` przed przekazaniem klatki do detektora |

---

*Niniejszy dokument jest częścią projektu **robo-eye-sense**. Więcej informacji o
samym systemie detekcji markerów znajdziesz w [README.md](README.md).*
