# Changelog

Wszystkie istotne zmiany w projekcie **robo-vision** są dokumentowane
w tym pliku. Format jest oparty na
[Keep a Changelog](https://keepachangelog.com/pl/1.1.0/), a wersjonowanie
stosuje [Semantic Versioning](https://semver.org/lang/pl/).

---

## [0.5.0] – 2026-03-20

### Poprawione

- **Detekcja tagów z nazwami** – naprawiono błąd, przez który wzbogacanie
  identyfikatorów tagów AprilTag o czytelne nazwy (`tag_names`) nie działało
  gdy detektor był konfigurowany bez wstępnej inicjalizacji pupil-apriltags.
  Zmieniono mechanizm wyłączania detektora: `disable_april()` przenosi teraz
  obiekt C do `_april_holder` (zapobiega wywołaniu `__del__`) i ustawia
  `_april_detector = None`, dzięki czemu warunek `is not None` jednoznacznie
  określa aktywność detektora.
- **Spójność wersji** – ujednolicono numer wersji pomiędzy `pyproject.toml`
  a `robo_vision/__init__.py` (oba pliki zawierają teraz `0.5.0`).

---

## [0.4.0] – 2026-03-19

### Dodane

- **Dokumentacja nowych trybów pracy** – uzupełniono README o szczegółowe
  instrukcje CLI i przykłady użycia dla trybów: *calibration* (kalibracja
  kamery szachownicą), *box* (detekcja pudełek/prostopadłościanów), *pose*
  (estymacja pozy 6-DoF tagów AprilTag) i *follow* (śledzenie markera
  z generowaniem sygnałów sterowania).
- **Numer wersji w tytule okna GUI** – okno główne Tkinter wyświetla
  aktualną wersję w pasku tytułu (np. „robo-vision v0.4.0").

### Poprawione

- Naprawiono tytuł okna GUI – zastąpiono stały ciąg „robot-vision"
  dynamicznym `f"{APP_NAME} v{__version__}"`.
- Poprawiono CHANGELOG – zamieniono `--scenario` na `--mode` w opisach
  trybów Offset i SLAM (v0.2.0).

---

## [0.3.0] – 2026-03-18

### Dodane

- **Podsumowanie konfiguracji w terminalu** – przy każdym uruchomieniu
  program wyświetla tryb wyświetlania, tryb detekcji, listę włączonych
  detektorów, źródło wideo, aktywny scenariusz i ścieżkę nagrywania
  (PR #35).
- **Uzupełniona dokumentacja** – tabela funkcji w README rozszerzona
  o scenariusze Offset/SLAM, estymację odległości, nagrywanie wideo,
  tryb bezgłowy i podsumowanie konfiguracji. Dodana sekcja z linkami
  do dokumentacji dodatkowej. Rozbudowane przykłady użycia
  programistycznego (API nagrywania, offset, SLAM).
- **CHANGELOG.md** – historia zmian i wydań projektu.

### Poprawione

- Poprawiona tabela strojenia wydajności – usunięto odniesienia do
  nieistniejących flag `--no-qr` / `--no-laser`.
- Rozbudowana sekcja architektury – pełna lista eksportów publicznego API
  pakietu.

---

## [0.2.0] – 2026-03-17

### Dodane

- **Scenariusz SLAM** (`--mode slam`) – inkrementalne budowanie mapy
  markerów 3-D z estymacją pozy 6-DoF robota; zapis/odczyt mapy JSON
  (PR #32).
- **Zakładka SLAM w GUI** – wizualizacja 3-D (widok z góry), lista
  markerów, pozycja i orientacja robota w czasie rzeczywistym (PR #33).
- **Nagrywanie wideo** (`--record FILE`) – klasa `VideoRecorder`
  (start/stop/write_frame), przycisk w GUI, obsługa w trybie bezgłowym
  i scenariuszowym (PR #27).
- **Scenariusz Offset** (`--mode offset`) – kalibracja przesunięcia
  kamery na podstawie pozycji AprilTagów z estymacją odległości do
  tagów i pozycji referencyjnej (PR #21, #26).
- **Integracja scenariuszy z GUI** – zakładki Offset i SLAM w panelu
  informacyjnym; przyciski Start/Capture Reference/Reset (PR #26).
- **Estymacja odległości** – model kamery otworkowej: odległość do
  każdego AprilTaga (`per_tag_distances_cm`) i dystans do pozycji
  referencyjnej (`distance_to_reference_cm`) (PR #26).
- **Dokumentacja SLAM** – `slam_marker_map.md`: algorytm budowania mapy,
  `solvePnP`, serializacja JSON, dobre praktyki (PR #32).
- **Dokumentacja kalibracji kamery** – `kalibracja_kamery.md`:
  intrinsyki, dystorsja, procedura krok po kroku (PR #20).
- **Przewodnik po markerach wizyjnych** – `markery_wizyjne_nawigacja.md`:
  AprilTagi, QR, ArUco, SLAM, odometria wizualna (PR #31).
- **Przewodnik GitHub Releases & Packages** –
  `github_releases_packages.md` (PR #28).
- **GitHub Actions workflow** (`release.yml`) – automatyczne testy,
  budowanie pakietu i tworzenie GitHub Release przy tagu `v*` (PR #30).
- Poprawka skalowania obrazu kamery w GUI – zachowanie proporcji
  i minimalne rozmiary paneli (PR #29).
- Naprawa ostrzeżenia Qt font directory (`QT_QPA_FONTDIR`) (PR #25).

### Zmienione

- **Refaktoryzacja publicznego API** – `RoboEyeDetector` z właściwościami
  `april_enabled`/`qr_enabled`/`laser_enabled`, metodami
  `enable_*()`/`disable_*()`, `mode`/`DetectionMode` enum (PR #24).
- QR i laser domyślnie wyłączone – włączane flagami `--qr` / `--laser`
  (PR #22).
- Poprawiona filtracja AprilTag – `min_decision_margin=25.0` (PR #22).
- README przetłumaczone na język polski (PR #23).

---

## [0.1.0] – 2026-03-15

### Dodane

- **Detekcja AprilTag** – `AprilTagDetector` oparty na `pupil-apriltags`,
  obsługa 4 rodzin tagów jednocześnie.
- **Detekcja kodów QR** – `QRCodeDetector` z preferowanym backendem
  `pyzbar` i alternatywą `cv2.QRCodeDetector`.
- **Detekcja punktu lasera** – `LaserSpotDetector` z progowaniem
  jasności, filtrem kołowości i regulowaną czułością.
- **Zunifikowany detektor** – `RoboEyeDetector` orkiestrujący wszystkie
  trzy podsystemy detekcji.
- **Śledzenie wielu obiektów** – `CentroidTracker` z trwałymi ID
  śledzenia; dopasowanie semantyczne dla tagów/QR, centroidowe
  dla obiektów bez etykiety.
- **Trzy tryby pracy** – Normal, Fast (50 % rozdzielczości), Robust
  (wyostrzanie unsharp-mask + filtr Kalmana).
- **GUI Tkinter** – panel sterowania z przełączaniem trybów
  (Ctrl+1/2/3), detektorów, suwakami parametrów lasera, nakładką
  progową.
- **CLI** (`main.py`) – `--source`, `--width`, `--height`, `--mode`,
  `--headless`, `--gui`, `--no-apriltag`, `--qr`, `--laser`,
  `--laser-threshold`.
- **Wrapper kamery** – `Camera` (context manager, obsługa indeksów,
  plików, strumieni RTSP).
