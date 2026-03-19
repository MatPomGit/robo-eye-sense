# Markery wizyjne w nawigacji robotów — kompleksowy przewodnik

Dokument opisuje, w jaki sposób urządzenia i roboty wykorzystują markery wizyjne
(takie jak AprilTagi czy kody QR) do określania swojej pozycji w przestrzeni
oraz pozycji otaczających obiektów. Omówiono również alternatywne metody
wizyjne, parametry nawigacyjne oraz techniki zwiększania dokładności
pozycjonowania.

---

## Spis treści

1. [Wprowadzenie — rola wizji maszynowej w nawigacji](#1-wprowadzenie--rola-wizji-maszynowej-w-nawigacji)
2. [Markery fiducjalne — AprilTagi, kody QR i inne](#2-markery-fiducjalne--apriltagi-kody-qr-i-inne)
   1. [Czym są markery fiducjalne](#21-czym-są-markery-fiducjalne)
   2. [AprilTagi — budowa i zasada działania](#22-apriltagi--budowa-i-zasada-działania)
   3. [Kody QR jako markery pozycyjne](#23-kody-qr-jako-markery-pozycyjne)
   4. [ArUco, STag i inne systemy znaczników](#24-aruco-stag-i-inne-systemy-znaczników)
3. [Rozmieszczenie markerów w przestrzeni](#3-rozmieszczenie-markerów-w-przestrzeni)
   1. [Ile markerów jest potrzebnych](#31-ile-markerów-jest-potrzebnych)
   2. [Gdzie umieszczać markery](#32-gdzie-umieszczać-markery)
   3. [Mapy markerów i układy odniesienia](#33-mapy-markerów-i-układy-odniesienia)
4. [Alternatywne metody wizyjnej nawigacji](#4-alternatywne-metody-wizyjnej-nawigacji)
   1. [Śledzenie linii na podłodze](#41-śledzenie-linii-na-podłodze)
   2. [Rozpoznawanie kolorów i stref kolorystycznych](#42-rozpoznawanie-kolorów-i-stref-kolorystycznych)
   3. [Rozpoznawanie kształtów i obiektów](#43-rozpoznawanie-kształtów-i-obiektów)
   4. [Cechy naturalne — Visual SLAM i odometria wizualna](#44-cechy-naturalne--visual-slam-i-odometria-wizualna)
5. [Parametry nawigacyjne wyznaczane z markerów](#5-parametry-nawigacyjne-wyznaczane-z-markerów)
   1. [Pozycja i orientacja — pose estimation](#51-pozycja-i-orientacja--pose-estimation)
   2. [Wektor przesunięcia (offset) i korekcja ruchu](#52-wektor-przesunięcia-offset-i-korekcja-ruchu)
   3. [Estymacja odległości — model kamery otworkowej](#53-estymacja-odległości--model-kamery-otworkowej)
   4. [Kąty obrotu — roll, pitch, yaw](#54-kąty-obrotu--roll-pitch-yaw)
   5. [Prędkość i przyspieszenie — śledzenie w czasie](#55-prędkość-i-przyspieszenie--śledzenie-w-czasie)
6. [Techniki zwiększania dokładności pozycjonowania](#6-techniki-zwiększania-dokładności-pozycjonowania)
   1. [Kalibracja kamery](#61-kalibracja-kamery)
   2. [Filtr Kalmana i śledzenie predykcyjne](#62-filtr-kalmana-i-śledzenie-predykcyjne)
   3. [Fuzja wielu markerów](#63-fuzja-wielu-markerów)
   4. [Fuzja sensorów — IMU, enkodery, LiDAR](#64-fuzja-sensorów--imu-enkodery-lidar)
   5. [Wiele kamer i stereo-wizja](#65-wiele-kamer-i-stereo-wizja)
   6. [Przetwarzanie obrazu — wyostrzanie i adaptacja](#66-przetwarzanie-obrazu--wyostrzanie-i-adaptacja)
   7. [Tryby pracy — kompromis między szybkością a dokładnością](#67-tryby-pracy--kompromis-między-szybkością-a-dokładnością)
7. [Praktyczne przykłady zastosowań](#7-praktyczne-przykłady-zastosowań)
8. [Podsumowanie](#8-podsumowanie)

---

## 1. Wprowadzenie — rola wizji maszynowej w nawigacji

Roboty mobilne, drony i urządzenia autonomiczne potrzebują niezawodnego
sposobu na ustalenie, gdzie się znajdują i jak są zorientowane w przestrzeni.
Bez tej informacji niemożliwe jest planowanie tras, omijanie przeszkód czy
precyzyjne wykonywanie zadań (np. pobieranie przedmiotów z regału).

Wizja maszynowa (ang. *computer vision*) oferuje jedno z najbardziej
uniwersalnych rozwiązań tego problemu. Kamera zamontowana na robocie
rejestruje obraz otoczenia, a algorytmy przetwarzania obrazu wyodrębniają
z niego informacje geometryczne — pozycję, orientację i odległość od znanych
punktów odniesienia. W odróżnieniu od GPS (którego sygnał bywa niedostępny
wewnątrz budynków) czy systemów UWB (wymagających dedykowanej
infrastruktury radiowej), kamera korzysta z informacji wizualnych, które
mogą być zarówno celowo umieszczonymi znacznikami, jak i naturalnymi cechami
otoczenia.

System **robo-vision** jest przykładem lekkiej biblioteki detekcji
wizualnej przeznaczonej dla robotów mobilnych. Łączy detekcję AprilTagów
(poprzez bibliotekę `pupil-apriltags`), kodów QR (biblioteka `pyzbar`) oraz
punktów laserowych, umożliwiając śledzenie markerów w czasie rzeczywistym
i wyznaczanie wektora korekcji pozycji kamery.

---

## 2. Markery fiducjalne — AprilTagi, kody QR i inne

### 2.1. Czym są markery fiducjalne

Marker fiducjalny (ang. *fiducial marker*) to wzór graficzny zaprojektowany
tak, aby był łatwy do wykrycia i jednoznacznej identyfikacji przez algorytmy
wizyjne. Każdy marker niesie zakodowaną informację (najczęściej unikalny
numer identyfikacyjny) i posiada ściśle zdefiniowaną geometrię, dzięki
czemu oprogramowanie może obliczyć jego pozycję i orientację w
trójwymiarowej przestrzeni.

Kluczowe cechy markerów fiducjalnych:

| Cecha | Znaczenie |
|---|---|
| **Unikalny identyfikator** | Pozwala odróżnić markery od siebie i przypisać im znane pozycje w przestrzeni |
| **Znana geometria** | Prostokątny kształt o znanym rozmiarze fizycznym umożliwia obliczenie pozycji 3D |
| **Wysoki kontrast** | Czarno-biały wzór zapewnia niezawodną detekcję w różnych warunkach oświetlenia |
| **Wbudowana korekcja błędów** | Kodowanie z nadmiarowością pozwala na poprawne odczytanie nawet przy częściowym zasłonięciu |

### 2.2. AprilTagi — budowa i zasada działania

AprilTagi to najbardziej rozpowszechniona rodzina markerów fiducjalnych
stosowanych w robotyce. Zostały opracowane na Uniwersytecie Michigan
i zaprojektowane z myślą o szybkiej, niezawodnej detekcji w trudnych
warunkach.

**Budowa AprilTaga:**
- Zewnętrzna czarna ramka o stałej szerokości — służy do wykrycia
  prostokątnego konturu markera w obrazie.
- Wewnętrzna siatka bitowa (matryca binarna) — koduje unikalny
  identyfikator (ID) taga.
- Znana fizyczna wielkość — zwykle kwadrat o boku 5–20 cm.

**Rodziny tagów** różnią się rozmiarem siatki bitowej i poziomem korekcji
błędów. Najpopularniejsze to:

| Rodzina | Rozmiar siatki | Liczba unikalnych ID | Uwagi |
|---|---|---|---|
| **tag36h11** | 6 × 6 w ramce 10 × 10 | 587 | Najczęściej stosowana — dobry kompromis między pojemnością a odpornością na błędy |
| **tag25h9** | 5 × 5 w ramce 9 × 9 | 35 | Mniejsza matryca, mniej ID, ale szybsza detekcja |
| **tag16h5** | 4 × 4 w ramce 8 × 8 | 30 | Bardzo mała matryca, wykrywalna z daleka, lecz podatna na fałszywe detekcje |

W systemie **robo-vision** klasa `AprilTagDetector` wykrywa jednocześnie
cztery standardowe rodziny tagów (`tag36h11`, `tag25h9`, `tag16h5`,
`tag12h10`). Aby wyeliminować fałszywe detekcje spowodowane szumem obrazu
lub przypadkowymi wzorami, stosuje próg minimalnej pewności
(`min_decision_margin`), domyślnie ustawiony na 25,0.

**Algorytm detekcji AprilTaga (uproszczony):**

1. **Konwersja do skali szarości** — obraz kolorowy zamieniany jest na
   jednokanałowy obraz jasności.
2. **Detekcja krawędzi i segmentów linii** — algorytm znajduje odcinki
   prostych w obrazie.
3. **Wyszukiwanie czworokątów** — z odcinków budowane są zamknięte
   czworokąty, które mogą odpowiadać zewnętrznej ramce taga.
4. **Dekodowanie matrycy bitowej** — wnętrze każdego czworokąta jest
   próbkowane, a wzór bitowy porównywany ze słownikiem danej rodziny.
5. **Wyznaczenie środka i narożników** — zwracane są współrzędne pikseli:
   środek taga oraz cztery narożniki z subpikselową dokładnością.

### 2.3. Kody QR jako markery pozycyjne

Kody QR (Quick Response) nie zostały zaprojektowane specjalnie do celów
nawigacyjnych, lecz mogą pełnić rolę markerów pozycyjnych dzięki kilku
właściwościom:

- **Trzy wzorce pozycyjne** (ang. *finder patterns*) w narożnikach
  umożliwiają szybkie zlokalizowanie kodu w obrazie.
- **Duża pojemność danych** — w kodzie QR można zapisać np. współrzędne
  miejsca, identyfikator strefy czy URL do mapy.
- **Powszechna dostępność** — kody QR można wydrukować na zwykłej
  drukarce.

Wadą kodów QR w kontekście nawigacji jest ich wolniejsze dekodowanie
w porównaniu z AprilTagami oraz mniejsza niezawodność wyznaczenia pozycji
3D (mniej precyzyjna lokalizacja narożników).

### 2.4. ArUco, STag i inne systemy znaczników

Oprócz AprilTagów i kodów QR w robotyce stosuje się również inne systemy
znaczników wizyjnych:

| System | Opis | Zalety |
|---|---|---|
| **ArUco** | Markery bitmapowe zintegrowane z OpenCV | Natywna obsługa w OpenCV, łatwa integracja |
| **STag** | Markery okrągłe z kodem binarnym | Lepsza detekcja pod dużym kątem obserwacji |
| **CCTag** | Markery oparte na koncentrycznych okręgach | Odporne na rozmycie ruchu (*motion blur*) |
| **ChromaTag** | Kolorowe warianty AprilTagów | Szybsza detekcja dzięki filtracji barw |
| **WhyCode** | Okrągłe markery binarne z unikalnym wzorem | Bardzo szybka detekcja i dekodowanie |

---

## 3. Rozmieszczenie markerów w przestrzeni

### 3.1. Ile markerów jest potrzebnych

Minimalna liczba markerów zależy od wymagań aplikacji:

| Scenariusz | Minimalna liczba | Zalecana liczba | Uzasadnienie |
|---|---|---|---|
| **Wyznaczenie pozycji 6-DoF** (x, y, z + obroty) | 1 marker (znany rozmiar + kalibracja kamery) | 2–4 markery | Jeden marker wystarczy matematycznie, ale wiele markerów zmniejsza błąd |
| **Korekcja przesunięcia (offset)** | 1 wspólny marker w obu klatkach | 3–5 markerów | Uśrednienie wektorów z wielu markerów tłumi szum (jak w `robo-vision`) |
| **Nawigacja w dużym pomieszczeniu** | Wystarczająco dużo, by ≥ 1 marker był widoczny z każdego miejsca | 1 marker na 2–3 m² widocznej powierzchni | Robot musi zawsze „widzieć" przynajmniej jeden znany punkt |
| **Lokalizacja z redundancją** | 3+ markerów widocznych jednocześnie | 5–8 markerów | Pozwala wykryć i odrzucić błędne odczyty (outliers) |

**Zasada ogólna:** im więcej markerów jest widocznych jednocześnie, tym
dokładniejsza i bardziej odporna na zakłócenia jest estymacja pozycji.

### 3.2. Gdzie umieszczać markery

Strategia rozmieszczenia markerów zależy od środowiska i zadania robota:

**Powierzchnie płaskie i sztywne:**
- Ściany (na wysokości kamery robota lub nieco wyżej).
- Sufit — szczególnie popularne w robotyce magazynowej, gdzie sufit jest
  wolny od przeszkód i dobrze widoczny z dołu.
- Podłoga — stosowana, gdy kamera skierowana jest w dół (np. drony
  latające nisko, roboty inspekcyjne).
- Blaty robocze, regały, stanowiska montażowe.

**Dobre praktyki rozmieszczenia:**

| Zasada | Dlaczego |
|---|---|
| Markery powinny być płaskie i prostopadłe (lub pod znanym kątem) do typowego kierunku patrzenia kamery | Duży kąt obserwacji zniekształca obraz markera i obniża dokładność |
| Rozmieszczenie powinno zapewniać widoczność ≥ 1 markera z każdego miejsca trasy robota | Utrata widoczności markerów oznacza utratę informacji o pozycji |
| Markery nie powinny być zasłaniane przez ruchome obiekty (ludzi, inne roboty, ładunki) | Przesłonięcie markera powoduje tymczasową utratę detekcji |
| Unikać miejsc o ekstremalnym oświetleniu (bezpośrednie światło słoneczne, głębokie cienie) | Wysoki kontrast otoczenia utrudnia segmentację czarno-białego wzoru |
| Stosować markery o odpowiednim rozmiarze do odległości detekcji | Mały marker jest nieczytelny z daleka; zbyt duży marker nie mieści się w polu widzenia z bliska |

**Orientacyjne rozmiary markerów w zależności od odległości:**

| Odległość detekcji | Zalecany rozmiar boku markera |
|---|---|
| do 0,5 m | 3–5 cm |
| 0,5–2 m | 5–10 cm |
| 2–5 m | 10–20 cm |
| powyżej 5 m | 20–30 cm lub więcej |

### 3.3. Mapy markerów i układy odniesienia

W zaawansowanych systemach pozycje wszystkich markerów zapisywane są
w **mapie markerów** (ang. *marker map* lub *tag map*), która definiuje
położenie i orientację każdego markera w globalnym układzie współrzędnych.

Mapa markerów umożliwia:
- **Globalną lokalizację** — robot widząc dowolny zbiór markerów oblicza
  swoją pozycję w tym samym, spójnym układzie odniesienia.
- **Wzajemną weryfikację** — jeśli pozycje obliczone na podstawie
  różnych markerów nie zgadzają się, system może zasygnalizować błąd.
- **Automatyczną kalibrację** — algorytmy SLAM mogą stopniowo budować
  mapę markerów w miarę eksploracji pomieszczenia.

---

## 4. Alternatywne metody wizyjnej nawigacji

Markery fiducjalne nie są jedynym sposobem na wizyjną nawigację.
Poniżej opisano metody wykorzystujące inne cechy wizualne otoczenia.

### 4.1. Śledzenie linii na podłodze

Jedną z najprostszych i najstarszych metod wizyjnej nawigacji jest
podążanie za linią narysowaną (lub naklejoną) na podłodze. Metoda ta
jest szeroko stosowana w logistyce magazynowej (roboty AGV — Automated
Guided Vehicles).

**Zasada działania:**
1. Kamera lub czujnik optyczny skierowany w dół rejestruje obraz podłogi.
2. Obraz jest progowany (binaryzowany), aby wyodrębnić linię o znanym
   kolorze (najczęściej czarna linia na jasnym tle lub biała na ciemnym).
3. Algorytm oblicza odchylenie linii od środka obrazu — jeśli linia
   przesuwa się w lewo, robot skręca w lewo, i odwrotnie.

**Zalety:**
- Prosta implementacja i niskie wymagania obliczeniowe.
- Łatwe do wdrożenia na tanich platformach.
- Intuicyjne — trasa jest widoczna gołym okiem.

**Wady:**
- Brak informacji o pozycji bezwzględnej (robot „wie" jedynie, że jest
  na linii, ale nie zna swojego miejsca na trasie).
- Wrażliwość na zabrudzenia, uszkodzenia linii, cienie.
- Brak elastyczności — zmiana trasy wymaga fizycznej modyfikacji podłogi.

**Rozszerzenia:**
- Kolorowe znaczniki na linii (np. czerwone kółko = punkt zatrzymania,
  niebieskie = skrzyżowanie) pozwalają dodać do trasy semantyczną
  informację.
- Dwie równoległe linie umożliwiają detekcję kąta odchylenia i lepszą
  regulację ruchu.

### 4.2. Rozpoznawanie kolorów i stref kolorystycznych

Niektóre systemy wykorzystują **kolorowe obszary** (strefy) w otoczeniu
jako punkty orientacyjne:

- **Kolorowe podłogi** — np. hala magazynowa z niebieską podłogą
  w strefie załadunku i szarą w strefie składowania.
- **Kolorowe ściany lub pasy** — robot wykrywa dominujący kolor
  w polu widzenia i na tej podstawie identyfikuje strefę.
- **Kolorowe obiekty** — np. pomarańczowy słupek oznaczający punkt
  docelowy.

**Techniki przetwarzania:**
- Konwersja obrazu do przestrzeni barw HSV (ang. *Hue, Saturation,
  Value*), która jest bardziej odporna na zmiany oświetlenia niż RGB.
- Progowanie zakresu barwy (np. „H ∈ [100, 130], S ≥ 50" = kolor
  niebieski).
- Obliczenie centroidu (środka ciężkości) wykrytego obszaru kolorowego.

**Ograniczenia:**
- Silna zależność od warunków oświetleniowych — zmiana koloru światła
  (np. dzienne vs. sztuczne) zmienia postrzegane barwy.
- Trudność w rozróżnianiu podobnych kolorów.
- Brak unikalnej identyfikacji — dwa niebieskie obszary wyglądają
  identycznie.

### 4.3. Rozpoznawanie kształtów i obiektów

Rozpoznawanie kształtów przedmiotów może służyć do nawigacji, gdy
w otoczeniu znajdują się obiekty o znanej geometrii:

- **Detekcja okręgów** (transformata Hougha) — np. okrągłe słupy,
  koła maszyn, zegary na ścianach.
- **Detekcja prostokątów** — drzwi, okna, tablice informacyjne.
- **Detekcja obiektów 3D** — z użyciem sieci neuronowych (np. YOLO,
  SSD) robot może rozpoznawać krzesła, stoły, regały, a na podstawie
  ich znanych wymiarów estymować odległość.

**Zalety:**
- Nie wymaga instalacji żadnych markerów — wykorzystuje istniejące
  elementy otoczenia.
- Działa w zmiennych warunkach (obiekty nie zmieniają kształtu tak
  szybko jak kolory zmieniają odcień).

**Wady:**
- Wyższe wymagania obliczeniowe (szczególnie przy użyciu sieci
  neuronowych).
- Mniejsza precyzja niż markery fiducjalne — krawędzie obiektów
  naturalnych nie są tak ostre i jednoznaczne.
- Konieczność budowy bazy danych znanych obiektów i ich wymiarów.

### 4.4. Cechy naturalne — Visual SLAM i odometria wizualna

Najbardziej zaawansowanym podejściem jest nawigacja bez żadnych
sztucznych znaczników, oparta wyłącznie na **naturalnych cechach
otoczenia** (ang. *natural features*):

- **Odometria wizualna** (ang. *Visual Odometry, VO*) — porównywanie
  kolejnych klatek kamery w celu śledzenia ruchu robota na podstawie
  przesunięcia punktów charakterystycznych (narożników, krawędzi).
- **Visual SLAM** (Simultaneous Localization and Mapping) — robot
  jednocześnie buduje mapę otoczenia i lokalizuje się na niej.
  Popularne implementacje to ORB-SLAM3, LSD-SLAM, RTAB-Map.

Metody te są potężne, lecz wymagają znacznych zasobów obliczeniowych
i są wrażliwe na monotonne otoczenie (np. długi, jednolity korytarz).

---

## 5. Parametry nawigacyjne wyznaczane z markerów

### 5.1. Pozycja i orientacja — pose estimation

Najważniejszym parametrem wyznaczanym na podstawie markera jest **poza**
(ang. *pose*) kamery (lub markera) — czyli pozycja i orientacja w
trójwymiarowej przestrzeni, opisywana łącznie sześcioma stopniami
swobody (6-DoF):

| Parametr | Opis | Jednostka |
|---|---|---|
| **tx** (translacja X) | Przesunięcie w osi poziomej | metry lub cm |
| **ty** (translacja Y) | Przesunięcie w osi pionowej | metry lub cm |
| **tz** (translacja Z) | Odległość od kamery (głębia) | metry lub cm |
| **roll** (przechylenie) | Obrót wokół osi Z kamery | stopnie lub radiany |
| **pitch** (pochylenie) | Obrót wokół osi X kamery | stopnie lub radiany |
| **yaw** (odchylenie) | Obrót wokół osi Y kamery | stopnie lub radiany |

**Jak to działa:**

1. Algorytm detekcji wykrywa narożniki markera w obrazie (współrzędne
   2D w pikselach).
2. Znane są współrzędne 3D narożników markera w jego lokalnym układzie
   (prostokąt o znanych wymiarach, np. 5 × 5 cm).
3. Algorytm **solvePnP** (OpenCV) lub równoważny rozwiązuje problem
   „Perspektywa-z-n-Punktów" — znajduje macierz rotacji **R** i wektor
   translacji **t**, które najlepiej odwzorowują punkty 3D markera na ich
   obserwowane pozycje 2D na obrazie.
4. Wynikiem jest poza kamery względem markera (lub odwrotnie).

### 5.2. Wektor przesunięcia (offset) i korekcja ruchu

W prostszych scenariuszach, gdy pełna pozycja 6-DoF nie jest wymagana,
wystarczy obliczyć **wektor przesunięcia** (ang. *offset*) — różnicę
między aktualną a oczekiwaną pozycją markera w obrazie.

System **robo-vision** realizuje to podejście w module
`offset_scenario`. Algorytm:

1. **Klatka referencyjna** — kamera w pozycji docelowej rejestruje obraz;
   środki wszystkich widocznych AprilTagów są zapamiętywane.
2. **Klatka bieżąca** — po przemieszczeniu kamery rejestrowany jest nowy
   obraz z tymi samymi tagami.
3. **Obliczenie offsetu** — dla każdego taga widocznego w obu klatkach
   obliczany jest wektor przesunięcia `(dx, dy)` w pikselach. Wynikowy
   offset to średnia z wektorów wszystkich dopasowanych tagów.
4. **Przeliczenie na centymetry** — znając rozmiar fizyczny taga
   (domyślnie 5 cm) i przybliżoną ogniskową kamery, piksele przeliczane
   są na centymetry metodą kamery otworkowej (pinhole camera model).

Wektor offsetu mówi robotowi: „przesuń się o tyle pikseli (lub
centymetrów) w prawo/lewo i w górę/dół, aby wrócić do pozycji
referencyjnej". Jest to informacja wystarczająca do sterowania napędami
w pętli regulacji.

### 5.3. Estymacja odległości — model kamery otworkowej

Model kamery otworkowej (ang. *pinhole camera model*) pozwala na
estymację odległości od kamery do markera na podstawie jego rozmiaru
pozornego (w pikselach):

```
odległość = (rozmiar_fizyczny × ogniskowa) / rozmiar_pozorny_w_pikselach
```

Gdzie:
- **rozmiar_fizyczny** — znana długość boku markera w centymetrach
  (np. 5 cm).
- **ogniskowa** — ogniskowa kamery w pikselach, wyznaczana z kalibracji
  lub przybliżana ze znanego kąta widzenia:
  `f = (szerokość_obrazu / 2) / tan(HFOV / 2)`.
- **rozmiar_pozorny_w_pikselach** — średnia długość boków markera
  zmierzona w obrazie.

System **robo-vision** implementuje tę estymację w funkcjach
`estimate_focal_length_px()` i `estimate_tag_distance_cm()`, domyślnie
przyjmując kąt widzenia kamery 60° i rozmiar taga 5 cm.

### 5.4. Kąty obrotu — roll, pitch, yaw

Poza translacją, markery pozwalają na wyznaczenie orientacji kamery
(lub obiektu, do którego kamera jest przymocowana):

- **Yaw** (odchylenie) — obrót wokół osi pionowej (robot skręca w lewo
  lub w prawo). Łatwo obserwowalne jako przesunięcie markera w poziomie.
- **Pitch** (pochylenie) — obrót wokół osi poprzecznej (robot pochyla
  się do przodu lub do tyłu). Widoczne jako przesunięcie markera
  w pionie i zmiana proporcji.
- **Roll** (przechylenie) — obrót wokół osi wzdłużnej (robot przechyla
  się na bok). Widoczne jako obrót markera w obrazie.

Algorytm **solvePnP** zwraca macierz rotacji, z której można wyodrębnić
te trzy kąty (tzw. kąty Eulera lub Tait-Bryana).

### 5.5. Prędkość i przyspieszenie — śledzenie w czasie

Gdy kamera rejestruje strumień wideo (a nie pojedyncze zdjęcia),
z kolejnych pozycji markerów można obliczać **pochodne czasowe**:

- **Prędkość** — zmiana pozycji markera w pikselach na klatkę
  (lub cm/s, jeśli znana jest odległość i częstotliwość klatek).
- **Przyspieszenie** — zmiana prędkości w czasie, przydatna do
  prognozowania trajektorii.

Te parametry są kluczowe w systemach śledzenia obiektów, takich jak
`CentroidTracker` w **robo-vision**, który przypisuje trwałe
identyfikatory (track ID) wykrytym obiektom między kolejnymi klatkami.

---

## 6. Techniki zwiększania dokładności pozycjonowania

### 6.1. Kalibracja kamery

Kalibracja kamery to fundamentalny krok, który wyznacza:
- **Macierz wewnętrzną kamery** (ogniskowa, punkt główny) — pozwala
  przekształcić współrzędne pikselowe na kierunki promieni w
  przestrzeni 3D.
- **Współczynniki dystorsji** (zniekształcenia soczewki) — pozwalają
  naprostować obraz przed detekcją markerów.

Bez kalibracji błędy pozycji mogą sięgać kilku centymetrów nawet na
niewielkich odległościach. Szczegółowy opis procesu kalibracji znajduje
się w dokumencie `kalibracja_kamery.md` w tym repozytorium.

### 6.2. Filtr Kalmana i śledzenie predykcyjne

Filtr Kalmana to matematyczne narzędzie, które łączy predykcję ruchu
obiektu z pomiarami z kamery, aby uzyskać wygładzoną i dokładniejszą
trajektorię.

W **robo-vision** klasa `CentroidTracker` oferuje opcjonalne
śledzenie Kalmana (`use_kalman=True`) dla obiektów bez etykiet (np.
punktów laserowych). Filtr pracuje z wektorem stanu
`[x, y, vx, vy]` (pozycja + prędkość) i modelem stałej prędkości:

| Parametr filtru | Wartość | Rola |
|---|---|---|
| Stan: `[x, y, vx, vy]` | 4 wymiary | Pozycja i prędkość w pikselach |
| Pomiar: `[x, y]` | 2 wymiary | Zmierzony środek obiektu |
| Szum procesu (Q) | 0,05 × I₄ | Modeluje nieprzewidywalność ruchu |
| Szum pomiaru (R) | 5,0 × I₂ | Modeluje niedokładność detekcji |

**Jak filtr Kalmana poprawia nawigację:**
1. **Predykcja** — na podstawie poprzedniej pozycji i prędkości filtr
   przewiduje, gdzie obiekt powinien się pojawić.
2. **Korekta** — nowy pomiar (detekcja markera) jest łączony z predykcją
   z wagami proporcjonalnymi do niepewności każdego z nich.
3. **Wynik** — wygładzona pozycja, która jest dokładniejsza niż sam
   pomiar, ponieważ uwzględnia historię ruchu.

Dzięki predykcji filtr potrafi także „przeskoczyć" klatki, w których
marker chwilowo zniknął (okluzja, rozmycie), co utrzymuje ciągłość
śledzenia.

### 6.3. Fuzja wielu markerów

Gdy w polu widzenia kamery widocznych jest jednocześnie kilka markerów,
system może **uśrednić** obliczone pozycje, co zmniejsza wpływ błędu
pojedynczego pomiaru.

Strategia fuzji wielu markerów:

| Metoda | Opis |
|---|---|
| **Średnia arytmetyczna** | Prosta i skuteczna — używana w `compute_offset()` w **robo-vision** |
| **Średnia ważona odległością** | Bliższe markery mają większy wpływ (mniejszy błąd perspektywy) |
| **Odrzucanie outlierów** | Markery, których offset drastycznie odbiega od mediany, są ignorowane |
| **Optymalizacja nieliniowa** | Minimalizacja błędu reprojection error dla wszystkich markerów jednocześnie (np. algorytm Levenberga-Marquardta) |

### 6.4. Fuzja sensorów — IMU, enkodery, LiDAR

Sama wizja maszynowa ma ograniczenia (rozmycie ruchu, zmienne
oświetlenie, chwilowy brak widoczności markerów). W profesjonalnych
systemach robotycznych informację wizyjną łączy się z innymi czujnikami:

| Czujnik | Co mierzy | Jak uzupełnia wizję |
|---|---|---|
| **IMU** (żyroskop + akcelerometr) | Kąty obrotu, przyspieszenie | Wypełnia luki, gdy marker jest niewidoczny; szybka odpowiedź |
| **Enkodery kół** | Obroty kół, odometria | Ciągły pomiar przejechanej drogi (drift w czasie) |
| **LiDAR** | Odległości do przeszkód (chmura punktów) | Precyzyjne mapowanie otoczenia, SLAM |
| **UWB** (Ultra-Wideband) | Odległość do nadajników | Bezwzględna pozycja wewnątrz budynku (±10–30 cm) |
| **GPS / RTK-GPS** | Pozycja globalna | Nawigacja na otwartej przestrzeni (±2 cm z RTK) |

Algorytm **Extended Kalman Filter** (EKF) lub **Unscented Kalman
Filter** (UKF) jest standardowym narzędziem fuzji tych danych — łączy
pomiary o różnej częstotliwości i dokładności w jedną, optymalną
estymację pozycji.

### 6.5. Wiele kamer i stereo-wizja

Zastosowanie wielu kamer znacząco poprawia jakość lokalizacji:

- **Stereo-wizja** — dwie kamery o znanej bazie (odległości między nimi)
  umożliwiają obliczenie głębi (odległości) dla każdego piksela metodą
  triangulacji. Eliminuje to potrzebę znania rozmiaru fizycznego
  markera.
- **Kamery wielokierunkowe** — montaż kamer dookoła robota (np. 4
  kamery po 90°) zapewnia widoczność markerów z każdej strony.
- **Kamera + kamera głębi** (RGB-D, np. Intel RealSense, Kinect) —
  dostarcza bezpośrednią informację o odległości do każdego piksela,
  co czyni estymację pozycji 3D markera trywialną.

### 6.6. Przetwarzanie obrazu — wyostrzanie i adaptacja

Jakość detekcji markerów zależy od jakości obrazu wejściowego. Stosuje
się różne techniki wstępnego przetwarzania:

| Technika | Opis | Kiedy stosować |
|---|---|---|
| **Unsharp masking** (wyostrzanie) | Wzmacnia krawędzie, poprawiając detekcję konturów markera | Rozmazane obrazy z kamer o małej rozdzielczości |
| **Adaptacyjna binaryzacja** | Lokalne progowanie jasności dostosowuje się do nierównomiernego oświetlenia | Sceny z cieniami i zmiennym światłem |
| **Equalizacja histogramu (CLAHE)** | Wyrównuje kontrast lokalny | Słabo oświetlone pomieszczenia |
| **Skalowanie obrazu** (downscale) | Zmniejszenie rozdzielczości przyspiesza detekcję kosztem zasięgu | Systemy czasu rzeczywistego na słabych procesorach |

W **robo-vision** tryb ROBUST automatycznie włącza wyostrzanie
(unsharp mask) obrazu przed detekcją, natomiast tryb FAST zmniejsza
rozdzielczość obrazu o połowę, aby przyspieszyć przetwarzanie.

### 6.7. Tryby pracy — kompromis między szybkością a dokładnością

Systemy wizyjne na platformach o ograniczonych zasobach obliczeniowych
(np. Raspberry Pi) muszą balansować między dokładnością a prędkością
przetwarzania:

| Tryb | Rozdzielczość | Preprocessing | Tracking | FPS | Dokładność |
|---|---|---|---|---|---|
| **NORMAL** | Pełna | Brak | Centroid | Średni | Dobra |
| **FAST** | Połowa (½) | Brak | Centroid | Wysoki | Niższa (mniejszy zasięg) |
| **ROBUST** | Pełna | Unsharp mask | Kalman + centroid | Niski | Najwyższa |

System **robo-vision** oferuje te trzy tryby (enum `DetectionMode`)
przełączane dynamicznie w zależności od bieżących potrzeb robota.

---

## 7. Praktyczne przykłady zastosowań

| Zastosowanie | Typ markera / metoda | Liczba markerów | Kluczowe parametry |
|---|---|---|---|
| **Robot magazynowy (AGV)** | AprilTagi na suficie + linia na podłodze | 50–200 (sufit) + linia ciągła | Pozycja 2D (x, y), kąt yaw |
| **Dron w hali** | AprilTagi na podłodze | 10–50 | Pełna poza 6-DoF, wysokość lotu |
| **Ramię robotyczne (pick & place)** | AprilTag na przedmiocie + ArUco na stole | 2–5 | Pozycja 3D chwytaka, orientacja obiektu |
| **Robot sprzątający** | Visual SLAM + cechy naturalne | 0 (bez markerów) | Mapa 2D, odometria wizualna |
| **Robot edukacyjny (line follower)** | Czarna linia na podłodze + kolorowe znaczniki | 0 markerów + linia + punkty kolorowe | Odchylenie od linii, detekcja skrzyżowań |
| **System kontroli jakości** | AprilTagi na taśmie produkcyjnej | 1 na każdy obiekt | Pozycja obiektu, kąt, odległość |
| **Lokalizacja w muzeum / szpitalu** | Kody QR na ścianach | 1 na pokój/strefę | Identyfikator strefy, kierunek |

---

## 8. Podsumowanie

Markery wizyjne, takie jak AprilTagi, stanowią jedno z najprostszych
i najskuteczniejszych narzędzi do pozycjonowania robotów w przestrzeni.
Ich siła tkwi w prostocie — wydrukowany wzór bitmapowy o znanym
rozmiarze wystarczy, aby za pomocą pojedynczej kamery obliczyć pełną
pozycję i orientację 3D robota.

**Kluczowe wnioski:**

1. **Jeden marker wystarczy** do wyznaczenia pozy, ale **więcej markerów
   = większa dokładność** dzięki uśrednianiu i detekcji błędnych
   odczytów.
2. **AprilTagi przewyższają kody QR** w robotyce dzięki szybszej
   detekcji, precyzyjniejszej lokalizacji narożników i wbudowanej
   korekcji błędów.
3. **Alternatywne metody** (linie, kolory, kształty, cechy naturalne)
   mogą uzupełniać lub zastępować markery — każda ma swoje zalety
   i ograniczenia.
4. **Kalibracja kamery** to konieczny krok dla dokładnego pozycjonowania
   — bez niej błędy mogą być wielokrotnie większe.
5. **Filtr Kalmana** i **fuzja sensorów** znacząco podnoszą niezawodność
   i dokładność, szczególnie w dynamicznych warunkach.
6. **Dobór trybu pracy** (szybkość vs. dokładność) pozwala dopasować
   system do możliwości sprzętowych platformy robota.

Systemy takie jak **robo-vision** pokazują, że nawet stosunkowo
prosty stos technologiczny (kamera + AprilTagi + filtr Kalmana) może
zapewnić niezawodną nawigację w środowiskach wewnętrznych, o ile
zachowane są dobre praktyki: kalibracja kamery, przemyślane rozmieszczenie
markerów, odpowiedni rozmiar tagów i dostosowanie parametrów detekcji
do warunków pracy.
