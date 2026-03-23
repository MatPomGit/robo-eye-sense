# Analiza utraty śledzenia AprilTagów podczas ruchów głowy robota

## Problem
Podczas chodu robot buja głową na boki, więc kamera wykonuje szybkie ruchy kątowe. To prowadzi do kilku nakładających się problemów:

1. **Rozmycie ruchowe** – krawędzie AprilTaga są mniej ostre, więc dekoder częściej odrzuca znacznik.
2. **Duży skok położenia między klatkami** – nawet gdy tag pozostaje w polu widzenia, jego pozycja zmienia się gwałtownie.
3. **Krótkie wypadnięcie z kadru** – przy bocznym wychyleniu głowy marker może zniknąć na 1–4 klatki.
4. **Oscylacja sygnału sterującego** – sterownik reaguje na ruch głowy zamiast na realne położenie znacznika w świecie.

## Źródła problemu w obecnej architekturze
Repozytorium już zawiera dwa elementy częściowo pomagające w takich warunkach:

- tryb `ROBUST` z wyostrzaniem obrazu i bardziej tolerancyjnym trackerem,
- stabilne ID dla etykietowanych obiektów.

Jednak dla samego trybu follow brakowało warstwy **czasowej stabilizacji celu**. Gdy marker znikał na chwilę, wektor śledzenia natychmiast wracał do zera. To powodowało szarpanie sterowania i utratę celu dokładnie w fazie bocznego ruchu głowy.

## Wprowadzone rozwiązania

### 1. Filtracja czasowa pozycji celu
Do `AutoFollowScenario` dodano wykładnicze wygładzanie pozycji markera. Dzięki temu sterowanie mniej reaguje na szybkie drgania kamery i pojedyncze błędne odczyty środka taga.

### 2. Krótkoterminowa predykcja po utracie znacznika
Scenariusz follow utrzymuje ostatnią pozycję i prędkość celu, a po chwilowej utracie markera przewiduje jego położenie jeszcze przez kilka klatek. To pozwala przetrwać typowe zaniki powodowane bujaniem głowy.

### 3. Kompensacja yaw głowy/kamery
API scenariusza follow przyjmuje opcjonalny `camera_yaw_deg`. Zwracany jest także `compensated_yaw`, czyli yaw celu po odjęciu bieżącego wychylenia głowy. Jeśli robot ma IMU albo zna kąt serwa szyi, sterownik może odfiltrować sam ruch głowy od ruchu celu.

### 4. Jawny stan śledzenia
Wynik follow zwraca teraz `tracking_state`:

- `detected` – marker widoczny,
- `predicted` – marker chwilowo niewidoczny, ale tor jest podtrzymany predykcją,
- `lost` – brak wiarygodnej estymacji.

Taki stan można bezpośrednio wykorzystać w logice sterowania: np. w `predicted` ograniczyć prędkość obrotu, a w `lost` przejść do skanowania.

## Co jeszcze warto wdrożyć sprzętowo/systemowo

### Priorytet A – stabilizacja mechaniczna
- usztywnić mocowanie kamery,
- zmniejszyć amplitudę ruchu głowy podczas fazy śledzenia,
- jeśli to możliwe, dodać mini-gimbal albo pasywne tłumienie drgań.

### Priorytet B – synchronizacja z kinematyką robota
Najsilniejsze rozwiązanie programowe to kompensacja ruchem własnym:

- odczyt kąta serwa szyi (`head_yaw`, `head_pitch`),
- albo IMU w głowie,
- transformacja pomiaru taga do układu bazowego robota.

Wtedy sterowanie opiera się na pozycji celu względem korpusu, a nie względem chwilowo wychylonej kamery.

### Priorytet C – parametry wizyjne
- używać krótszego czasu ekspozycji,
- zwiększyć FPS kamery,
- uruchamiać detekcję w trybie `ROBUST`,
- dobrać większe tagi lub zmniejszyć odległość od nich,
- pilnować dobrego oświetlenia.

## Zalecana strategia sterowania
Najbardziej praktyczny pipeline dla chodzącego robota:

1. Kamera pracuje w wysokim FPS i krótkiej ekspozycji.
2. Detekcja działa w trybie `ROBUST`.
3. Follow używa wygładzania + krótkiej predykcji.
4. Jeśli dostępny jest kąt głowy, sterownik używa `compensated_yaw`.
5. W stanie `predicted` robot ogranicza agresywność ruchu.
6. Dopiero po kilku klatkach `lost` uruchamiane jest aktywne ponowne wyszukiwanie taga.

## Efekt oczekiwany
Po tych zmianach system powinien:

- rzadziej gubić taga przy bocznym bujaniu głowy,
- generować mniej oscylacyjny sygnał sterowania,
- lepiej przetrwać krótkie zaniki detekcji,
- łatwiej integrować się z IMU albo odczytem serw szyi.
