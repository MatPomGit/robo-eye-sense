# GitHub Releases i GitHub Packages — kompletny przewodnik

Dokument opisuje dwie kluczowe usługi platformy GitHub służące do dystrybucji
oprogramowania: **GitHub Releases** oraz **GitHub Packages**. Zawiera wyjaśnienie
czym są, jak działają, jak samodzielnie je tworzyć oraz jakich zasad należy
przestrzegać.

---

## Spis treści

1. [GitHub Releases](#1-github-releases)
   1. [Czym są GitHub Releases](#11-czym-są-github-releases)
   2. [Jak działają GitHub Releases](#12-jak-działają-github-releases)
   3. [Tworzenie Release — krok po kroku](#13-tworzenie-release--krok-po-kroku)
   4. [Automatyzacja za pomocą GitHub Actions](#14-automatyzacja-za-pomocą-github-actions)
   5. [Zasady i dobre praktyki](#15-zasady-i-dobre-praktyki)
2. [GitHub Packages](#2-github-packages)
   1. [Czym są GitHub Packages](#21-czym-są-github-packages)
   2. [Obsługiwane rejestry pakietów](#22-obsługiwane-rejestry-pakietów)
   3. [Jak działają GitHub Packages](#23-jak-działają-github-packages)
   4. [Publikowanie pakietu — krok po kroku](#24-publikowanie-pakietu--krok-po-kroku)
   5. [Automatyzacja za pomocą GitHub Actions](#25-automatyzacja-za-pomocą-github-actions)
   6. [Zasady i dobre praktyki](#26-zasady-i-dobre-praktyki)
3. [Porównanie GitHub Releases i GitHub Packages](#3-porównanie-github-releases-i-github-packages)
4. [Podsumowanie](#4-podsumowanie)

---

## 1. GitHub Releases

### 1.1. Czym są GitHub Releases

GitHub Releases to mechanizm platformy GitHub pozwalający na formalne
publikowanie wersji oprogramowania. Release opiera się na tagu Git i rozszerza
go o dodatkowe metadane: tytuł, opis zmian (release notes) oraz opcjonalne
pliki binarne (assets) do pobrania.

Releases pełnią kilka ważnych funkcji:

| Funkcja | Opis |
|---|---|
| **Punkt odniesienia** | Wyznacza konkretną, stabilną wersję kodu, do której można się odwołać |
| **Dystrybucja** | Umożliwia dołączenie skompilowanych plików binarnych, archiwów i instalatorów |
| **Dokumentacja zmian** | Zawiera opis zmian (changelog) pomiędzy wersjami |
| **Komunikacja** | Informuje użytkowników i współpracowników o nowych wersjach projektu |

Releases są widoczne na stronie repozytorium w zakładce **Releases** i mogą
być pobierane zarówno przez interfejs webowy, jak i poprzez API GitHub.

### 1.2. Jak działają GitHub Releases

Mechanizm releases opiera się na tagach Git, które wskazują na konkretne
commity w historii repozytorium.

```
commit A ── commit B ── commit C ── commit D  (main)
                          │
                       tag v1.0.0
                          │
                     Release v1.0.0
                     ├── Tytuł
                     ├── Opis zmian
                     └── Assets (pliki do pobrania)
```

**Przepływ działania:**

1. **Tag Git** — tworzy się tag (lekki lub z adnotacją) wskazujący na
   konkretny commit.
2. **Obiekt Release** — GitHub tworzy obiekt Release powiązany z tagiem.
   Może on zawierać:
   - **Tytuł** — czytelna nazwa wersji (np. `Wersja 1.0.0`).
   - **Opis** — notatki o zmianach, nowe funkcje, poprawki błędów.
   - **Assets** — pliki binarne dołączane do release (np. `.zip`, `.tar.gz`,
     `.exe`, `.deb`).
3. **Archiwum kodu źródłowego** — GitHub automatycznie generuje archiwa
   `.zip` i `.tar.gz` z kodem źródłowym na podstawie tagu.
4. **Powiadomienia** — obserwatorzy repozytorium otrzymują powiadomienie
   o nowym wydaniu.

**Typy wydań:**

| Typ | Opis | Zastosowanie |
|---|---|---|
| **Stabilny release** | Domyślny typ, uznawany za wersję produkcyjną | Wersje gotowe dla użytkowników końcowych |
| **Pre-release** | Oznaczony jako wersja wstępna | Wersje alpha, beta, release candidate (RC) |
| **Draft** | Szkic — niewidoczny publicznie do momentu opublikowania | Przygotowywanie release przed oficjalną publikacją |
| **Latest** | Automatycznie oznaczany najnowszy stabilny release | Wskazuje użytkownikom aktualną zalecaną wersję |

### 1.3. Tworzenie Release — krok po kroku

#### Metoda 1: Przez interfejs webowy GitHub

1. Przejdź do repozytorium na GitHub.
2. W menu bocznym kliknij zakładkę **Releases** (lub wejdź na
   `https://github.com/<OWNER>/<REPO>/releases`).
3. Kliknij przycisk **Draft a new release**.
4. **Wybierz tag** — wybierz istniejący tag lub utwórz nowy wpisując nazwę
   (np. `v1.0.0`) i wybierając gałąź docelową.
5. **Tytuł** — podaj czytelny tytuł release (np. `Wersja 1.0.0`).
6. **Opis** — wpisz opis zmian. Możesz:
   - Napisać go ręcznie w formacie Markdown.
   - Kliknąć **Generate release notes**, aby GitHub automatycznie wygenerował
     opis na podstawie pull requestów i commitów od ostatniego tagu.
7. **Assets** — opcjonalnie przeciągnij lub wybierz pliki binarne do
   dołączenia (np. skompilowane programy, paczki instalacyjne).
8. **Opcje**:
   - Zaznacz **Set as pre-release**, jeśli to wersja wstępna.
   - Zaznacz **Set as the latest release**, aby oznaczyć jako najnowszą.
9. Kliknij **Publish release**.

#### Metoda 2: Przez wiersz poleceń (GitHub CLI)

```bash
# Instalacja GitHub CLI (jeśli jeszcze nie zainstalowany)
# Ubuntu/Debian:
sudo apt install gh

# Uwierzytelnienie
gh auth login

# Utworzenie tagu (jeśli jeszcze nie istnieje)
git tag -a v1.0.0 -m "Wersja 1.0.0"
git push origin v1.0.0

# Utworzenie release
gh release create v1.0.0 \
  --title "Wersja 1.0.0" \
  --notes "Opis zmian w wersji 1.0.0" \
  ./dist/program.zip ./dist/program.tar.gz

# Utworzenie release z automatycznie wygenerowanymi notatkami
gh release create v1.0.0 --generate-notes

# Utworzenie pre-release
gh release create v1.1.0-beta.1 --prerelease --generate-notes
```

#### Metoda 3: Przez API REST GitHub

```bash
# Utworzenie release za pomocą API
curl -X POST \
  -H "Authorization: token TWOJ_TOKEN" \
  -H "Accept: application/vnd.github+json" \
  https://api.github.com/repos/OWNER/REPO/releases \
  -d '{
    "tag_name": "v1.0.0",
    "target_commitish": "main",
    "name": "Wersja 1.0.0",
    "body": "Opis zmian w wersji 1.0.0",
    "draft": false,
    "prerelease": false,
    "generate_release_notes": true
  }'
```

### 1.4. Automatyzacja za pomocą GitHub Actions

Releases można tworzyć automatycznie przy pomocy GitHub Actions. Poniżej
przykładowy workflow, który tworzy release po wypchnięciu tagu:

```yaml
# .github/workflows/release.yml
name: Utwórz Release

on:
  push:
    tags:
      - 'v*'  # Uruchom przy tagach zaczynających się od "v"

permissions:
  contents: write

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Pobierz kod
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Pełna historia dla generowania notatek

      - name: Utwórz Release na GitHub
        uses: softprops/action-gh-release@v2
        with:
          generate_release_notes: true
          files: |
            dist/*.zip
            dist/*.tar.gz
```

**Przepływ automatyczny:**

```
git tag v1.0.0 → git push --tags → GitHub Actions → Release tworzony automatycznie
```

### 1.5. Zasady i dobre praktyki

1. **Stosuj wersjonowanie semantyczne (SemVer)** — używaj formatu
   `MAJOR.MINOR.PATCH` (np. `v1.2.3`), gdzie:
   - `MAJOR` — zmiany łamiące wsteczną kompatybilność,
   - `MINOR` — nowe funkcje kompatybilne wstecz,
   - `PATCH` — poprawki błędów kompatybilne wstecz.

2. **Stosuj prefiksy tagów** — stosuj konwencję `v` przed numerem wersji
   (np. `v1.0.0`), co jest powszechnie przyjętą praktyką.

3. **Używaj tagów z adnotacją** — preferuj tagi z adnotacją
   (`git tag -a v1.0.0 -m "opis"`) zamiast lekkich, ponieważ przechowują
   informacje o autorze, dacie i komunikacie.

4. **Opisuj zmiany dokładnie** — release notes powinny zawierać:
   - Listę nowych funkcji.
   - Listę poprawek błędów.
   - Zmiany łamiące wsteczną kompatybilność (breaking changes).
   - Instrukcje migracji (jeśli dotyczą).

5. **Korzystaj z automatycznego generowania notatek** — GitHub potrafi
   automatycznie wygenerować release notes na podstawie PR-ów i commitów.

6. **Oznaczaj pre-release** — wersje niestabilne (alpha, beta, RC) powinny
   być wyraźnie oznaczone jako pre-release, aby użytkownicy wiedzieli, że
   mogą zawierać błędy.

7. **Nie usuwaj opublikowanych releases** — usunięcie release może złamać
   linki i zależności u użytkowników.

8. **Dołączaj pliki binarne** — jeśli projekt wymaga kompilacji, dołączaj
   gotowe pliki binarne dla popularnych platform, aby ułatwić instalację.

9. **Automatyzuj proces** — wykorzystaj GitHub Actions do automatycznego
   tworzenia releases, co eliminuje ryzyko ludzkiego błędu.

---

## 2. GitHub Packages

### 2.1. Czym są GitHub Packages

GitHub Packages to usługa hostingu pakietów oprogramowania zintegrowana
z platformą GitHub. Pozwala na publikowanie, przechowywanie i zarządzanie
pakietami bezpośrednio w ramach repozytorium GitHub. Dzięki temu kod
źródłowy i jego skompilowane pakiety mogą współistnieć w jednym miejscu.

GitHub Packages obsługuje wiele popularnych ekosystemów pakietów i rejestrów
kontenerów, co czyni go uniwersalnym narzędziem do dystrybucji
oprogramowania.

**Kluczowe cechy:**

| Cecha | Opis |
|---|---|
| **Integracja z GitHub** | Pakiety są powiązane z repozytoriami, issues i pull requestami |
| **Kontrola dostępu** | Uprawnienia dziedziczone z repozytorium lub konfigurowane niezależnie |
| **Wiele rejestrów** | Obsługa npm, Maven, Gradle, NuGet, RubyGems, Docker (Container Registry) |
| **Wersjonowanie** | Każdy pakiet może mieć wiele wersji |
| **Darmowe dla publicznych repozytoriów** | Nieograniczony transfer i przechowywanie dla projektów open source |

### 2.2. Obsługiwane rejestry pakietów

GitHub Packages obsługuje następujące rejestry:

| Rejestr | Ekosystem | Adres rejestru | Klient |
|---|---|---|---|
| **npm** | JavaScript / Node.js | `npm.pkg.github.com` | `npm` / `yarn` |
| **Maven** | Java | `maven.pkg.github.com` | `mvn` / `gradle` |
| **Gradle** | Java / Kotlin | `maven.pkg.github.com` | `gradle` |
| **NuGet** | .NET | `nuget.pkg.github.com` | `dotnet nuget` |
| **RubyGems** | Ruby | `rubygems.pkg.github.com` | `gem` |
| **Container Registry** | Docker / OCI | `ghcr.io` | `docker` / `podman` |

> **Uwaga:** Rejestr kontenerów (Container Registry, `ghcr.io`) zastąpił
> starszy Docker Registry (`docker.pkg.github.com`). Nowe projekty powinny
> używać wyłącznie `ghcr.io`.

### 2.3. Jak działają GitHub Packages

GitHub Packages działa jako rejestr pakietów — serwer, który przechowuje
i udostępnia pakiety oprogramowania. Przepływ pracy wygląda następująco:

```
Deweloper                    GitHub Packages              Użytkownik
    │                              │                          │
    ├── Buduje pakiet ────────────>│                          │
    ├── Publikuje (push) ─────────>│                          │
    │                              ├── Przechowuje pakiet     │
    │                              ├── Zarządza wersjami      │
    │                              │                          │
    │                              │<── Pobiera (pull) ───────┤
    │                              │<── Instaluje ────────────┤
```

**Szczegółowy przebieg:**

1. **Uwierzytelnienie** — aby publikować pakiety, wymagane jest
   uwierzytelnienie za pomocą tokena osobistego (PAT) z odpowiednimi
   uprawnieniami lub tokena `GITHUB_TOKEN` w kontekście GitHub Actions.

2. **Konfiguracja klienta** — klient pakietów (np. `npm`, `docker`) musi
   zostać skonfigurowany tak, aby wskazywał na rejestr GitHub zamiast
   domyślnego rejestru publicznego.

3. **Publikacja** — pakiet jest budowany lokalnie lub w CI/CD, a następnie
   przesyłany (push) do rejestru GitHub Packages.

4. **Przechowywanie** — pakiet jest przechowywany i powiązany z konkretnym
   repozytorium (lub organizacją w przypadku Container Registry).

5. **Instalacja** — użytkownicy mogą pobrać i zainstalować pakiet,
   konfigurując swój klient pakietów do korzystania z rejestru GitHub.

**Uprawnienia i widoczność:**

| Widoczność pakietu | Kto może pobierać | Kto może publikować |
|---|---|---|
| **Publiczny** | Każdy (bez uwierzytelnienia) | Osoby z uprawnieniami do zapisu w repozytorium |
| **Prywatny** | Tylko osoby z dostępem do repozytorium | Osoby z uprawnieniami do zapisu w repozytorium |
| **Wewnętrzny** (organizacje) | Członkowie organizacji | Osoby z uprawnieniami do zapisu |

### 2.4. Publikowanie pakietu — krok po kroku

Poniżej przedstawiono przykłady publikowania pakietów w dwóch popularnych
rejestrach.

#### Przykład A: Pakiet npm

**1. Konfiguracja pliku `.npmrc`**

Utwórz lub edytuj plik `.npmrc` w katalogu głównym projektu:

```
@OWNER:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${NODE_AUTH_TOKEN}
```

Gdzie `OWNER` to nazwa użytkownika lub organizacji na GitHub.

**2. Konfiguracja `package.json`**

```json
{
  "name": "@OWNER/nazwa-pakietu",
  "version": "1.0.0",
  "publishConfig": {
    "registry": "https://npm.pkg.github.com"
  },
  "repository": {
    "type": "git",
    "url": "https://github.com/OWNER/REPO.git"
  }
}
```

> **Ważne:** Nazwa pakietu musi być poprzedzona scope organizacji lub
> użytkownika (np. `@mojafirma/moj-pakiet`).

**3. Uwierzytelnienie**

```bash
# Eksportuj token (PAT z uprawnieniem read:packages i write:packages)
export NODE_AUTH_TOKEN=ghp_TWOJ_TOKEN

# Lub zaloguj się interaktywnie
npm login --registry=https://npm.pkg.github.com
```

**4. Publikacja**

```bash
npm publish
```

**5. Instalacja przez użytkownika**

```bash
# Konfiguracja rejestru dla danego scope
echo "@OWNER:registry=https://npm.pkg.github.com" >> .npmrc

# Instalacja pakietu
npm install @OWNER/nazwa-pakietu
```

#### Przykład B: Obraz Docker (Container Registry)

**1. Uwierzytelnienie w `ghcr.io`**

```bash
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

**2. Zbudowanie obrazu**

```bash
docker build -t ghcr.io/OWNER/nazwa-obrazu:1.0.0 .
```

> **Ważne:** Nazwa obrazu musi być poprzedzona adresem `ghcr.io/OWNER/`.

**3. Wypchnięcie obrazu**

```bash
docker push ghcr.io/OWNER/nazwa-obrazu:1.0.0
```

**4. Pobranie obrazu przez użytkownika**

```bash
docker pull ghcr.io/OWNER/nazwa-obrazu:1.0.0
```

### 2.5. Automatyzacja za pomocą GitHub Actions

Publikowanie pakietów można w pełni zautomatyzować. Poniżej przykładowe
workflow dla npm i Docker.

#### Workflow: Publikacja pakietu npm

```yaml
# .github/workflows/publish-npm.yml
name: Publikuj pakiet npm

on:
  release:
    types: [published]

permissions:
  packages: write
  contents: read

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Pobierz kod
        uses: actions/checkout@v4

      - name: Skonfiguruj Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '20'
          registry-url: 'https://npm.pkg.github.com'

      - name: Zainstaluj zależności
        run: npm ci

      - name: Opublikuj pakiet
        run: npm publish
        env:
          NODE_AUTH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### Workflow: Publikacja obrazu Docker

```yaml
# .github/workflows/publish-docker.yml
name: Publikuj obraz Docker

on:
  release:
    types: [published]

permissions:
  packages: write
  contents: read

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - name: Pobierz kod
        uses: actions/checkout@v4

      - name: Zaloguj do Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Wyodrębnij metadane
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}

      - name: Zbuduj i wypchnij obraz
        uses: docker/build-push-action@v6
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

### 2.6. Zasady i dobre praktyki

1. **Używaj `GITHUB_TOKEN` zamiast PAT w CI/CD** — token automatyczny
   (`secrets.GITHUB_TOKEN`) jest bezpieczniejszy, ponieważ jest
   ograniczony do jednego uruchomienia workflow i repozytorium.

2. **Stosuj scope w nazwach pakietów** — pakiety npm muszą używać scope
   organizacji/użytkownika (`@OWNER/pakiet`). Obrazy Docker muszą
   zawierać prefix `ghcr.io/OWNER/`.

3. **Wersjonuj pakiety konsekwentnie** — stosuj wersjonowanie semantyczne
   (SemVer), synchronizuj wersje pakietów z tagami Git i releases.

4. **Nie publikuj wrażliwych danych** — upewnij się, że pakiet nie zawiera
   sekretów, kluczy API, haseł ani plików konfiguracyjnych z danymi
   dostępowymi. Użyj pliku `.npmignore` lub `.dockerignore`.

5. **Konfiguruj widoczność pakietu** — domyślnie pakiety dziedziczą
   widoczność repozytorium. Świadomie decyduj o tym, czy pakiet ma być
   publiczny, prywatny czy wewnętrzny.

6. **Ustaw minimalnie wymagane uprawnienia** — w workflow GitHub Actions
   definiuj `permissions` z najniższym wystarczającym poziomem uprawnień
   (zasada najmniejszych uprawnień).

7. **Usuwaj nieużywane wersje** — regularnie przeglądaj i usuwaj stare
   lub nieużywane wersje pakietów, aby utrzymać porządek i zmniejszyć
   koszty przechowywania (w przypadku prywatnych repozytoriów).

8. **Linkuj pakiet do repozytorium** — w przypadku Container Registry
   dodaj etykietę `org.opencontainers.image.source` w Dockerfile, aby
   GitHub automatycznie powiązał obraz z repozytorium:
   ```dockerfile
   LABEL org.opencontainers.image.source=https://github.com/OWNER/REPO
   ```

9. **Testuj przed publikacją** — zawsze uruchom pełen zestaw testów
   zanim opublikujesz nową wersję pakietu. Najlepiej zautomatyzuj to
   w pipeline CI/CD.

---

## 3. Porównanie GitHub Releases i GitHub Packages

| Aspekt | GitHub Releases | GitHub Packages |
|---|---|---|
| **Cel** | Publikowanie wersji projektu z opisem zmian | Hostowanie pakietów w rejestrach ekosystemów |
| **Format** | Tagi Git + pliki binarne (assets) | Pakiety natywne (npm, Maven, Docker itp.) |
| **Instalacja** | Ręczne pobranie pliku lub `gh release download` | Instalacja przez menedżer pakietów (`npm install`, `docker pull`) |
| **Wersjonowanie** | Tagi Git (np. `v1.0.0`) | Wersje pakietów natywnych (np. `1.0.0`) |
| **Integracja** | Każde repozytorium, każdy język | Wymaga konfiguracji klienta dla danego rejestru |
| **Odbiorcy** | Użytkownicy końcowi, deweloperzy | Deweloperzy integrujący pakiet jako zależność |
| **Automatyzacja** | Workflow wyzwalany przez tagi lub ręcznie | Workflow wyzwalany przez release lub push |
| **Koszty** | Bezpłatne (limity na rozmiar assets) | Bezpłatne dla publicznych repozytoriów; limity transferu i przestrzeni dla prywatnych |

**Kiedy używać czego?**

- **GitHub Releases** — gdy chcesz udostępnić gotowy produkt (binaria,
  instalatory, archiwa) lub poinformować użytkowników o nowej wersji.
- **GitHub Packages** — gdy tworzysz bibliotekę lub moduł, który inni
  deweloperzy będą dodawać jako zależność do swoich projektów.
- **Oba razem** — w wielu projektach stosuje się oba mechanizmy:
  release wyzwala workflow, który automatycznie publikuje pakiet.

---

## 4. Podsumowanie

GitHub Releases i GitHub Packages to uzupełniające się usługi platformy
GitHub, które wspólnie tworzą kompletny ekosystem dystrybucji
oprogramowania:

- **GitHub Releases** umożliwia formalne publikowanie wersji projektu
  z opisem zmian i dołączonymi plikami binarnymi. Opiera się na tagach
  Git i jest idealny do komunikowania zmian użytkownikom końcowym.

- **GitHub Packages** to w pełni funkcjonalny rejestr pakietów
  obsługujący wiele ekosystemów (npm, Maven, Docker i inne). Pozwala
  na dystrybucję bibliotek i modułów jako zależności do instalacji
  przez menedżery pakietów.

Oba mechanizmy można w pełni zautomatyzować za pomocą **GitHub Actions**,
co pozwala na stworzenie niezawodnego pipeline'u CI/CD: od wypchnięcia
tagu, przez uruchomienie testów, po automatyczną publikację release
i pakietów.

**Najważniejsze zasady do zapamiętania:**

1. Stosuj **wersjonowanie semantyczne** (SemVer) w tagach i pakietach.
2. **Automatyzuj** tworzenie releases i publikację pakietów za pomocą
   GitHub Actions.
3. Korzystaj z `GITHUB_TOKEN` zamiast tokenów osobistych w workflow.
4. Nie publikuj **wrażliwych danych** — używaj `.gitignore`,
   `.npmignore`, `.dockerignore`.
5. Dokumentuj zmiany w **release notes** przy każdym wydaniu.
6. Stosuj zasadę **najmniejszych uprawnień** w konfiguracji workflow.
