# OCR pro skenování dokladů

Tento projekt obsahuje Python skript pro automatické rozpoznávání a extrakci údajů z českých občanských průkazů a cestovních pasů pomocí počítačového vidění (OpenCV) a optického rozpoznávání znaků (Tesseract OCR).

## Funkce

- Automatická detekce typu dokladu (občanský průkaz/pas)
- Detekce a extrakce strojově čitelné zóny (MRZ)
- Rozpoznávání osobních údajů v dokladu
- Kombinace dat z různých částí dokladu pro zvýšení přesnosti
- Export výsledků do JSON formátu
- Vizualizace detekovaných oblastí a výsledků

## Požadavky

- Python 3.6+
- OpenCV (cv2)
- NumPy
- pytesseract
- Pillow (PIL)
- Tesseract OCR Engine (verze 4.0+)
- Český jazykový balíček pro Tesseract

## Instalace

Použijte pip pro instalaci potřebných knihoven:

```bash
pip install opencv-python numpy pytesseract pillow
```

Tesseract OCR lze nainstalovat podle operačního systému:

### Windows

1. Stáhněte instalační program z oficiálních stránek: https://github.com/UB-Mannheim/tesseract/wiki
2. Během instalace vyberte možnost instalace českého jazyka
3. Přidejte cestu k Tesseractu do systémové proměnné PATH (typicky `C:\Program Files\Tesseract-OCR`)
4. V případě potřeby upravte cestu k Tesseractu ve skriptu v proměnné `TESSERACT_CMD`

### Linux
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ces  # český jazykový balíček

# Fedora
sudo dnf install tesseract
sudo dnf install tesseract-langpack-ces
```

### macOS
```bash
# Pomocí Homebrew
brew install tesseract
brew install tesseract-lang  # obsahuje všechny jazykové balíčky včetně češtiny
```

## Použití

### Základní použití

Skript spustíte příkazem:

```bash
python ocr_doklady.py -i cesta/k/obrazku.jpg
```

### Parametry příkazové řádky

- `-i, --image` (povinný): Cesta k obrázku dokladu
- `-t, --type` (volitelný): Typ dokladu - možnosti: "auto" (automatická detekce), "pas" (cestovní pas), "op" (občanský průkaz). Výchozí hodnota je "auto".
- `-d, --debug` (volitelný): Zapíná debug režim s vizualizací jednotlivých kroků zpracování
- `-o, --output` (volitelný): Cesta k výstupnímu JSON souboru pro uložení extrahovaných dat
- `--vizualize` (volitelný): Zobrazí vizualizaci detekovaných oblastí a výsledků OCR

### Příklady použití

Základní rozpoznání občanského průkazu s automatickou detekcí typu dokladu:
```bash
python ocr_doklady.py -i obcansky_prukaz.jpg
```

Explicitní specifikace typu dokladu (pas) a uložení výsledků do JSON:
```bash
python ocr_doklady.py -i cestovni_pas.jpg -t pas -o vysledky.json
```

Spuštění v debug režimu s vizualizací:
```bash
python ocr_doklady.py -i obcansky_prukaz.jpg -d --vizualize
```

## Integrace do aplikací

Skript lze snadno integrovat do vlastních aplikací:

```python
from ocr_doklady import preprocess_image, detect_mrz_region, ocr_mrz, parse_mrz_data

# Načtení a předzpracování obrázku
image = cv2.imread('cesta/k/obrazku.jpg')
processed, _ = preprocess_image(image)

# Detekce a extrakce MRZ zóny
mrz_region, _ = detect_mrz_region(image)
mrz_text = ocr_mrz(mrz_region)

# Parsování dat z MRZ
document_type = "OBCANSKY_PRUKAZ"  # nebo "PAS"
data = parse_mrz_data(mrz_text, document_type)

print(data)
```

## Omezení

- Skript je primárně optimalizován pro české občanské průkazy a pasy
- Přesnost rozpoznávání textu závisí na kvalitě vstupního obrázku
- Pro doklady s výrazným poškozením nebo nečitelnou MRZ zónou může být přesnost nízká

## Tipy pro zlepšení přesnosti

1. Použijte kvalitní skener nebo fotoaparát s dobrým osvětlením
2. Ujistěte se, že je doklad správně zaostřen
3. Vyhněte se fotografování s reflexními odlesky
4. Pro problémové obrázky můžete před použitím skriptu aplikovat dodatečné předzpracování (např. úprava kontrastu, odstranění šumu)

## Licence

Tento projekt je distribuován pod licencí MIT. Viz soubor `LICENSE` pro více informací.