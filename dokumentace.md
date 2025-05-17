# Dokumentace OCR skriptu pro skenování dokladů

Tento dokument obsahuje kompletní dokumentaci pro Python skript určený k rozpoznávání údajů z občanských průkazů a pasů pomocí technologie OCR.

## Obsah
1. [Úvod](#úvod)
2. [Požadavky](#požadavky)
3. [Instalace](#instalace)
4. [Použití skriptu](#použití-skriptu)
5. [Struktura výstupního JSON](#struktura-výstupního-json)

## Úvod

Skript slouží k automatickému rozpoznávání a extrakci údajů z českých občanských průkazů a cestovních pasů. Využívá počítačové vidění (knihovna OpenCV) a optické rozpoznávání znaků (OCR) pomocí knihovny Tesseract.

## Požadavky

- Python 3.6+
- OpenCV
- NumPy
- pytesseract
- Pillow (PIL)
- Tesseract OCR Engine (4.0+)
- Český jazykový balíček pro Tesseract

## Instalace

### Instalace Python závislostí:
```bash
pip install opencv-python numpy pytesseract pillow
```

### Instalace Tesseract OCR:

#### Windows:
1. Stáhněte instalační program z: https://github.com/UB-Mannheim/tesseract/wiki
2. Vyberte instalaci českého jazyka
3. Přidejte cestu k Tesseractu do PATH

#### Linux:
```bash
sudo apt update
sudo apt install tesseract-ocr
sudo apt install tesseract-ocr-ces
```

## Použití skriptu

### Základní použití:
```bash
python ocr_doklady.py -i cesta/k/obrazku.jpg
```

### Parametry:
- `-i, --image`: Cesta k obrázku dokladu (povinný)
- `-t, --type`: Typ dokladu ("auto", "pas", "op")
- `-d, --debug`: Debug režim s vizualizací
- `-o, --output`: Výstupní JSON soubor
- `--vizualize`: Zobrazení výsledků

## Struktura výstupního JSON

```json
{
    "cislo_dokladu": "123456789",
    "jmeno": "Jan",
    "prijmeni": "Novák",
    "datum_narozeni": "01.01.1980",
    "pohlavi": "M",
    "statni_prislusnost": "CZE",
    "platnost_do": "01.01.2030"
}
```

## Integrace do aplikací

Skript lze použít jako samostatný nástroj nebo integrovat do aplikací pomocí importu funkcí. Příklad integrace do webové aplikace:

```python
from flask import Flask, request, jsonify
import cv2
import numpy as np
from ocr_doklady import process_document_image

app = Flask(__name__)

@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    data = process_document_image(image)
    
    return jsonify(data)
```

## Řešení problémů

Pokud skript nefunguje správně, zkontrolujte:
- Kvalitu vstupního obrázku
- Správnou instalaci Tesseract
- Viditelnost MRZ zóny na dokladu