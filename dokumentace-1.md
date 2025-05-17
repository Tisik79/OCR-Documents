# Instalační a uživatelská příručka pro OCR skript na zpracování dokladů

## Obsah
1. [Úvod](#úvod)
2. [Požadavky](#požadavky)
3. [Instalace](#instalace)
   - [Instalace závislostí](#instalace-závislostí)
   - [Instalace Tesseract OCR](#instalace-tesseract-ocr)
4. [Použití skriptu](#použití-skriptu)
   - [Základní použití](#základní-použití)
   - [Parametry příkazové řádky](#parametry-příkazové-řádky)
   - [Příklady použití](#příklady-použití)
5. [Integrační příručka](#integrační-příručka)
   - [Použití jako modul](#použití-jako-modul)
   - [Integrace do vlastní aplikace](#integrace-do-vlastní-aplikace)
6. [Struktura výstupního JSON](#struktura-výstupního-json)
7. [Řešení problémů](#řešení-problémů)
8. [Omezení](#omezení)
9. [Tipy pro zlepšení přesnosti](#tipy-pro-zlepšení-přesnosti)

## Úvod

Tento skript slouží k automatickému rozpoznávání a extrakci údajů z českých občanských průkazů a cestovních pasů. Využívá počítačové vidění (knihovna OpenCV) pro detekci relevantních oblastí v dokladu a optické rozpoznávání znaků (OCR) pomocí knihovny Tesseract pro převod textu z obrázku do strojově čitelné podoby.

Skript je navržen tak, aby fungoval s různými typy a formáty dokladů, ale jeho přesnost může záviset na kvalitě vstupního obrázku a podmínkách skenování.

## Požadavky

Pro správnou funkci skriptu potřebujete:

- Python 3.6 nebo novější
- Následující Python knihovny:
  - OpenCV (cv2)
  - NumPy
  - pytesseract
  - Pillow (PIL)
- Tesseract OCR Engine (verze 4.0 nebo novější)
- Český jazykový balíček pro Tesseract