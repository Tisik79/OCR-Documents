#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OCR Skript pro skenování občanských průkazů a pasů
=================================================
Tento skript umožňuje detekci a extrakci údajů z českých občanských průkazů 
a cestovních pasů pomocí OCR. Využívá knihovny OpenCV pro zpracování obrazu
a Tesseract pro OCR.

Autor: Claude
Datum: 17.5.2025
"""

import os
import sys
import argparse
import re
import json
import numpy as np
import cv2
import pytesseract
from PIL import Image

# Konfigurace
TESSERACT_CMD = r'tesseract'  # Cesta k tesseract executable - upravte dle vašeho systému
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Definice typů dokladů
PASSPORT = "PAS"
ID_CARD = "OBCANSKY_PRUKAZ"

def configure_arguments():
    """Konfigurace argumentů příkazové řádky"""
    ap = argparse.ArgumentParser(description="OCR skenování občanských průkazů a pasů")
    ap.add_argument("-i", "--image", required=True, help="Cesta k obrázku dokladu")
    ap.add_argument("-t", "--type", default="auto", choices=["auto", "pas", "op"],
                   help="Typ dokladu: auto = automatická detekce, pas = cestovní pas, op = občanský průkaz")
    ap.add_argument("-d", "--debug", action="store_true", help="Zapnout debug režim s vizualizací kroků")
    ap.add_argument("-o", "--output", help="Výstupní JSON soubor s daty")
    ap.add_argument("--vizualize", action="store_true", help="Zobrazit vizualizaci detekovaných oblastí")
    return vars(ap.parse_args())

def preprocess_image(image, debug=False):
    """
    Předzpracování obrázku pro lepší OCR výsledky
    """
    # Převod na grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplikace Gaussova rozostření pro redukci šumu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptivní threshold pro lepší kontrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Aplikace morfologických operací pro odstranění šumu
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Dilatace pro zvýraznění textu
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(opening, kernel, iterations=1)
    
    # Inverze zpět, aby text byl černý na bílém pozadí
    processed = cv2.bitwise_not(dilated)
    
    if debug:
        cv2.imshow("Originální obrázek", image)
        cv2.imshow("Šedotónový", gray)
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Zpracovaný obrázek", processed)
        cv2.waitKey(0)
    
    return processed, gray

def detect_mrz_region(image, debug=False):
    """
    Detekce oblasti MRZ (strojově čitelná zóna) v dokladu
    """
    # Kopie obrázku pro vizualizaci
    vis_image = image.copy()
    
    # Převod na grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplikace Gaussova rozostření
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplikace Blackhat morfologické operace pro detekci tmavého textu na světlém pozadí
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    blackhat = cv2.morphologyEx(blurred, cv2.MORPH_BLACKHAT, rectKernel)
    
    if debug:
        cv2.imshow("Blackhat", blackhat)
    
    # Výpočet gradientu pro zvýraznění MRZ
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
    
    # Aplikace closing operace
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    
    # Threshold pro binarizaci
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Další closing operace pro spojení blízkých oblastí
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, squareKernel)
    
    if debug:
        cv2.imshow("Threshold", thresh)
    
    # Hledání kontur v obrázku
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Inicializace proměnných pro nejlepší konturu
    best_contour = None
    max_width = 0
    
    # Procházení kontur a hledání té, která odpovídá MRZ
    for contour in contours:
        # Výpočet ohraničujícího obdélníku
        (x, y, w, h) = cv2.boundingRect(contour)
        
        # Filtrování kontur podle rozměrů - MRZ má specifický poměr stran
        aspect_ratio = w / float(h)
        
        # MRZ by měl mít určitou minimální šířku a specifický poměr stran
        if w > 100 and aspect_ratio > 4 and aspect_ratio < 12:
            if w > max_width:
                max_width = w
                best_contour = contour
    
    # Pokud nebyla nalezena žádná vhodná kontura
    if best_contour is None:
        return None, vis_image
    
    # Výpočet ohraničujícího obdélníku pro nejlepší konturu
    (x, y, w, h) = cv2.boundingRect(best_contour)
    
    # Zvětšení oblasti pro zahrnutí celého MRZ
    y = max(0, y - 5)
    h = min(gray.shape[0] - y, h + 10)
    
    # Extrakce oblasti MRZ
    mrz_region = gray[y:y+h, x:x+w]
    
    # Vizualizace detekované oblasti
    if debug:
        cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis_image, "MRZ", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detekovaná MRZ oblast", vis_image)
        cv2.imshow("MRZ region", mrz_region)
    
    return mrz_region, vis_image

def detect_document_type(mrz_text):
    """
    Detekce typu dokladu na základě textu MRZ zóny
    """
    if mrz_text and "IDCZE" in mrz_text:
        return ID_CARD
    elif mrz_text and "P<CZE" in mrz_text:
        return PASSPORT
    else:
        return None

def preprocess_mrz_for_ocr(mrz_region):
    """
    Speciální předzpracování MRZ oblasti pro lepší OCR výsledky
    """
    # Zvětšení obrázku pro lepší OCR
    mrz_region = cv2.resize(mrz_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # Adaptivní threshold pro lepší kontrast
    mrz_region = cv2.adaptiveThreshold(mrz_region, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 11, 2)
    
    # Aplikace Gaussova rozostření
    mrz_region = cv2.GaussianBlur(mrz_region, (3, 3), 0)
    
    # Opětovná aplikace thresholdu
    mrz_region = cv2.threshold(mrz_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    return mrz_region

def ocr_mrz(mrz_region):
    """
    OCR na oblasti MRZ s použitím specializovaných tesseract parametrů
    """
    # Speciální konfigurace tesseract pro MRZ
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<" -l eng'
    
    # Konverze numpy array na PIL Image
    mrz_pil = Image.fromarray(mrz_region)
    
    # OCR
    mrz_text = pytesseract.image_to_string(mrz_pil, config=custom_config)
    
    # Odstranění mezer a nových řádků
    mrz_text = mrz_text.replace(" ", "").strip()
    
    return mrz_text

def parse_personal_data_region(image, document_type, debug=False):
    """
    Detekce a extrakce oblasti s osobními údaji v dokladu
    """
    # Vytvořit kopii obrázku pro vizualizaci
    vis_image = image.copy()
    
    # Převod na grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplikace Gaussova rozostření
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold pro binarizaci
    thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Hledání kontur
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Seřazení kontur podle velikosti (sestupně)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Výběr největších kontur
    largest_contours = contours[:10]
    
    # Inicializace oblasti s osobními údaji
    personal_data_region = None
    
    if document_type == ID_CARD:
        # Hledání oblasti s osobními údaji podle pozice a velikosti
        for contour in largest_contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            
            # Filtrování kontur podle velikosti a pozice
            if w > 100 and h > 100 and y < gray.shape[0] * 0.6:
                personal_data_region = gray[y:y+h, x:x+w]
                
                if debug:
                    cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(vis_image, "Osobní údaje", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                break
    
    elif document_type == PASSPORT:
        # Pro pas zkusíme trochu jiný přístup - hledáme centrální oblast
        h, w = gray.shape
        center_x, center_y = w // 2, h // 2
        
        # Oblast uprostřed pasu
        x = center_x - (w // 4)
        y = center_y - (h // 4)
        w = w // 2
        h = h // 2
        
        personal_data_region = gray[y:y+h, x:x+w]
        
        if debug:
            cv2.rectangle(vis_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis_image, "Osobní údaje", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if debug and personal_data_region is not None:
        cv2.imshow("Oblast s osobními údaji", personal_data_region)
        cv2.imshow("Detekce oblastí", vis_image)
    
    return personal_data_region, vis_image

def extract_personal_data(image, personal_data_region, document_type):
    """
    Extrakce osobních údajů z detekované oblasti
    """
    # Předzpracování obrazu pro lepší OCR
    processed = cv2.threshold(personal_data_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Konverze numpy array na PIL Image
    pil_image = Image.fromarray(processed)
    
    # Základní konfigurace pro české znaky
    custom_config = r'--oem 3 --psm 6 -l ces'
    
    # OCR
    text = pytesseract.image_to_string(pil_image, config=custom_config)
    
    # Inicializace slovníku pro osobní údaje
    personal_data = {}
    
    # Extrakce jména a příjmení
    name_match = re.search(r'Jméno(?:/Name)?[:\s]+([^\n]+)', text)
    if name_match:
        personal_data['jmeno'] = name_match.group(1).strip()
    
    surname_match = re.search(r'Příjmení(?:/Surname)?[:\s]+([^\n]+)', text)
    if surname_match:
        personal_data['prijmeni'] = surname_match.group(1).strip()
    
    # Extrakce data narození
    birth_date_match = re.search(r'Datum narození(?:/Date of birth)?[:\s]+([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4}|[0-9]{1,2}\s+[A-Za-zěščřžýáíéůúťďňĚŠČŘŽÝÁÍÉŮÚŤĎŇ]+\s+[0-9]{4})', text)
    if birth_date_match:
        personal_data['datum_narozeni'] = birth_date_match.group(1).strip()
    
    # Extrakce čísla dokladu
    if document_type == ID_CARD:
        id_match = re.search(r'Číslo dokladu(?:/Document No)?[:\s]+([0-9]{9}|[0-9]{8})', text)
        if id_match:
            personal_data['cislo_dokladu'] = id_match.group(1).strip()
    elif document_type == PASSPORT:
        passport_match = re.search(r'Cestovní pas č(?:\.|íslo)(?:/Passport No)?[:\s]+([0-9A-Z]{8})', text)
        if passport_match:
            personal_data['cislo_dokladu'] = passport_match.group(1).strip()
    
    # Extrakce pohlaví
    gender_match = re.search(r'Pohlaví(?:/Sex)?[:\s]+([MFmf])', text)
    if gender_match:
        personal_data['pohlavi'] = gender_match.group(1).upper().strip()
    
    # Extrakce státní příslušnosti
    nationality_match = re.search(r'Státní příslušnost(?:/Nationality)?[:\s]+([^\n]+)', text)
    if nationality_match:
        personal_data['statni_prislusnost'] = nationality_match.group(1).strip()
    
    # Extrakce data platnosti
    expiry_match = re.search(r'Platnost do(?:/Date of expiry)?[:\s]+([0-9]{1,2}\.[0-9]{1,2}\.[0-9]{4})', text)
    if expiry_match:
        personal_data['platnost_do'] = expiry_match.group(1).strip()
    
    return personal_data

def parse_mrz_data(mrz_text, document_type):
    """
    Parsování dat z MRZ textu podle typu dokladu
    """
    data = {}
    
    if not mrz_text:
        return data
    
    # Odstranění nadbytečných znaků
    mrz_text = mrz_text.replace(" ", "").replace("\n", "")
    
    if document_type == ID_CARD:
        # Struktura MRZ pro občanský průkaz (IDCZE...)
        if "IDCZE" in mrz_text:
            # První řádek: IDCZE12345678<<<<<<<
            id_match = re.search(r'IDCZE([A-Z0-9<]{9})', mrz_text)
            if id_match:
                data['cislo_dokladu'] = id_match.group(1).replace("<", "")
            
            # Druhý řádek: obsahuje datum narození a pohlaví
            dob_match = re.search(r'([0-9]{6})([MF])', mrz_text)
            if dob_match:
                birth_date = dob_match.group(1)
                data['datum_narozeni'] = f"{birth_date[4:6]}.{birth_date[2:4]}.{birth_date[0:2]}"
                data['pohlavi'] = dob_match.group(2)
            
            # Třetí řádek: příjmení a jméno
            name_match = re.search(r'([A-Z]+)<<([A-Z]+)', mrz_text)
            if name_match:
                data['prijmeni'] = name_match.group(1).replace("<", " ")
                data['jmeno'] = name_match.group(2).replace("<", " ")
    
    elif document_type == PASSPORT:
        # Struktura MRZ pro pas (P<CZE...)
        if "P<CZE" in mrz_text:
            # Extrakce příjmení a jména
            name_match = re.search(r'P<CZE([A-Z]+)<<([A-Z]+)', mrz_text)
            if name_match:
                data['prijmeni'] = name_match.group(1).replace("<", " ")
                data['jmeno'] = name_match.group(2).replace("<", " ")
            
            # Extrakce čísla pasu
            passport_match = re.search(r'P<CZE[A-Z<]+\n([A-Z0-9<]{9})', mrz_text)
            if passport_match:
                data['cislo_dokladu'] = passport_match.group(1).replace("<", "")
            
            # Extrakce data narození a pohlaví
            dob_match = re.search(r'([0-9]{6})([MF])', mrz_text)
            if dob_match:
                birth_date = dob_match.group(1)
                data['datum_narozeni'] = f"{birth_date[4:6]}.{birth_date[2:4]}.{birth_date[0:2]}"
                data['pohlavi'] = dob_match.group(2)
    
    return data

def combine_data(mrz_data, personal_data):
    """
    Kombinace dat z MRZ a osobních údajů
    """
    combined_data = {}
    
    # Přidání dat z MRZ
    combined_data.update(mrz_data)
    
    # Přepsání nebo doplnění dat z OCR osobních údajů
    for key, value in personal_data.items():
        # Pokud údaj není v MRZ nebo má přednost údaj z osobní části
        if key not in combined_data or key in ['jmeno', 'prijmeni']:
            combined_data[key] = value
    
    return combined_data

def visualize_results(image, data, document_type):
    """
    Vizualizace výsledků OCR na obrázku
    """
    result_image = image.copy()
    
    # Přidání textu s extrahovanými údaji
    y_pos = 30
    for key, value in data.items():
        text = f"{key}: {value}"
        cv2.putText(result_image, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 30
    
    # Přidání typu dokladu
    doc_type_text = "Typ dokladu: Občanský průkaz" if document_type == ID_CARD else "Typ dokladu: Cestovní pas"
    cv2.putText(result_image, doc_type_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return result_image

def main():
    """
    Hlavní funkce programu
    """
    # Načtení argumentů
    args = configure_arguments()
    
    # Načtení obrázku
    image_path = args["image"]
    if not os.path.exists(image_path):
        print(f"Chyba: Soubor {image_path} neexistuje!")
        sys.exit(1)
    
    # Načtení obrázku
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba: Nelze načíst obrázek {image_path}!")
        sys.exit(1)
    
    # Předzpracování obrázku
    processed, gray = preprocess_image(image, args["debug"])
    
    # Detekce MRZ oblasti
    mrz_region, vis_image = detect_mrz_region(image, args["debug"])
    
    # Kontrola, zda byla nalezena MRZ oblast
    if mrz_region is None:
        print("Chyba: Nepodařilo se detekovat MRZ oblast v dokladu!")
        sys.exit(1)
    
    # Předzpracování MRZ pro OCR
    processed_mrz = preprocess_mrz_for_ocr(mrz_region)
    
    # OCR na MRZ
    mrz_text = ocr_mrz(processed_mrz)
    
    # Detekce typu dokladu
    if args["type"] == "auto":
        document_type = detect_document_type(mrz_text)
        if document_type is None:
            print("Varování: Nepodařilo se automaticky detekovat typ dokladu, použijeme výchozí typ: Občanský průkaz")
            document_type = ID_CARD
    else:
        document_type = PASSPORT if args["type"] == "pas" else ID_CARD
    
    print(f"Detekovaný typ dokladu: {document_type}")
    
    # Parsování dat z MRZ
    mrz_data = parse_mrz_data(mrz_text, document_type)
    
    # Detekce oblasti s osobními údaji
    personal_data_region, vis_image = parse_personal_data_region(image, document_type, args["debug"])
    
    # Extrakce osobních údajů
    personal_data = extract_personal_data(image, personal_data_region, document_type) if personal_data_region is not None else {}
    
    # Kombinace dat
    combined_data = combine_data(mrz_data, personal_data)
    
    # Výpis výsledků
    print("\nExtrahované údaje:")
    for key, value in combined_data.items():
        print(f"{key}: {value}")
    
    # Uložení výsledků do JSON souboru
    if args["output"]:
        with open(args["output"], "w", encoding="utf-8") as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
        print(f"\nVýsledky uloženy do souboru: {args['output']}")
    
    # Vizualizace výsledků
    if args["vizualize"]:
        result_image = visualize_results(image, combined_data, document_type)
        cv2.imshow("Výsledky OCR", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()