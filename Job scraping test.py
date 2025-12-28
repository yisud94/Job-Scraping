import os
import re
import time
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pytesseract

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException, MoveTargetOutOfBoundsException

# --- CONFIG ---
BASE_URL = "https://www.seek.com.au/Nurses-jobs?page={}"
START_PAGE = 1
END_PAGE = 25
OUT_DIR = Path("screenshots")
OUT_DIR.mkdir(exist_ok=True)
OUTPUT_XLSX = "pacu_rn_jobs.xlsx"
HEADLESS = False  

def setup_driver():
    options = Options()
    if HEADLESS:
        options.add_argument("--headless=new")
    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(options=options)  
    driver.set_window_size(1200, 2000)  
    return driver

def scroll_into_view(driver, elem):
    try:
        driver.execute_script("arguments[0].scrollIntoView({behavior:'auto', block:'center', inline:'nearest'});", elem)
        time.sleep(0.4)
    except MoveTargetOutOfBoundsException:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(0.5)
        driver.execute_script("arguments[0].scrollIntoView({behavior:'auto', block:'center'});", elem)
        time.sleep(0.4)

def preprocess_for_ocr(pil_img: Image.Image) -> Image.Image:
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if w < 800:
        scale = 800.0 / w
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((1, 1), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return Image.fromarray(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))

def run_ocr_on_image(pil_img: Image.Image) -> str:
    pre = preprocess_for_ocr(pil_img)
    custom_config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(pre, config=custom_config)
    return text

def simple_parse_job_text(text_block: str) -> dict:
    lines = [ln.strip() for ln in text_block.splitlines() if ln.strip()]
    result = {"title": None, "employer": None, "location": None, "pay": None, "full_time": None, "raw_text": text_block}

    if not lines:
        return result

    result["title"] = lines[0]

    for i in range(1, min(4, len(lines))):
        ln = lines[i]
        if re.search(r'\b(hospital|health|clinic|inc|llc|company|systems|medical|care|university)\b', ln, re.I):
            result["employer"] = ln
            break
    if not result["employer"] and len(lines) > 1:
        result["employer"] = lines[1]

    for ln in lines:
        if re.search(r'\b(remote|work from home)\b', ln, re.I):
            result["location"] = ln
            break
        m = re.search(r'([A-Za-z .]+,\s*[A-Z]{2}\b)', ln)
        if m:
            result["location"] = m.group(1)
            break
    if not result["location"]:
        for ln in lines[::-1]:
            if ',' in ln and len(ln) < 50:
                result["location"] = ln
                break
    for ln in lines:
        if '$' in ln or re.search(r'per hour|/hr|hourly|year|yr|annum|month', ln, re.I):
            result["pay"] = ln
            break

    for ln in lines:
        if re.search(r'\b(full-?time|part-?time|per diem|prn|contract|casual|temporary)\b', ln, re.I):
            result["full_time"] = ln
            break

    return result

def handle_cookies(driver):
    """Handle cookie consent popup if present"""
    try:
        btns = driver.find_elements(By.TAG_NAME, "button")
        for b in btns:
            try:
                txt = b.text.lower()
                if "accept" in txt or "agree" in txt or "cookies" in txt:
                    b.click()
                    time.sleep(0.5)
                    break
            except Exception:
                pass
    except Exception:
        pass

def find_job_elements(driver):
    """Find job listing elements on the page"""
    selectors = [
        "a.tapItem",                 
        "div.job_seen_beacon",      
        "div.jobsearch-SerpJobCard", 
        "div.slider_item"           
    ]

    job_elements = []
    for sel in selectors:
        elems = driver.find_elements(By.CSS_SELECTOR, sel)
        if elems:
            job_elements = elems
            print(f"Found {len(job_elements)} job elements using selector: {sel}")
            break

    if not job_elements:
        print("No job cards found with standard selectors. Trying fallback: all article elements.")
        job_elements = driver.find_elements(By.TAG_NAME, "article")
        print(f"Found {len(job_elements)} article elements for fallback.")
    
    return job_elements

def scrape_page(driver, page_num, job_counter):
    """Scrape a single page and return results"""
    url = BASE_URL.format(page_num)
    print(f"Scraping page {page_num}: {url}")
    
    driver.get(url)
    time.sleep(4)

    handle_cookies(driver)
    
    job_elements = find_job_elements(driver)
    if not job_elements:
        print(f"No job elements found on page {page_num}")
        return [], job_counter

    page_results = []
    for i, elem in enumerate(job_elements):
        try:
            scroll_into_view(driver, elem)
        except StaleElementReferenceException:
            print(f"Stale element at index {i}, skipping.")
            continue

        try:
            driver.execute_script("arguments[0].style.border='3px solid red'", elem)
        except Exception:
            pass
        time.sleep(0.3)

        # Use global counter for unique filenames
        out_path = OUT_DIR / f"job_{job_counter:04d}.png"
        try:
            elem.screenshot(str(out_path))
        except Exception as e:
            # Fallback: take full screenshot and crop
            png = driver.get_screenshot_as_png()
            img = Image.open(BytesIO(png))
            loc = elem.location_once_scrolled_into_view
            size = elem.size
            left = int(loc['x'])
            top = int(loc['y'])
            right = left + int(size['width'])
            bottom = top + int(size['height'])
            crop = img.crop((left, top, right, bottom))
            crop.save(str(out_path))
        
        pil_img = Image.open(str(out_path)).convert("RGB")
        text = run_ocr_on_image(pil_img)
        parsed = simple_parse_job_text(text)
        parsed["screenshot"] = str(out_path)
        parsed["index"] = job_counter
        parsed["page"] = page_num
        page_results.append(parsed)
        
        job_counter += 1
        time.sleep(0.6)

    print(f"Scraped {len(page_results)} jobs from page {page_num}")
    return page_results, job_counter

def main():
    driver = setup_driver()
    all_results = []
    job_counter = 1  # Global counter for unique job IDs
    
    try:
        for page_num in range(START_PAGE, END_PAGE + 1):
            page_results, job_counter = scrape_page(driver, page_num, job_counter)
            all_results.extend(page_results)
            
            # Small delay between pages
            if page_num < END_PAGE:
                time.sleep(2)

        # Save all results to Excel
        if all_results:
            df = pd.DataFrame(all_results)
            cols = ["index", "page", "title", "employer", "location", "pay", "full_time", "screenshot", "raw_text"]
            for c in cols:
                if c not in df.columns:
                    df[c] = None
            df = df[cols]
            df.to_excel(OUTPUT_XLSX, index=False)
            print(f"Done. {len(df)} total rows written to {OUTPUT_XLSX}. Screenshots saved to {OUT_DIR}")
        else:
            print("No results found across all pages.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()