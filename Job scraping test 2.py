from pynput import keyboard
from PIL import ImageGrab
import pytesseract
import pandas as pd
import time
import os

OUTPUT_DIR = "captures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []

def capture_and_ocr():
    timestamp = int(time.time())
    img_path = f"{OUTPUT_DIR}/capture_{timestamp}.png"

    screenshot = ImageGrab.grab()
    screenshot.save(img_path)

    print(f"[✓] Screenshot saved to {img_path}")

    text = pytesseract.image_to_string(screenshot)
    print("[✓] OCR extraction complete.")
    print("-" * 40)
    print(text)
    print("-" * 40)

    results.append({
        "timestamp": timestamp,
        "text": text
    })

def on_press(key):
    try:
        if key.char == 'q':
            print("[✔] Exiting and saving Excel file...")
            df = pd.DataFrame(results)
            df.to_excel("ocr_results.xlsx", index=False)
            print("[✔] Saved to ocr_results.xlsx")
            return False

        if key.char == ' ':
            print("[...] Capturing screenshot...")
            capture_and_ocr()

    except AttributeError:
        pass

print("------------------------------------------------")
print(" Manual Screenshot OCR Tool")
print("------------------------------------------------")
print(" Instructions:")
print("   • Press SPACE → Capture screenshot + OCR")
print("   • Press Q → Quit and save Excel")
print("------------------------------------------------")

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()
