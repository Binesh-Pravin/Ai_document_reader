"""
Extract fields from Aadhaar / PAN cards (PDF or image) using a LOCAL open-source
vision model via Ollama. Nothing leaves your machine.

Setup (one time):
  1. Install Ollama: https://ollama.com
  2. ollama pull gemma3:12b          # vision-capable; alt: llama3.2-vision:11b
  3. pip install pymupdf requests

Usage:
  python kyc_extract.py card.pdf
  python kyc_extract.py aadhaar_front.jpg aadhaar_back.jpg     # several files = one person
  python kyc_extract.py *.pdf --out results.json
"""
import sys
import re
import json
import base64
import requests
import pymupdf  # PyMuPDF
import os
 
OLLAMA = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_TOKEN = os.environ.get("OLLAMA_PROXY_TOKEN", "")
VISION_MODEL = "gemma3:12b"

# OLLAMA = "http://localhost:11434"
# VISION_MODEL = "gemma3:12b"
DPI = 200

FIELDS = [
    "document_type",   # "aadhaar" or "pan"
    "name",
    "father_name",     # PAN: printed on card. Aadhaar: only if S/O, D/O, W/O, C/O appears
    "dob",
    "gender",
    "aadhaar_number",
    "pan_number",
    "address",
    "pincode",
]

PROMPT = f"""You are reading an Indian identity document (Aadhaar or PAN card).
Extract these fields and return ONLY a JSON object with exactly these keys:
{FIELDS}

Rules:
- Copy values exactly as printed. Do not guess or fix anything.
- Use null for any field that is not visible on this image.
- aadhaar_number: the 12-digit number (digits only, spaces allowed).
- pan_number: the 10-character alphanumeric Permanent Account Number.
- father_name: on a PAN card this is the 'Father's Name' line. On an Aadhaar
  card use the name after S/O, D/O, W/O or C/O in the address, if present.
- address: the full address text on the Aadhaar back side, in one string.
- LANGUAGE: Aadhaar cards print every text twice, in a regional language (Telugu,
  Hindi, Tamil, etc.) and in English. Always return the ENGLISH version of
  name, father_name and address, copied exactly from the English lines on the card.
  Never return regional-language script. Only if no English text exists for a
  field, transliterate it into English letters.
- document_type: "aadhaar" or "pan".
"""


# ---------- Load images ----------
def load_images(path):
    """Return a list of PNG/JPEG bytes: one per PDF page, or the image itself."""
    if path.lower().endswith(".pdf"):
        doc = pymupdf.open(path)
        return [page.get_pixmap(dpi=DPI).tobytes("png") for page in doc]
    with open(path, "rb") as f:
        return [f.read()]


# ---------- Vision model call ----------
DEBUG = False


def _chat(img_bytes, use_json_format):
    body = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [base64.b64encode(img_bytes).decode()],
        }],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    # if use_json_format:
    #     body["format"] = "json"
    # r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    if use_json_format:
        body["format"] = "json"
 
    headers = {}
 
    if OLLAMA_TOKEN:
        headers["Authorization"] = f"Bearer {OLLAMA_TOKEN}"
 
    r = requests.post(
        f"{OLLAMA}/api/chat",
        headers=headers,
        json=body,
        timeout=600
)
    if r.status_code != 200:
        sys.exit(f"Ollama error {r.status_code}: {r.text[:500]}")
    return r.json()["message"]["content"]


def _parse_json(text):
    """Parse JSON, tolerating ```json fences or extra text around it."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def read_image(img_bytes):
    raw = _chat(img_bytes, use_json_format=True)
    if DEBUG:
        print("--- RAW MODEL REPLY (json mode) ---\n" + (raw or "<empty>") + "\n")
    data = _parse_json(raw)
    if not data:  # some models return nothing in JSON mode; retry in plain mode
        raw = _chat(img_bytes, use_json_format=False)
        if DEBUG:
            print("--- RAW MODEL REPLY (plain mode) ---\n" + (raw or "<empty>") + "\n")
        data = _parse_json(raw)
    if not data:
        print("WARNING: model returned no usable JSON for one image. Run with --debug.", file=sys.stderr)
    return data


# ---------- Validation (deterministic, no LLM) ----------
_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],
      [3,4,0,1,2,8,9,5,6,7],[4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],
      [6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],[8,7,6,5,9,3,2,1,0,4],
      [9,8,7,6,5,4,3,2,1,0]]
_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],
      [8,9,1,6,0,4,3,5,2,7],[9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],
      [2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def verhoeff_ok(number):
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _D[c][_P[i % 8][int(ch)]]
    return c == 0


def check_aadhaar(value):
    digits = re.sub(r"\D", "", value or "")
    if len(digits) != 12 or digits[0] in "01":
        return False
    return verhoeff_ok(digits)


def check_pan(value):
    return bool(re.fullmatch(r"[A-Z]{5}[0-9]{4}[A-Z]", (value or "").strip().upper()))


def check_pincode(value):
    return bool(re.fullmatch(r"[1-9][0-9]{5}", (value or "").strip()))


# ---------- Merge pages / files for one person ----------
def merge(results):
    out = {k: None for k in FIELDS}
    for res in results:
        for k in FIELDS:
            v = res.get(k)
            if v in (None, "", "null"):
                continue
            v = str(v).strip()
            if out[k] is None or (k == "address" and len(v) > len(out[k])):
                out[k] = v
    return out


def clean_and_validate(data):
    if data["pan_number"]:
        data["pan_number"] = data["pan_number"].upper().replace(" ", "")
    if data["aadhaar_number"]:
        d = re.sub(r"\D", "", data["aadhaar_number"])
        if len(d) == 12:
            data["aadhaar_number"] = f"{d[:4]} {d[4:8]} {d[8:]}"
    if not data["pincode"] and data["address"]:
        m = re.search(r"\b[1-9][0-9]{5}\b", data["address"])
        data["pincode"] = m.group(0) if m else None

    data["validation"] = {
        "aadhaar_number_valid": check_aadhaar(data["aadhaar_number"]) if data["aadhaar_number"] else None,
        "pan_number_valid": check_pan(data["pan_number"]) if data["pan_number"] else None,
        "pincode_valid": check_pincode(data["pincode"]) if data["pincode"] else None,
    }
    return data


def process(paths):
    results = []
    for p in paths:
        for img in load_images(p):
            results.append(read_image(img))
    return clean_and_validate(merge(results))


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--debug" in args:
        DEBUG = True
        args.remove("--debug")
    out_file = None
    if "--out" in args:
        i = args.index("--out")
        out_file = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args:
        print(__doc__)
        sys.exit(1)

    result = process(args)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if out_file:
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
