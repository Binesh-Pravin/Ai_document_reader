"""
Local web app for the ID Card Reader.

Folder layout:
  id-card-reader/
    app.py            <- this file
    kyc_extract.py    <- reads ID cards
    statement_extract.py <- reads bank statements
    static/index.html <- the front end

Run:
  pip install flask pymupdf requests openpyxl pdfplumber
  python app.py
Then open http://127.0.0.1:5000 on this computer, or the address printed at
startup from other computers on the same network.

Optional password for other computers (PowerShell):
  $env:KYC_PASSWORD = "choose-a-password"; python app.py
"""
import hmac
import io
import os
import re
import socket
import sqlite3
import tempfile
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import kyc_extract as kyc
import statement_extract as stx

app = Flask(__name__, static_folder="static")
app.json.sort_keys = False  # keep the field order defined in SHOW below
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB per request
CORS(app)

# ---------- Optional password (set KYC_PASSWORD before starting) ----------
ACCESS_PASSWORD = os.environ.get("KYC_PASSWORD", "")


@app.before_request
def require_password():
    if not ACCESS_PASSWORD:
        return None
    auth = request.authorization
    if auth and hmac.compare_digest(auth.password or "", ACCESS_PASSWORD):
        return None
    return Response("Password required.", 401,
                    {"WWW-Authenticate": 'Basic realm="ID Card Reader"'})


ALLOWED = {".pdf", ".png", ".jpg", ".jpeg"}

# Which fields each card type shows, in the order they appear on screen.
SHOW = {
    "pan": ["name", "pan_number", "father_name", "dob"],
    "aadhaar": ["name", "aadhaar_number", "gender", "dob", "father_name", "address", "pincode"],
}


def guess_type(result):
    """Use the model's document_type, else infer it from which number it found."""
    t = str(result.get("document_type") or "").lower()
    if "pan" in t:
        return "pan"
    if "aadhaar" in t or "aadhar" in t:
        return "aadhaar"
    if result.get("pan_number"):
        return "pan"
    if result.get("aadhaar_number"):
        return "aadhaar"
    return None


# ---------- Saved records (SQLite table) ----------
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "records.db")
STORE_AADHAAR_MASKED = False  # True = save only "XXXX XXXX 1234" instead of the full number

COLUMNS = ["name", "pan_number", "aadhaar_number", "gender", "dob",
           "father_name", "address", "pincode"]


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS records (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_type       TEXT NOT NULL,
                name           TEXT,
                pan_number     TEXT,
                aadhaar_number TEXT,
                gender         TEXT,
                dob            TEXT,
                father_name    TEXT,
                address        TEXT,
                pincode        TEXT,
                source_files   TEXT,
                saved_at       TEXT NOT NULL
            )""")


def mask_aadhaar(value):
    digits = re.sub(r"\D", "", value or "")
    return f"XXXX XXXX {digits[-4:]}" if len(digits) >= 4 else value


@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/models")
def models():
    """Installed Ollama models, so the page can show a dropdown and a ready badge."""
    try:
        r = requests.get(f"{kyc.OLLAMA}/api/tags", timeout=3)
        r.raise_for_status()
        names = sorted(m["name"] for m in r.json().get("models", []))
        return jsonify({"ready": True, "models": names, "default": kyc.VISION_MODEL})
    except requests.RequestException:
        return jsonify({"ready": False, "models": [], "default": kyc.VISION_MODEL})


@app.post("/api/extract")
def extract():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files received."}), 400

    model = request.form.get("model")
    if model:
        kyc.VISION_MODEL = model

    groups = {"pan": [], "aadhaar": []}   # page results grouped by card type
    sources = {"pan": [], "aadhaar": []}  # file names per card type
    unread = []                            # files where no card type was found

    with tempfile.TemporaryDirectory() as tmp:
        for i, f in enumerate(files):
            ext = os.path.splitext(f.filename)[1].lower()
            if ext not in ALLOWED:
                unread.append(f"{f.filename} (unsupported file type)")
                continue
            path = os.path.join(tmp, f"{i}{ext}")
            f.save(path)

            found = False
            for img in kyc.load_images(path):
                res = kyc.read_image(img)
                t = guess_type(res)
                if t:
                    groups[t].append(res)
                    if f.filename not in sources[t]:
                        sources[t].append(f.filename)
                    found = True
            if not found:
                unread.append(f"{f.filename} (could not recognise a PAN or Aadhaar card)")

    cards = []
    for t, results in groups.items():
        if not results:
            continue
        data = kyc.clean_and_validate(kyc.merge(results))
        cards.append({
            "type": t,
            "files": sources[t],
            "fields": {k: data.get(k) for k in SHOW[t]},
            "validation": data["validation"],
        })
    return jsonify({"cards": cards, "unread": unread})


@app.post("/api/save")
def save():
    """Save one card. If the same PAN / Aadhaar number is already stored, update that row."""
    body = request.get_json(silent=True) or {}
    doc_type = body.get("type")
    if doc_type not in SHOW:
        return jsonify({"error": "Unknown card type."}), 400

    values = {c: (str(body.get("fields", {}).get(c) or "").strip() or None) for c in SHOW[doc_type]}
    if STORE_AADHAAR_MASKED and values.get("aadhaar_number"):
        values["aadhaar_number"] = mask_aadhaar(values["aadhaar_number"])
    files = ", ".join(body.get("files", []))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    key = "pan_number" if doc_type == "pan" else "aadhaar_number"
    with db() as conn:
        existing = None
        if values.get(key):
            existing = conn.execute(
                f"SELECT id FROM records WHERE doc_type = ? AND {key} = ?",
                (doc_type, values[key])).fetchone()
        if existing:
            sets = ", ".join(f"{c} = ?" for c in SHOW[doc_type])
            conn.execute(
                f"UPDATE records SET {sets}, source_files = ?, saved_at = ? WHERE id = ?",
                [values[c] for c in SHOW[doc_type]] + [files, now, existing["id"]])
            return jsonify({"id": existing["id"], "updated": True})
        cols = ["doc_type"] + SHOW[doc_type] + ["source_files", "saved_at"]
        conn.execute(
            f"INSERT INTO records ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [doc_type] + [values[c] for c in SHOW[doc_type]] + [files, now])
        new_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    return jsonify({"id": new_id, "updated": False})


@app.get("/api/records")
def records():
    with db() as conn:
        rows = conn.execute("SELECT * FROM records ORDER BY id DESC").fetchall()
    return jsonify([dict(r) for r in rows])


@app.delete("/api/records/<int:record_id>")
def delete_record(record_id):
    with db() as conn:
        conn.execute("DELETE FROM records WHERE id = ?", (record_id,))
    return jsonify({"deleted": record_id})


# ---------- Excel download ----------
EXCEL_SHEETS = {
    "PAN": ("pan", [("name", "Name"), ("pan_number", "PAN number"),
                    ("father_name", "Father's name"), ("dob", "Date of birth"),
                    ("saved_at", "Saved on")]),
    "Aadhaar": ("aadhaar", [("name", "Name"), ("aadhaar_number", "Aadhaar number"),
                            ("gender", "Gender"), ("dob", "Date of birth"),
                            ("father_name", "Father's name"), ("address", "Address"),
                            ("pincode", "Pincode"), ("saved_at", "Saved on")]),
}


@app.get("/api/export.xlsx")
def export_excel():
    with db() as conn:
        rows = [dict(r) for r in conn.execute("SELECT * FROM records ORDER BY id").fetchall()]

    wb = Workbook()
    wb.remove(wb.active)
    for sheet_name, (doc_type, cols) in EXCEL_SHEETS.items():
        ws = wb.create_sheet(sheet_name)
        ws.append([label for _, label in cols])
        for cell in ws[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="0F766E")
            cell.alignment = Alignment(vertical="center")
        for r in (x for x in rows if x["doc_type"] == doc_type):
            ws.append([r.get(key) or "" for key, _ in cols])
        # Store every value as plain text so Excel never treats it as a formula
        # and keeps leading zeros in numbers.
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.data_type = "s"
                cell.alignment = Alignment(vertical="top", wrap_text=True)
        for i, (key, label) in enumerate(cols, start=1):
            longest = max([len(label)] + [len(str(c.value or "")) for c in ws[get_column_letter(i)][1:]])
            ws.column_dimensions[get_column_letter(i)].width = min(max(longest + 3, 14), 60)
        ws.freeze_panes = "A2"

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    name = f"id_card_records_{datetime.now():%Y%m%d_%H%M}.xlsx"
    return send_file(buf, as_attachment=True, download_name=name,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# ---------- Bank statements ----------
@app.get("/statement")
def statement_page():
    return send_from_directory(app.static_folder, "statement.html")


@app.post("/api/statement")
def statement_read():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Please upload a PDF file."}), 400
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "statement.pdf")
        f.save(path)
        try:
            result = stx.read_statement(
                path,
                password=request.form.get("password") or None,
                model=request.form.get("model") or kyc.VISION_MODEL)
        except stx.NeedPassword:
            return jsonify({"error": "password_required"}), 422
        except stx.NotTextPdf as e:
            return jsonify({"error": str(e)}), 422
        except Exception as e:  # unreadable or damaged PDF
            return jsonify({"error": f"Could not read this PDF: {e}"}), 500
    return jsonify(stx.to_json_safe(result))


@app.post("/api/statement/xlsx")
def statement_xlsx():
    data = request.get_json(silent=True)
    if not data or "transactions" not in data:
        return jsonify({"error": "No statement data received."}), 400
    try:
        blob = stx.to_excel(data)
    except Exception as e:
        return jsonify({"error": f"Could not build the Excel file: {e}"}), 400
    name = f"bank_statement_{datetime.now():%Y%m%d_%H%M}.xlsx"
    return send_file(io.BytesIO(blob), as_attachment=True, download_name=name,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


def local_ip():
    """This computer's address on the local network (what other people type in)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))  # no data is sent
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


if __name__ == "__main__":
    init_db()
    port = 5000
    print(f"\n  On this computer:      http://127.0.0.1:{port}")
    print(f"  From other computers:  http://{local_ip()}:{port}")
    print("  Password protection:  " + ("ON" if ACCESS_PASSWORD else "OFF (set KYC_PASSWORD to turn on)") + "\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    
