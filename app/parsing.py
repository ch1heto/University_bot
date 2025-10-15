from docx import Document as Docx
import pdfplumber
from pathlib import Path

def parse_docx(path:str) -> list[dict]:
    doc = Docx(path)
    sections, buf = [], []
    title, level = "Документ", 0
    def flush():
        if buf:
            sections.append({"title": title, "level": level, "text": "\n".join(buf), "page": None, "section_path": title})
            buf.clear()

    for p in doc.paragraphs:
        style = (p.style.name or "").lower()
        if style.startswith("heading"):
            flush()
            title = p.text.strip() or "Без названия"
            try:
                level = int(style.replace("heading",""))
            except:
                level = 1
        else:
            buf.append(p.text)
    flush()
    return sections

def parse_pdf(path:str) -> list[dict]:
    out=[]
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            out.append({"title": f"Стр. {i}", "level":1, "text": text, "page": i, "section_path": f"p.{i}"})
    return out

def save_upload(raw: bytes, filename: str, upload_dir="./uploads") -> str:
    p = Path(upload_dir) / filename
    p.write_bytes(raw)
    return str(p)
