# app/ooxml_lite.py
# -*- coding: utf-8 -*-
"""
OOXML-lite индексатор DOCX:
- Без LibreOffice / PDF-рендера.
- Извлекает главы/подглавы, подписи и содержимое "Рисунок N" (диаграммы и картинки),
  "Таблица N", список литературы/источников.
- Для диаграмм парсит chart XML (+ embedded xlsx при необходимости) и возвращает реальные данные.
  ВАЖНО: если диаграмма отображает проценты (формат numFmt с '%', percentStacked-группировка
  или значения в диапазоне 0..1 с суммой ≈ 1 для круговых/кольцевых), значения будут
  возвращены в строковом виде с символом '%' (например, '47%').

Экспорт:
    build_index(docx_path: str) -> dict
    figure_lookup(index: dict, n: int) -> dict | None
    table_lookup(index: dict, n: int) -> dict | None
"""
from __future__ import annotations

import os
import re
import io
import uuid
import json
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from zipfile import ZipFile
from lxml import etree
from PIL import Image
from openpyxl import load_workbook


# --------- Константы и пространства имён ---------

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}

FIG_CAP_RE = re.compile(r"^(?:Рис\.?|Рисунок|Figure)\s*\.?\s*(\d+)\b.*", re.IGNORECASE)
TAB_CAP_RE = re.compile(r"^(?:Таблица|Table)\s*\.?\s*(\d+)\b.*", re.IGNORECASE)

# Захватываем больше разновидностей заголовков списка источников
HEAD_REF_RE = re.compile(
    r"^(Список\s+(?:использованных\s+)?(?:литературы|источников)|Библиографический\s+список|References)$",
    re.IGNORECASE,
)

# Популярные ID стилей заголовков в DOCX (могут отличаться в кастомных шаблонах)
HEAD_STYLE_RE = re.compile(r"(heading|заголовок)\s*([1-6])", re.IGNORECASE)

RUNTIME_DIR = os.getenv("RUNTIME_DIR", "runtime")
MEDIA_ROOT = os.path.join(RUNTIME_DIR, "media")
INDEX_ROOT = os.path.join(RUNTIME_DIR, "indexes")

MAX_TABLE_ROWS = int(os.getenv("DOC_MAX_TABLE_ROWS", "50"))
MAX_SERIES_POINTS = int(os.getenv("DOC_MAX_SERIES_POINTS", "200"))


# --------- Вспомогательные dataclass-сущности ---------

@dataclass
class Heading:
    level: int
    text: str

@dataclass
class Figure:
    # kind: "chart" | "image"
    kind: str
    caption: Optional[str] = None
    n: Optional[int] = None
    # Для image:
    image_path: Optional[str] = None
    # Для chart:
    chart: Optional[Dict] = None
    # Служебное:
    _order: int = field(default=0)

@dataclass
class Table:
    rows: List[List[str]]
    caption: Optional[str] = None
    n: Optional[int] = None
    _order: int = field(default=0)


# --------- Утилиты чтения XML / DOCX ---------

def _read_xml(z: ZipFile, path: str) -> etree._Element:
    data = z.read(path)
    return etree.fromstring(data)

def _text_of(node: etree._Element) -> str:
    """Собрать видимый текст параграфа или ячейки."""
    parts = node.xpath(".//w:t/text()", namespaces=NS)
    return "".join(parts).replace("\xa0", " ").strip()

def _doc_rels(z: ZipFile, rels_path: str) -> Dict[str, Dict[str, str]]:
    """Вернёт map rId -> {'Target': '...', 'Type':'...'} из файла отношений."""
    rels = {}
    if rels_path in z.namelist():
        root = _read_xml(z, rels_path)
        for rnode in root.findall(".//", namespaces=NS):
            if rnode.tag.endswith("Relationship"):
                rid = rnode.get(f"{{{NS['r']}}}Id") or rnode.get("Id")
                tgt = rnode.get("Target")
                typ = rnode.get("Type")
                if rid and tgt:
                    rels[rid] = {"Target": tgt, "Type": typ or ""}
    return rels

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# --------- Парсинг документа: линейный проход body ---------

def _iter_body_children(doc: etree._Element):
    """Итерируемся по прямым дочерним узлам <w:body>: абзацы и таблицы в исходном порядке."""
    body = doc.find("w:body", namespaces=NS)
    if body is None:
        return
    for child in body:
        if isinstance(child.tag, str) and (child.tag.endswith("p") or child.tag.endswith("tbl")):
            yield child


# --------- Извлечение заголовков ---------

def _extract_heading_level(p: etree._Element) -> Optional[int]:
    """Пытаемся определить уровень из стиля параграфа."""
    val = p.xpath("w:pPr/w:pStyle/@w:val", namespaces=NS)
    if not val:
        return None
    sid = (val[0] or "").strip()
    m = HEAD_STYLE_RE.search(sid)
    if m:
        try:
            return int(m.group(2))
        except Exception:
            return None
    # Иногда в тексте самого абзаца встречается шаблон "Глава 1." — но это уже контент, не стиль.
    return None


# --------- Извлечение таблиц (по XML w:tbl) ---------

def _parse_tbl(tbl: etree._Element, limit_rows: int = MAX_TABLE_ROWS) -> List[List[str]]:
    rows: List[List[str]] = []
    for tr in tbl.findall("w:tr", namespaces=NS):
        row = []
        for tc in tr.findall("w:tc", namespaces=NS):
            # внутри ячейки может быть несколько параграфов
            txt = " ".join(_text_of(p) for p in tc.findall(".//w:p", namespaces=NS) or [])
            row.append(txt.strip())
        if any(cell for cell in row):
            rows.append(row)
        if len(rows) >= limit_rows:
            break
    return rows


# --------- Извлечение рисунков: картинка и диаграмма ---------

def _save_image_from_rel(z: ZipFile, rel_target: str, doc_id: str) -> Optional[str]:
    """
    rel_target приходит из document.xml.rels, обычно 'media/image1.png'.
    Сохраняем в runtime/media/<doc_id>/...
    """
    # Нормализуем путь
    internal = "word/" + rel_target.lstrip("/\\")
    if internal not in z.namelist():
        # иногда target может быть '../media/image1.png' — нормализуем:
        internal = os.path.normpath(os.path.join("word", rel_target))
        internal = internal.replace("\\", "/")
        if internal not in z.namelist():
            return None
    data = z.read(internal)
    ext = os.path.splitext(internal)[1].lower() or ".bin"
    out_dir = os.path.join(MEDIA_ROOT, doc_id)
    _ensure_dir(out_dir)
    fname = f"img_{uuid.uuid4().hex[:8]}{ext}"
    out_path = os.path.join(out_dir, fname)
    with open(out_path, "wb") as f:
        f.write(data)
    # валидация через Pillow (опционально, но полезно)
    try:
        with Image.open(out_path) as im:
            im.verify()
    except Exception:
        pass
    return out_path


# ---- helpers для процентов ----

def _fmt_percent_value(v: float) -> str:
    """Человечный вид: 47% или 12.5% — без лишних нулей."""
    p = v * 100.0
    # если почти целое — отдадим без дробной части
    if abs(p - round(p)) < 0.05:
        return f"{int(round(p))}%"
    s = f"{p:.2f}".rstrip("0").rstrip(".")
    return f"{s}%"

def _has_percent_numfmt(root: etree._Element) -> bool:
    """Есть ли в диаграмме явный числовой формат с % (оси/лейблы/таблица значений)."""
    for nm in root.findall(".//c:numFmt", namespaces=NS):
        fc = (nm.get("formatCode") or "").lower()
        if "%" in fc:
            return True
    return False

def _is_percent_grouping(root: etree._Element) -> bool:
    """Группировка percentStacked (100%-ная составная)."""
    for g in root.findall(".//c:grouping", namespaces=NS):
        val = (g.get("val") or g.get(f"{{{NS['c']}}}val") or "").lower()
        if "percentstacked" in val:
            return True
    return False

def _vals_flat(series: List[Dict]) -> List[float]:
    out: List[float] = []
    for s in series:
        for v in (s.get("vals") or []):
            try:
                out.append(float(v))
            except Exception:
                pass
    return out

def _sum_close_to_one_for_pie(series: List[Dict]) -> bool:
    if not series:
        return False
    vals = [float(v) for v in (series[0].get("vals") or []) if isinstance(v, (int, float)) or str(v).strip()]
    if not vals:
        return False
    s = sum(vals)
    return (0.98 <= s <= 1.02) and max(vals) <= 1.0

def _per_category_sums_close_to_one(series: List[Dict]) -> bool:
    """Эвристика для percentStacked, если формат не указан, но все значения 0..1 и суммы по категориям ≈ 1."""
    if not series:
        return False
    max_len = max(len(s.get("vals") or []) for s in series)
    if max_len == 0:
        return False
    sums: List[float] = []
    for i in range(max_len):
        acc = 0.0
        present = False
        for s in series:
            vals = s.get("vals") or []
            if i < len(vals):
                try:
                    acc += float(vals[i])
                    present = True
                except Exception:
                    pass
        if present:
            sums.append(acc)
    if len(sums) < 2:
        return False
    ok = [0.95 <= x <= 1.05 for x in sums]
    # должно быть хотя бы 2 категории с суммой ≈ 1
    return sum(1 for x in ok if x) >= 2


def _parse_chart_data(z: ZipFile, chart_xml_path: str) -> Dict:
    """
    Извлекаем тип диаграммы, подписи осей, серии (имя, категории, значения).
    Сначала используем кэши (numCache/strCache) в chart.xml,
    затем (если нужно) поднимаем embedded xlsx по chart.rels и читаем диапазоны из формул c:f.
    Дополнительно: детектируем «процентный» контекст и форматируем значения как 'NN%'.
    """
    root = _read_xml(z, chart_xml_path)

    # 1) Тип диаграммы — берем первый найденный
    plot = root.find(".//c:plotArea", namespaces=NS)
    chart_type = None
    chart_node = None
    if plot is not None:
        for tname in ("barChart", "lineChart", "pieChart", "areaChart", "scatterChart",
                      "radarChart", "bar3DChart", "ofPieChart", "doughnutChart", "colChart", "col3DChart"):
            node = plot.find(f"c:{tname}", namespaces=NS)
            if node is not None:
                chart_type = tname.replace("Chart", "").lower()
                chart_node = node
                break

    # 2) Подписи осей (если есть соответствующие узлы)
    def _axis_title(ax_xpath: str) -> Optional[str]:
        title = root.find(ax_xpath + "/c:title", namespaces=NS)
        if title is None:
            return None
        # варианты извлечения текста
        t = title.find(".//c:tx//c:rich//a:t", namespaces=NS)
        if t is not None and t.text:
            return t.text.strip()
        v = title.find(".//c:tx//c:v", namespaces=NS)
        if v is not None and v.text:
            return v.text.strip()
        return None

    x_title = _axis_title(".//c:catAx") or _axis_title(".//c:dateAx")
    y_title = _axis_title(".//c:valAx") or _axis_title(".//c:serAx")

    # 3) Серии: имя, категории, значения
    series: List[Dict] = []

    def _cache_texts(parent: etree._Element, path: str) -> List[str]:
        pts = parent.findall(path, namespaces=NS)
        out = []
        for p in pts[:MAX_SERIES_POINTS]:
            val = (p.text or "").strip()
            out.append(val)
        return out

    def _cache_nums(parent: etree._Element, path: str) -> List[float]:
        pts = parent.findall(path, namespaces=NS)
        out = []
        for p in pts[:MAX_SERIES_POINTS]:
            txt = (p.text or "").strip()
            if not txt:
                continue
            try:
                out.append(float(txt.replace(",", ".")))
            except Exception:
                try:
                    out.append(float(txt))
                except Exception:
                    pass
        return out

    # Сначала пробуем вытащить всё из кешей (numCache/strCache/numLit)
    if chart_node is not None:
        for ser in chart_node.findall("c:ser", namespaces=NS):
            name = ser.find(".//c:tx//c:v", namespaces=NS)
            if name is None:
                name = ser.find(".//c:tx//c:rich//a:t", namespaces=NS)
            series_name = (name.text.strip() if (name is not None and name.text) else None)

            # Категории
            cats = []
            cat_parent = ser.find(".//c:cat", namespaces=NS)
            if cat_parent is not None:
                cats = _cache_texts(cat_parent, ".//c:strCache//c:pt//c:v")
                if not cats:
                    cats = [str(x) for x in _cache_nums(cat_parent, ".//c:numCache//c:pt//c:v")]

            # Значения
            vals = []
            val_parent = ser.find(".//c:val", namespaces=NS)
            if val_parent is not None:
                vals = _cache_nums(val_parent, ".//c:numCache//c:pt//c:v")
                if not vals:
                    vals = _cache_nums(val_parent, ".//c:numLit//c:pt//c:v")

            series.append({"name": series_name, "cats": cats, "vals": vals})

    # 4) Если данных нет или они неполные — попробуем обратиться к embedded xlsx
    needs_xlsx = any((s.get("cats") == [] or s.get("vals") == []) for s in series) or (not series)

    percent_hint_from_xlsx = False

    if needs_xlsx:
        # найдём связи chart -> embeddings/*.xlsx
        base_dir = os.path.dirname(chart_xml_path)
        rels_path = os.path.join(base_dir, "_rels", os.path.basename(chart_xml_path) + ".rels").replace("\\", "/")
        rels = _doc_rels(z, rels_path)
        xlsx_target = None
        for r in rels.values():
            tgt = r.get("Target", "")
            if "embeddings/" in tgt and tgt.lower().endswith(".xlsx"):
                xlsx_target = tgt
                break

        if xlsx_target:
            # нормализуем путь до xlsx
            xlsx_internal = os.path.normpath(os.path.join(base_dir, xlsx_target)).replace("\\", "/")
            if xlsx_internal not in z.namelist():
                xlsx_internal = os.path.normpath(os.path.join("word", xlsx_target)).replace("\\", "/")
            if xlsx_internal in z.namelist():
                wb = load_workbook(filename=io.BytesIO(z.read(xlsx_internal)), data_only=True)

                def read_range_plus(fstr: str) -> Tuple[List, bool]:
                    """
                    Читает диапазон вида 'Лист1'!$B$2:$B$20 -> (список значений, есть_ли_формат_%)
                    """
                    if "!" not in fstr:
                        return [], False
                    sheet, rng = fstr.split("!", 1)
                    sheet = sheet.strip().strip("'").strip('"')
                    ws = wb[sheet]
                    rng = rng.replace("$", "")
                    if ":" in rng:
                        a, b = rng.split(":", 1)
                    else:
                        a = b = rng
                    try:
                        cells = ws[a:b]
                    except Exception:
                        return [], False
                    out: List = []
                    percent_fmt = False
                    if isinstance(cells, tuple):
                        for row in cells:
                            for c in (row if isinstance(row, tuple) else (row,)):
                                out.append(c.value)
                                try:
                                    nf = str(getattr(c, "number_format", "") or "").lower()
                                    if "%" in nf:
                                        percent_fmt = True
                                except Exception:
                                    pass
                    else:
                        out.append(cells.value)
                        try:
                            nf = str(getattr(cells, "number_format", "") or "").lower()
                            if "%" in nf:
                                percent_fmt = True
                        except Exception:
                            pass
                    return out, percent_fmt

                # Перезаполним из ссылок (refs)
                series = []
                if chart_node is not None:
                    for ser in chart_node.findall("c:ser", namespaces=NS):
                        # имя серии
                        nm = ser.find(".//c:tx//c:v", namespaces=NS)
                        if nm is None:
                            nm = ser.find(".//c:tx//c:rich//a:t", namespaces=NS)
                        series_name = (nm.text.strip() if (nm is not None and nm.text) else None)

                        # категории
                        cats = []
                        cat_ref = ser.find(".//c:cat//c:strRef//c:f", namespaces=NS)
                        if cat_ref is None:
                            cat_ref = ser.find(".//c:cat//c:numRef//c:f", namespaces=NS)
                        if cat_ref is not None and cat_ref.text:
                            cat_vals, cat_pct = read_range_plus(cat_ref.text)
                            cats = [str(x) if x is not None else "" for x in cat_vals[:MAX_SERIES_POINTS]]
                            percent_hint_from_xlsx = percent_hint_from_xlsx or cat_pct  # не критично, но пусть будет

                        # значения
                        vals = []
                        val_ref = ser.find(".//c:val//c:numRef//c:f", namespaces=NS)
                        if val_ref is not None and val_ref.text:
                            val_vals, val_pct = read_range_plus(val_ref.text)
                            percent_hint_from_xlsx = percent_hint_from_xlsx or val_pct
                            for v in val_vals[:MAX_SERIES_POINTS]:
                                try:
                                    if v is None:
                                        continue
                                    vals.append(float(v))
                                except Exception:
                                    pass

                        series.append({"name": series_name, "cats": cats, "vals": vals})

    # --- 5) Пост-обработка: детект процентов и форматирование как 'NN%'
    percent_by_fmt = _has_percent_numfmt(root) or _is_percent_grouping(root) or percent_hint_from_xlsx
    percent_by_values = False
    t = (chart_type or "").lower()
    if t in ("pie", "ofpie", "doughnut"):
        percent_by_values = _sum_close_to_one_for_pie(series)
    else:
        # общая эвристика
        vals = _vals_flat(series)
        if vals and max(vals) <= 1.0:
            percent_by_values = _per_category_sums_close_to_one(series) or (sum(vals) > 0 and sum(vals) / len(vals) <= 1.0)

    percent_context = bool(percent_by_fmt or percent_by_values)

    if percent_context:
        # Преобразуем значения: 0.47 -> '47%' (строкой, чтобы дальше в боте печаталось как в диаграмме)
        for s in series:
            new_vals: List[str] = []
            for v in (s.get("vals") or []):
                try:
                    fv = float(v)
                except Exception:
                    # уже строка? оставим как есть
                    new_vals.append(str(v))
                    continue
                # Значения могут быть уже в 0..100 (редко). Если >1.0 и <=100 — считаем, что это проценты как числа.
                if 1.0 < fv <= 100.0:
                    val_percent_str = f"{int(round(fv))}%" if abs(fv - round(fv)) < 0.05 else (f"{fv:.2f}".rstrip("0").rstrip(".") + "%")
                else:
                    val_percent_str = _fmt_percent_value(fv)
                new_vals.append(val_percent_str)
            s["vals"] = new_vals

    return {
        "type": chart_type or "unknown",
        "x_title": x_title,
        "y_title": y_title,
        "percent": percent_context,  # флаг для возможного использования дальше
        "series": series,
    }


# --------- Основной индексатор ---------

def build_index(docx_path: str) -> Dict:
    """
    Строит индекс по документу. Возвращает словарь:
    {
      "meta": {"doc_id": "...", "file": "..."},
      "headings": [{"level":1,"text":"..."}],
      "figures": [{"n":1,"caption":"...","kind":"chart","chart":{...}} | {"n":2,"kind":"image","image_path":"..."}],
      "tables": [{"n":1,"caption":"...","rows":[...]}],
      "references": ["...", "..."]
    }
    """
    doc_id = uuid.uuid4().hex[:8]
    _ensure_dir(MEDIA_ROOT)
    _ensure_dir(INDEX_ROOT)

    with ZipFile(docx_path) as z:
        # 1) Главный документ и отношения
        doc = _read_xml(z, "word/document.xml")
        docrels = _doc_rels(z, "word/_rels/document.xml.rels")

        # 2) Линейный обход body
        headings: List[Heading] = []
        figures: List[Figure] = []
        tables: List[Table] = []

        fig_captions: List[Tuple[int, str]] = []     # (N, caption text) в порядке встречаемости
        tab_captions: List[Tuple[int, str]] = []

        content_seq: List[Tuple[str, int]] = []      # ("figure"/"table"/"p", local_index)

        p_index = 0
        fig_idx = 0
        tbl_idx = 0

        for node in _iter_body_children(doc):
            if node.tag.endswith("p"):
                # Текст параграфа и стиль
                txt = _text_of(node)
                lvl = _extract_heading_level(node)
                if lvl and txt:
                    headings.append(Heading(level=lvl, text=txt))

                # Подписи
                if txt:
                    m_fig = FIG_CAP_RE.match(txt)
                    if m_fig:
                        try:
                            num = int(m_fig.group(1))
                            fig_captions.append((num, txt))
                        except Exception:
                            pass
                    m_tab = TAB_CAP_RE.match(txt)
                    if m_tab:
                        try:
                            num = int(m_tab.group(1))
                            tab_captions.append((num, txt))
                        except Exception:
                            pass

                # Рисунки в параграфе (w:drawing)
                drawings = node.findall(".//w:drawing", namespaces=NS)
                for d in drawings:
                    kind = None
                    image_path = None
                    chart_block = None

                    gdata = d.find(".//a:graphic/a:graphicData", namespaces=NS)
                    if gdata is None:
                        continue
                    uri = gdata.get("uri", "")

                    if uri.endswith("/chart"):
                        # Диаграмма: найти r:id и по нему chart xml
                        cnode = gdata.find(".//c:chart", namespaces=NS)
                        if cnode is not None:
                            rid = cnode.get(f"{{{NS['r']}}}id")
                            if rid and rid in docrels:
                                tgt = docrels[rid]["Target"]
                                # обычно 'charts/chart1.xml'
                                chart_xml = os.path.normpath(os.path.join("word", tgt)).replace("\\", "/")
                                if chart_xml in z.namelist():
                                    chart_block = _parse_chart_data(z, chart_xml)
                                    kind = "chart"

                    else:
                        # Встроенная картинка: ищем pic:blipFill/a:blip@r:embed
                        blip = d.find(".//pic:pic/pic:blipFill/a:blip", namespaces=NS)
                        if blip is None:
                            # могут быть другие варианты представления картинок
                            blip = d.find(".//a:blip", namespaces=NS)
                        rid = blip.get(f"{{{NS['r']}}}embed") if blip is not None else None
                        if rid and rid in docrels:
                            tgt = docrels[rid]["Target"]
                            image_path = _save_image_from_rel(z, tgt, doc_id)
                            if image_path:
                                kind = "image"

                    if kind:
                        figures.append(Figure(kind=kind, image_path=image_path, chart=chart_block, _order=fig_idx))
                        content_seq.append(("figure", fig_idx))
                        fig_idx += 1
                # для порядка
                content_seq.append(("p", p_index))
                p_index += 1

            elif node.tag.endswith("tbl"):
                rows = _parse_tbl(node, limit_rows=MAX_TABLE_ROWS)
                tbl = Table(rows=rows, _order=tbl_idx)
                tables.append(tbl)
                content_seq.append(("table", tbl_idx))
                tbl_idx += 1

        # 3) Сопоставление объектов и подписей по порядку (простой и надёжный способ)
        #    Если количество подписей отличается, сделаем best-effort привязку.
        # Фигуры
        for i, fig in enumerate(figures):
            if i < len(fig_captions):
                n, cap = fig_captions[i]
                fig.n = n
                fig.caption = cap
            else:
                # если подписи не хватило — пронумеруем по порядку
                fig.n = (i + 1) if fig.n is None else fig.n

        # Таблицы
        for i, tbl in enumerate(tables):
            if i < len(tab_captions):
                n, cap = tab_captions[i]
                tbl.n = n
                tbl.caption = cap
            else:
                tbl.n = (i + 1) if tbl.n is None else tbl.n

        # 4) Референсы (Список литературы/источников)
        references: List[str] = []
        # Пройдёмся снова по параграфам в конце — более простой способ:
        paras = doc.findall(".//w:p", namespaces=NS)
        ref_mode = False
        for p in paras:
            txt = _text_of(p)
            if not txt:
                continue
            if not ref_mode and HEAD_REF_RE.match(txt):
                ref_mode = True
                continue
            if ref_mode:
                # следующий заголовок — стоп (по стилю)
                if _extract_heading_level(p):
                    break
                # собираем непустые
                if txt.strip():
                    references.append(txt.strip())

        # 5) Сериализация индекс-структуры
        index = {
            "meta": {"doc_id": doc_id, "file": os.path.abspath(docx_path)},
            "headings": [{"level": h.level, "text": h.text} for h in headings],
            "figures": [],
            "tables": [],
            "references": references,
        }

        # Нормализуем и отсортируем по номеру, если есть
        # Фигуры
        def _fig_sort_key(f: Figure):
            return (f.n if isinstance(f.n, int) else 10**9, f._order)

        for f in sorted(figures, key=_fig_sort_key):
            entry = {
                "n": f.n,
                "caption": f.caption,
                "kind": f.kind,
                "chart": f.chart if f.kind == "chart" else None,
                "image_path": f.image_path if f.kind == "image" else None,
            }
            index["figures"].append(entry)

        # Таблицы
        def _tbl_sort_key(t: Table):
            return (t.n if isinstance(t.n, int) else 10**9, t._order)

        for t in sorted(tables, key=_tbl_sort_key):
            index["tables"].append({
                "n": t.n,
                "caption": t.caption,
                "rows": t.rows,
            })

        # 6) Сохраним индекс на диск (по желанию; пригодится боту)
        _ensure_dir(INDEX_ROOT)
        idx_path = os.path.join(INDEX_ROOT, f"{doc_id}.json")
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)

        return index


# --------- Утилиты поиска ---------

def figure_lookup(index: Dict, n: int) -> Optional[Dict]:
    """Вернёт фигуру с номером n из индекса (или None)."""
    try:
        n = int(n)
    except Exception:
        return None
    for f in index.get("figures", []):
        if f.get("n") == n:
            return f
    return None

def table_lookup(index: Dict, n: int) -> Optional[Dict]:
    """Вернёт таблицу с номером n из индекса (или None)."""
    try:
        n = int(n)
    except Exception:
        return None
    for t in index.get("tables", []):
        if t.get("n") == n:
            return t
    return None


# --------- Опционально: очистка артефактов по doc_id ---------

def purge_media(doc_id: str) -> None:
    """Удалить сохранённые изображения для данного документа (по необходимости)."""
    p = os.path.join(MEDIA_ROOT, doc_id)
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)

def purge_index(doc_id: str) -> None:
    p = os.path.join(INDEX_ROOT, f"{doc_id}.json")
    if os.path.isfile(p):
        try:
            os.remove(p)
        except Exception:
            pass


# --------- Простая CLI-проверка (локально) ---------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m app.ooxml_lite <path-to-docx>")
        sys.exit(1)
    idx = build_index(sys.argv[1])
    print(json.dumps(idx, ensure_ascii=False, indent=2))
