# app/vision_analyzer.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import Cfg
from .polza_client import vision_extract_values, vision_describe

try:
    from lxml import etree as ET  # type: ignore
except Exception:  # pragma: no cover
    import xml.etree.ElementTree as ET  # type: ignore

log = logging.getLogger("vision_analyzer")
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(h)
log.setLevel(logging.INFO)

# ----------------------------- cache -----------------------------

_ANALYZER_CACHE_DIR = os.path.join(Cfg.VISION_CACHE_DIR, "analysis")
os.makedirs(_ANALYZER_CACHE_DIR, exist_ok=True)
_ANALYZER_MEM: Dict[str, Dict[str, Any]] = {}


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b or b"")
    return h.hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1024 * 1024), b""):
            h.update(ch)
    return h.hexdigest()


def _cache_key(image_path: str, caption: str | None, context: str | None, chart_xml: Optional[bytes]) -> str:
    parts = [
        "v4",  # версия алгоритма
        _sha256_file(image_path) if os.path.exists(image_path) else image_path,
        _sha256_bytes((caption or "").encode("utf-8")),
        _sha256_bytes((context or "").encode("utf-8")),
        _sha256_bytes(chart_xml or b""),
        f"conf={getattr(Cfg, 'VISION_EXTRACT_CONF_MIN', 0.0)}",
        f"pie_tol={Cfg.VISION_PIE_SUM_TOLERANCE_PP}",
        f"pct_dec={Cfg.VISION_PERCENT_DECIMALS}",
        f"strict={getattr(Cfg, 'FIG_STRICT', True)}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not Cfg.ANALYZER_ENABLE_CACHE:
        return None
    if key in _ANALYZER_MEM:
        return _ANALYZER_MEM[key]
    fp = os.path.join(_ANALYZER_CACHE_DIR, f"{key}.json")
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            obj = json.load(f)
        ttl = int(Cfg.ANALYZER_CACHE_TTL_SEC or (7 * 24 * 3600))
        if time.time() - int(obj.get("_ts", 0)) > ttl:
            return None
        _ANALYZER_MEM[key] = obj.get("payload", obj)
        return _ANALYZER_MEM[key]
    except Exception:
        return None


def _cache_put(key: str, payload: Dict[str, Any]) -> None:
    if not Cfg.ANALYZER_ENABLE_CACHE:
        return
    _ANALYZER_MEM[key] = payload
    fp = os.path.join(_ANALYZER_CACHE_DIR, f"{key}.json")
    try:
        with open(fp, "w", encoding="utf-8") as f:
            json.dump({"_ts": int(time.time()), "payload": payload}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# ----------------------------- utilities -----------------------------

def _clean(s: Optional[str]) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").replace("\xa0", " ")).strip()


def _norm_unit(u: Optional[str]) -> str:
    t = (_clean(u) or "").lower().strip()
    if t in {"%", "percent", "perc", "pct"}:
        return "%"
    if t in {"pcs", "шт", "штук", "ед", "units", "unit"}:
        return "шт"
    if t in {"pp", "п.п.", "пп"}:
        return "п.п."
    if t in {"rub", "₽", "руб", "руб.", "р", "р."}:
        return "₽"
    if t in {"eur", "€", "евро"}:
        return "€"
    if t in {"usd", "$", "доллар", "долл."}:
        return "$"
    if t in {"тыс", "тыс.", "k"}:
        return "тыс."
    if t in {"млн", "млн.", "mln", "m"}:
        return "млн"
    return t


def _to_float(val: Union[str, float, int, None]) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        try:
            return float(val)
        except Exception:
            return None
    s = str(val).strip()
    s = s.replace("−", "-")  # минус из Unicode
    s = s.replace(" ", "")
    s = s.replace(",", ".")
    s = re.sub(r"[%‰]+$", "", s)  # убираем хвостовой %/‰, если попался в value
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _fmt_percent(x: float, decimals: int) -> str:
    # Русский стиль: запятая, пробел перед %
    if x is None:
        return "—"
    s = f"{x:.{max(0, decimals)}f}"
    s = s.replace(".", ",")
    return f"{s} %"


def _fmt_number(x: float) -> str:
    if x is None:
        return "—"
    n = float(x)
    s = f"{int(round(n)):,}"
    # заменяем запятые на пробелы в качестве разрядного разделителя
    return s.replace(",", " ")


def _delta_fmt(a: float, b: float, unit: str) -> str:
    d = a - b
    if unit == "%":
        return f"{_fmt_percent(abs(d), Cfg.VISION_PERCENT_DECIMALS)}".replace(" %", " п.п.")
    if unit:
        return f"{_fmt_number(abs(d))} {unit}"
    return f"{_fmt_number(abs(d))}"


def _largest_remainder_to_100(vals: List[float]) -> List[int]:
    """
    Округление к целым с сохранением суммы = 100 (метод наибольших остатков).
    Предполагается, что vals ~ проценты и sum(vals) ~ 100.
    """
    base = [max(0.0, float(v)) for v in vals]
    floors = [int(math.floor(x)) for x in base]
    need = int(round(100 - sum(floors)))
    rema = [x - f for x, f in zip(base, floors)]
    # Положительные остатки — распределяем в порядке убывания
    order_desc = sorted(range(len(base)), key=lambda i: rema[i], reverse=True)
    # Отрицательная коррекция теоретически не должна понадобиться (после floor),
    # но на всякий случай уменьшим у наименьших остатков
    order_asc = list(reversed(order_desc))
    while need != 0:
        if need > 0:
            j = order_desc[(100 - need) % len(order_desc)]
            floors[j] += 1
            need -= 1
        else:
            j = order_asc[(-100 - need) % len(order_asc)]
            if floors[j] > 0:
                floors[j] -= 1
                need += 1
            else:
                break
    return floors


def _maybe_normalize_pie(values: List[Tuple[str, float, str]]) -> Tuple[List[Tuple[str, float, str]], float, bool, List[int] | None]:
    """
    Если это Pie и сумма 95..105% — пропорционально домножим до 100 и округлим до целых процента.
    Возвращает (values, sum_before, normalized, ints_or_none)
    """
    if not values:
        return values, 0.0, False, None
    if not all((_norm_unit(u) in {"%", ""}) for (_, _, u) in values):
        return values, 0.0, False, None
    s = sum(v for (_, v, _) in values if v is not None)
    if s <= 0:
        return values, s, False, None
    target = float(Cfg.VISION_PIE_SUM_TARGET or 100.0)
    tol = float(Cfg.VISION_PIE_SUM_TOLERANCE_PP or 5.0)
    if abs(s - target) <= tol:
        # нормализуем
        k = target / s
        normed = [(lbl, v * k, "%") for (lbl, v, _) in values]
        ints = _largest_remainder_to_100([v for (_, v, _) in normed])
        normed_int = [(lbl, float(iv), "%") for (iv, (lbl, _, _)) in zip(ints, normed)]
        return normed_int, s, True, ints
    return values, s, False, None


# ----------------------------- chart XML parser (DOCX) -----------------------------

NS = {
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}


def _cache_points(root, path: str) -> List[Tuple[int, str]]:
    pts: List[Tuple[int, str]] = []
    for pt in root.findall(path + "/c:pt", NS):
        try:
            idx = int(pt.attrib.get("idx", "0"))
        except Exception:
            continue
        val_el = pt.find("c:v", NS)
        val = (val_el.text if val_el is not None else "") or ""
        pts.append((idx, val))
    pts.sort(key=lambda x: x[0])
    return pts


def _series_name(ser) -> Optional[str]:
    try:
        v = ser.find(".//c:tx/c:v", NS)
        if v is not None and (v.text or "").strip():
            return v.text.strip()
        v = ser.find(".//c:tx/c:strRef/c:strCache/c:pt/c:v", NS)
        if v is not None and (v.text or "").strip():
            return v.text.strip()
    except Exception:
        pass
    return None


def _parse_chart_xml(xml_bytes: bytes) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Возвращает (chart_type, rows[{label, value}])
    Поддержка Pie/Bar/Column/Line. Если не узнали — ("Unknown", []).
    """
    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return "Unknown", []

    def _first(*paths):
        for p in paths:
            el = root.find(".//" + p, NS)
            if el is not None:
                return el
        return None

    chart_el = _first("c:pieChart", "c:barChart", "c:bar3DChart", "c:colChart", "c:col3DChart", "c:lineChart")
    if chart_el is None:
        return "Unknown", []

    tag = chart_el.tag.split("}")[-1]
    ctype = {
        "pieChart": "Pie",
        "barChart": "Bar",
        "bar3DChart": "Bar",
        "colChart": "Bar",
        "col3DChart": "Bar",
        "lineChart": "Line",
    }.get(tag, "Unknown")

    out: List[Dict[str, Any]] = []
    series = chart_el.findall("./c:ser", NS) or []
    multi = len(series) > 1

    for ser in series:
        sname = _series_name(ser)
        cat_pts = _cache_points(ser, ".//c:cat/c:strRef/c:strCache") or _cache_points(ser, ".//c:cat/c:numRef/c:numCache")
        val_pts = _cache_points(ser, ".//c:val/c:numRef/c:numCache") or _cache_points(ser, ".//c:val/c:numCache")
        by_idx = {i: {"label": l, "value": None} for i, l in cat_pts}
        for i, v in val_pts:
            try:
                by_idx.setdefault(i, {})["value"] = float(str(v).replace(",", "."))
            except Exception:
                by_idx.setdefault(i, {})["value"] = None
        for i in sorted(by_idx):
            row = by_idx[i]
            label = str(row.get("label") or "").strip()
            value = row.get("value")
            if multi and sname:
                label = f"{sname}: {label}" if label else sname
            if label or (value is not None):
                out.append({"label": label, "value": value})
    return ctype, out


# ----------------------------- reconciliation -----------------------------

@dataclass
class Item:
    label: str
    value: float
    unit: str  # "%", "шт", "₽", "" ...


def _choose_unit(candidates: List[str]) -> str:
    # простое большинство, при равенстве — "%", иначе первая непустая
    norm = [_norm_unit(u) for u in candidates if _norm_unit(u)]
    if not norm:
        return ""
    best: Dict[str, int] = {}
    for u in norm:
        best[u] = best.get(u, 0) + 1
    winner = max(best.items(), key=lambda kv: kv[1])[0]
    return winner


_BAD_LABEL_RE = re.compile(r"^[\W_]+$", re.UNICODE)


def _is_bad_label(lbl: str) -> bool:
    t = (lbl or "").strip()
    if len(t) < 2:
        return True
    if _BAD_LABEL_RE.match(t):
        return True
    if re.fullmatch(r"\d+([.,]\d+)?", t):  # «метка» — чистое число
        return True
    return False


def _from_extract(obj: Dict[str, Any]) -> Tuple[str, List[Item], int]:
    """
    Превращает ответ vision_extract_values в список Item'ов.
    Возвращает (chart_type, items, ignored_count)
    """
    ctype = str(obj.get("type") or "Unknown").strip().title()
    rows = obj.get("data") or []
    items: List[Item] = []
    ignored = 0

    # выбор единицы измерения
    unit_candidates: List[str] = []
    for r in rows:
        unit_candidates.append(_norm_unit(r.get("unit")))
    default_unit = _choose_unit(unit_candidates)

    conf_min = float(getattr(Cfg, "VISION_EXTRACT_CONF_MIN", 0.0) or 0.0)

    for r in rows:
        lbl = _clean(r.get("label"))
        if _is_bad_label(lbl):
            ignored += 1
            continue
        unit = _norm_unit(r.get("unit")) or default_unit
        val = _to_float(r.get("value"))
        conf = r.get("conf") or r.get("confidence") or r.get("score")
        try:
            conf = float(conf) if conf is not None else 1.0
        except Exception:
            conf = 1.0
        if conf < conf_min or val is None:
            ignored += 1
            continue
        # если unit = "%" и значения в 0..1 — домножим
        if unit == "%" and 0.0 <= val <= 1.0:
            val *= 100.0
        items.append(Item(lbl, float(val), unit))
    return ctype or "Unknown", items, ignored


def _from_xml(xml_bytes: bytes) -> Tuple[str, List[Item]]:
    ctype, rows = _parse_chart_xml(xml_bytes)
    items: List[Item] = []
    for r in rows or []:
        lbl = _clean(r.get("label"))
        if _is_bad_label(lbl):
            continue
        val = _to_float(r.get("value"))
        if val is None:
            continue
        # DOCX chart XML не содержит unit → определим по контексту дальше
        items.append(Item(lbl, float(val), ""))  # unit пустой; для Pie назначим '%' ниже
    return (ctype or "Unknown"), items


def _guess_is_pie(items: List[Item]) -> bool:
    if not items:
        return False
    # если большинство единиц — % или сумма близка к 100
    units = [it.unit for it in items if it.unit]
    if units and all(_norm_unit(u) == "%" for u in units):
        return True
    s = sum(it.value for it in items)
    return 85.0 <= s <= 115.0  # грубая эвристика


def _reconcile(
    xml_bytes: Optional[bytes],
    extract_obj: Optional[Dict[str, Any]],
) -> Tuple[str, List[Item], Dict[str, Any], bool]:
    """
    Возвращает (chart_type, items, extra_meta, is_from_xml)
    """
    # 1) XML — истина, если есть
    if xml_bytes:
        ctype_xml, items_xml = _from_xml(xml_bytes)
        # если это круговая (по величинам), подставим unit='%'
        if _guess_is_pie(items_xml):
            items_xml = [Item(it.label, it.value, "%") for it in items_xml]
        meta = {"source": "ooxml", "ignored_items": 0}
        return ctype_xml, items_xml, meta, True

    # 2) Vision Extract
    if extract_obj:
        ctype, items, ignored = _from_extract(extract_obj)
        meta = {"source": "image_text", "ignored_items": ignored}
        return ctype, items, meta, False

    return "Unknown", [], {"source": "none", "ignored_items": 0}, False


# ----------------------------- chart hint helpers -----------------------------

def _chart_hint_from_text(*texts: str) -> Optional[str]:
    """
    Извлекаем хинт типа диаграммы из caption/context (ru+en).
    Возвращает: 'pie'|'stacked_bar'|'bar'|'line'|'histogram'|None
    Приоритет: pie > stacked_bar > bar > line > histogram
    """
    t = " ".join([_clean(x).lower() for x in texts if x]).strip()
    if not t:
        return None
    if any(k in t for k in ("кругов", "кольцев", "pie", "donut", "ring")):
        return "pie"
    if any(k in t for k in ("stacked", "накоп", "100%", "100 %", "составн")):
        return "stacked_bar"
    if any(k in t for k in ("столбч", "bar", "column", "колонн", "bar chart", "column chart")):
        return "bar"
    if any(k in t for k in ("линейн", "line", "trend", "time series")):
        return "line"
    if any(k in t for k in ("гистограм", "histogram", "freq", "распределен")):
        return "histogram"
    return None


def classify_figure(
    image_path: str,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    chart_xml: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Лёгкая классификация типа рисунка БЕЗ вызова vision-модели.

    Использует:
    - наличие chart_xml;
    - текст подписи/контекста;
    - хинт типа диаграммы (_chart_hint_from_text).

    Возвращает dict:
        {
          "kind": "pie" | "bar" | "line" | "stacked_bar" |
                   "org_chart" | "flow_diagram" | "text_blocks" |
                   "photo" | "other",
          "numeric": bool,      # ожидать ли точные числа
          "structural": bool,   # структурная/текстовая схема
          "chart_hint": str|None,
          "reason": str,        # для дебага
        }
    """
    caption_clean = _clean(caption)
    context_clean = _clean(context)
    joined_text = (caption_clean + " " + context_clean).strip().lower()
    chart_hint = _chart_hint_from_text(caption_clean, context_clean)

    kind = "other"
    reason = "default"
    numeric = False

    # 1) Если есть chart_xml — это точно диаграмма
    if chart_xml:
        ctype, _rows = _parse_chart_xml(chart_xml)
        c = (ctype or "Unknown").lower()
        if c.startswith("pie"):
            kind = "pie"
        elif c.startswith("line"):
            kind = "line"
        elif c.startswith("bar"):
            kind = "bar"
        else:
            kind = "bar"
        reason = "chart_xml"
        numeric = True

    # 2) Иначе — подсказка из текста
    elif chart_hint in {"pie", "stacked_bar", "bar", "line", "histogram"}:
        if chart_hint == "stacked_bar":
            kind = "stacked_bar"
        elif chart_hint == "pie":
            kind = "pie"
        elif chart_hint == "line":
            kind = "line"
        else:
            kind = "bar"
        reason = "caption_chart_hint"
        numeric = True

    # 3) Текстовые/структурные рисунки
    else:
        t = joined_text
        if any(word in t for word in ("организацион", "оргструктур", "структура управления", "org chart")):
            kind = "org_chart"
            reason = "org_chart_keywords"
        elif any(word in t for word in ("схема", "diagram", "блок-схем", "flow", "mind-map", "карта проблем")):
            kind = "flow_diagram"
            reason = "flow_diagram_keywords"
        elif any(word in t for word in ("проблем", "фактор", "этап", "элементы", "пункты", "характеристик", "преимуществ", "недостатк")):
            kind = "text_blocks"
            reason = "text_blocks_keywords"
        else:
            kind = "photo"
            reason = "fallback_photo"

    # Применяем знания из конфигурации
    numeric = bool(numeric and Cfg.is_numeric_figure_kind(kind))
    structural = bool(Cfg.is_textual_figure_kind(kind))

    return {
        "kind": kind,
        "numeric": numeric,
        "structural": structural,
        "chart_hint": chart_hint,
        "reason": reason,
    }


# ----------------------------- NLG -----------------------------

def _pick_top(items: List[Item]) -> Tuple[Optional[Item], Optional[Item], Optional[Item]]:
    if not items:
        return None, None, None
    arr = sorted(items, key=lambda it: it.value, reverse=True)
    top = arr[0]
    runner = arr[1] if len(arr) > 1 else None
    tail = arr[-1] if len(arr) > 1 else None
    return top, runner, tail


def _fmt_item(it: Item) -> str:
    if it.unit == "%":
        return f"{_fmt_percent(it.value, Cfg.VISION_PERCENT_DECIMALS)}"
    if it.unit:
        return f"{_fmt_number(it.value)} {it.unit}"
    return _fmt_number(it.value)


def _nlg_pie(items: List[Item], caption: Optional[str], ocr_caveat: bool, percent_sum: Optional[float], normalized: bool) -> str:
    top, runner, _min = _pick_top(items)
    parts: List[str] = []
    cap = _clean(caption)
    if top and runner:
        parts.append(
            f"{('' if not cap else cap + ': ')}лидирует {top.label} — { _fmt_item(top) }, "
            f"на { _delta_fmt(top.value, runner.value, '%') } больше, чем {runner.label} ({ _fmt_item(runner) })."
        )
    elif top:
        parts.append(f"{('' if not cap else cap + ': ')}наибольшая доля у {top.label} — { _fmt_item(top) }.")
    if percent_sum is not None:
        s = f"Сумма долей ≈ { _fmt_percent(percent_sum, Cfg.VISION_PERCENT_DECIMALS) }".replace(" %", " %")
        if normalized:
            s += "; нормализовано до 100 %."
        parts.append(s)
    if ocr_caveat and Cfg.VISION_APPEND_CAVEAT_FOR_OCR:
        parts.append("Числа считаны с изображения; возможна небольшая погрешность.")
    return " ".join(parts).strip()


def _nlg_bar(items: List[Item], caption: Optional[str], ocr_caveat: bool) -> str:
    top, runner, _min = _pick_top(items)
    parts: List[str] = []
    cap = _clean(caption)
    unit = items[0].unit if items and items[0].unit else ""

    if top and runner:
        delta = _delta_fmt(top.value, runner.value, unit or top.unit)
        parts.append(
            f"{('' if not cap else cap + ': ')}лидирует {top.label} — { _fmt_item(top) }, "
            f"что на {delta} выше, чем у {runner.label} ({ _fmt_item(runner) })."
        )
    elif top:
        parts.append(f"{('' if not cap else cap + ': ')}максимум у {top.label} — { _fmt_item(top) }.")

    if _min:
        parts.append(f"Минимум у { _min.label } — { _fmt_item(_min) }.")

    if ocr_caveat and Cfg.VISION_APPEND_CAVEAT_FOR_OCR:
        parts.append("Значения считаны с изображения; возможна небольшая погрешность.")
    return " ".join(parts).strip()


def _nlg_line(items: List[Item], caption: Optional[str], ocr_caveat: bool) -> str:
    # Для line предполагаем, что labels — это «время/категории по порядку».
    # Берём первый/последний + пик/минимум.
    parts: List[str] = []
    cap = _clean(caption)
    if not items:
        return cap or "Данные не распознаны."

    # Попробуем извлечь временной порядок как есть
    first = items[0]
    last = items[-1]
    unit = last.unit or first.unit

    trend = "вырос" if last.value >= first.value else "снизился"
    delta = _delta_fmt(last.value, first.value, unit)

    if cap:
        parts.append(f"{cap}: показатель {trend} с { _fmt_item(first) } до { _fmt_item(last) } (на {delta}).")
    else:
        parts.append(f"Показатель {trend} с { _fmt_item(first) } до { _fmt_item(last) } (на {delta}).")

    peak = max(items, key=lambda it: it.value)
    trough = min(items, key=lambda it: it.value)

    if peak and trough and (peak.label or trough.label):
        parts.append(f"Пик — { _fmt_item(peak) } в {peak.label}; минимум — { _fmt_item(trough) }.")

    if ocr_caveat and Cfg.VISION_APPEND_CAVEAT_FOR_OCR:
        parts.append("Значения считаны с изображения; возможна небольшая погрешность.")
    return " ".join(parts).strip()


def _nlg_generic(items: List[Item], caption: Optional[str], ocr_caveat: bool) -> str:
    # Универсальный fallback
    if not items:
        return (caption + ": " if caption else "") + "данные не распознаны."
    top, runner, _ = _pick_top(items)
    unit = items[0].unit if items[0].unit else ""
    if top and runner:
        delta = _delta_fmt(top.value, runner.value, unit or top.unit)
        base = f"{('' if not caption else caption + ': ')}больше всего у {top.label} — { _fmt_item(top) }; далее {runner.label} — { _fmt_item(runner) } (разница {delta})."
    else:
        base = f"{('' if not caption else caption + ': ')}главное значение — { _fmt_item(items[0]) } ({items[0].label})."
    caveat = " Числа считаны с изображения; возможна небольшая погрешность." if (ocr_caveat and Cfg.VISION_APPEND_CAVEAT_FOR_OCR) else ""
    return (base + caveat).strip()


# ----------------------------- strict validation -----------------------------

def _strict_validate(chart_type: str, items: List[Item]) -> Tuple[bool, List[Item], Optional[str], Optional[float]]:
    """
    Возвращает (ok, cleaned_items, error_msg, percent_sum_scaled_or_None)
    В строгом режиме:
      - Для Pie/процентов: принимаем только данные, которые уже выглядят как проценты
        (сумма в разумном допуске к целевому значению). Если сумма «поехала» — считаем,
        что OCR ошибся, и отказываемся от чисел.
      - Для прочих: проверка единиц (все одинаковые, если заданы), NaN/Inf/пустые.
      - Требуем минимум 2 надёжные пары «метка→значение».
    """
    if not getattr(Cfg, "FIG_STRICT", True):
        return True, items, None, None

    ctype = (chart_type or "Unknown").lower()
    clean_items = [
        it for it in items
        if (not _is_bad_label(it.label))
        and (it.value is not None)
        and (not (math.isnan(it.value) or math.isinf(it.value)))
    ]

    if len(clean_items) < 2:
        return False, [], "Строгий режим: недостаточно надёжных пар «метка→значение».", None

    # Круговые/процентные диаграммы
    if ctype.startswith("pie") or all(it.unit in {"%", ""} for it in clean_items):
        vals = [max(0.0, float(it.value)) for it in clean_items]
        if not vals or sum(vals) <= 0:
            return False, [], "Строгий режим: нулевые значения.", None

        s = sum(vals)
        target = float(getattr(Cfg, "VISION_PIE_SUM_TARGET", 100.0) or 100.0)
        tol = float(getattr(Cfg, "VISION_PIE_SUM_TOLERANCE_PP", 5.0) or 5.0)

        # Если сумма вовсе не похожа на проценты — считаем, что OCR ошибся
        if not (target - tol <= s <= target + tol):
            return False, [], "Строгий режим: сумма долей диаграммы не похожа на 100 %.", s

        # Нормируем только когда исходные данные уже около 100
        k = target / s if s != 0 else 1.0
        scaled = [v * k for v in vals]
        ints = _largest_remainder_to_100(scaled)

        # ints по определению должны давать ровно target (обычно 100), но перепроверим
        if sum(ints) != int(round(target)):
            return False, [], "Строгий режим: не удалось согласовать доли до 100 %.", sum(scaled)

        out = [
            Item(clean_items[i].label, float(ints[i]), "%")
            for i in range(len(clean_items))
        ]
        # в strict_sum вернём исходную сумму (до нормализации), чтобы meta могло её показать
        return True, out, None, s

    # Прочие типы графиков — проверяем единицы измерения
    units = {it.unit for it in clean_items if it.unit != ""}
    if len(units) > 1:
        return False, [], "Строгий режим: несогласованные единицы измерения.", None

    return True, clean_items, None, None


# ----------------------------- public API -----------------------------

def analyze_chart(
    image_path: str,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    intent: Optional[str] = "describe-with-numbers",
    chart_xml: Optional[bytes] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Анализ диаграммы (круговая/столбчатая/линейная) с числами.

    Логика:
      1) кэш по хэшу изображения+контексту;
      2) если есть chart_xml → берём его данные как «истину»;
      3) иначе vision_extract_values();
      4) сведение/валидация (строгий режим exact-or-fail при FIG_STRICT=True);
      5) генерация краткого текстового описания по типу графика.

    Возвращаем унифицированный payload:
      {
        "text": str,              # краткое описание графика
        "data": [...],            # список {label, value, unit}
        "meta": {...},            # служебная информация
        "kind": "chart",
        "exact_numbers": [...],   # тот же список data (если прошли strict)
        "exact_text": False,      # описание всегда NLG, не OCR
        "warnings": [...],        # человекочитаемые предупреждения
        "raw_text": [...],        # поднятые подписи/контекст/фрагменты
      }
    """
    # совместимость с вызовами вида analyze_figure(..., caption_hint=..., lang="ru")
    cap_hint = kwargs.get("caption_hint")
    if cap_hint and not caption:
        caption = cap_hint
    # lang и прочие дополнительные kwargs на этой стадии не используются

    # Ключ кэша (для диаграмм)
    key = "chart|" + _cache_key(image_path, caption, context, chart_xml)
    cached = _cache_get(key)
    if cached:
        return cached

    # Хинт типа диаграммы (из текста подписи/контекста)
    chart_hint = _chart_hint_from_text(caption or "", context or "")

    # 1) Получаем данные
    extract_obj: Optional[Dict[str, Any]] = None
    ctype: str = "Unknown"
    items: List[Item] = []
    extra_meta: Dict[str, Any] = {"chart_hint": chart_hint}
    used_xml = chart_xml is not None and len(chart_xml) > 0

    # текстовые фрагменты для raw_text
    raw_fragments: List[str] = []
    if caption:
        raw_fragments.append(_clean(caption))
    if context:
        raw_fragments.append(_clean(context))

    if used_xml:
        ctype, items, extra_meta_xml, _ = _reconcile(chart_xml, None)
        extra_meta.update(extra_meta_xml or {})
    else:
        if Cfg.VISION_EXTRACT_VALUES_ENABLED and Cfg.vision_active():
            try:
                langs = getattr(Cfg, "VISION_LANGS", None)
                lang = getattr(Cfg, "VISION_LANG", None)
                lang_to_use = ",".join(langs) if (isinstance(langs, list) and langs) else (lang or "ru,en")

                # Встраиваем хинт внутрь caption_hint/ocr_hint, чтобы модель учитывала тип
                cap_hint_loc = _clean(caption)
                ocr_hint = _clean(context)
                if chart_hint:
                    tag = f"[chart_hint={chart_hint}]"
                    cap_hint_loc = f"{cap_hint_loc} {tag}".strip() if cap_hint_loc else tag
                    ocr_hint = f"{ocr_hint} {tag}".strip() if ocr_hint else tag

                extract_obj = vision_extract_values(
                    image_or_images=image_path,
                    caption_hint=cap_hint_loc,
                    ocr_hint=ocr_hint,
                    temperature=0.0,
                    max_tokens=1400,
                    lang=lang_to_use,
                )
            except Exception as e:
                log.warning("vision_extract_values failed: %s", e)
                extract_obj = None
        ctype, items, extra_meta_v, _ = _reconcile(None, extract_obj)
        extra_meta.update(extra_meta_v or {})

        # вытащим видимые тексты из ответа vision_extract_values (если есть)
        if isinstance(extract_obj, dict):
            rt = extract_obj.get("raw_text") or extract_obj.get("texts") or []
            if isinstance(rt, str):
                raw_fragments.append(_clean(rt))
            elif isinstance(rt, list):
                for t in rt:
                    if t:
                        raw_fragments.append(_clean(str(t)))

    # 2) Нормализация для Pie / процентов
    percent_sum = None
    normalized = False
    ints_used: List[int] | None = None
    if (ctype or "").lower().startswith("pie") or (items and all(it.unit in {"%", ""} for it in items)):
        # если в items unit пустой — назначим '%' (vision может не прислать)
        items = [Item(it.label, it.value, it.unit or "%") for it in items]
        items_norm, s, normalized_norm, ints = _maybe_normalize_pie([(it.label, it.value, it.unit) for it in items])
        percent_sum = round(s, Cfg.VISION_PERCENT_DECIMALS) if s else None
        if normalized_norm and ints:
            items = [Item(lbl, float(v), unit) for (lbl, v, unit) in items_norm]
            ints_used = ints
            normalized = True

    # 3) Строгая валидация (exact-or-fail) — если источник не OOXML
    ok, strict_items, strict_msg, strict_sum = _strict_validate(ctype, items)
    if getattr(Cfg, "FIG_STRICT", True) and not used_xml:
        if not ok:
            # Падение в описание без чисел
            try:
                desc = vision_describe(image_or_images=image_path, lang=getattr(Cfg, "VISION_LANG", "ru"))
                text = (desc.get("description") or "").strip()
                if caption:
                    text = f"{caption}: {text}" if text else caption
            except Exception:
                text = caption or "Описание изображения недоступно."

            warnings: List[str] = []
            if strict_msg:
                warnings.append(strict_msg)
            ignored_items = int(extra_meta.get("ignored_items") or 0)
            if ignored_items:
                warnings.append(
                    f"Игнорировано {ignored_items} подозрительных точек (низкая уверенность или некорректные метки)."
                )

            payload = {
                "kind": "chart",
                "text": text,
                "data": [],
                "exact_numbers": None,
                "exact_text": False,
                "warnings": warnings,
                "raw_text": [t for t in raw_fragments if t],
                "meta": {
                    "chart_type": (ctype or "Unknown").strip().title(),
                    "source": extra_meta.get("source") or "image_text",
                    "used_items": [],
                    "ignored_items": ignored_items,
                    "percent_sum": None,
                    "normalized": False,
                    "caveat": strict_msg or "Строгий режим: числа не подтверждены текстом изображения.",
                    "caption": caption,
                    "context": context,
                    "strict_passed": False,
                    "chart_hint": chart_hint,
                },
            }
            _cache_put(key, payload)
            return payload
        else:
            items = strict_items
            if strict_sum is not None:
                percent_sum = round(strict_sum, Cfg.VISION_PERCENT_DECIMALS)
            normalized = True if (ints_used or (items and all(it.unit == "%" for it in items))) else normalized

    # 4) Если после сводки нет данных — попробуем хотя бы «описание без чисел»
    if not items:
        try:
            desc = vision_describe(image_or_images=image_path, lang=getattr(Cfg, "VISION_LANG", "ru"))
            text = (desc.get("description") or "").strip()
            if caption:
                text = f"{caption}: {text}" if text else caption
        except Exception:
            text = caption or "Описание изображения недоступно."

        ignored_items = int(extra_meta.get("ignored_items") or 0)
        warnings: List[str] = []
        if ignored_items:
            warnings.append(
                f"Игнорировано {ignored_items} подозрительных точек (низкая уверенность или некорректные метки)."
            )
        if extra_meta.get("source") == "image_text" and Cfg.VISION_APPEND_CAVEAT_FOR_OCR:
            warnings.append("Числа считаны с изображения; возможна погрешность.")

        payload = {
            "kind": "chart",
            "text": text,
            "data": [],
            "exact_numbers": None,
            "exact_text": False,
            "warnings": warnings,
            "raw_text": [t for t in raw_fragments if t],
            "meta": {
                "chart_type": (ctype or "Unknown").strip().title(),
                "source": extra_meta.get("source") or ("ooxml" if used_xml else "image_text"),
                "used_items": [],
                "ignored_items": ignored_items,
                "percent_sum": None,
                "normalized": False,
                "caveat": (
                    "Числа считаны с изображения; возможна погрешность."
                    if (extra_meta.get("source") == "image_text" and Cfg.VISION_APPEND_CAVEAT_FOR_OCR)
                    else None
                ),
                "caption": caption,
                "context": context,
                "strict_passed": not getattr(Cfg, "FIG_STRICT", True),
                "chart_hint": chart_hint,
            },
        }
        _cache_put(key, payload)
        return payload

    # 5) Сбор финального текста
    ctype_u = (ctype or "Unknown").strip().title()
    is_ocr = (extra_meta.get("source") == "image_text") and (not used_xml)

    # Если тип не распознан, но есть хинт — используем хинт для NLG-ветки
    if ctype_u == "Unknown" and chart_hint:
        map_hint = {
            "pie": "Pie",
            "bar": "Bar",
            "stacked_bar": "Bar",
            "line": "Line",
            "histogram": "Bar",
        }
        ctype_u = map_hint.get(chart_hint, "Unknown")

    # Отсортируем по value по убыванию для консистентности
    items = sorted(items, key=lambda it: it.value, reverse=True)

    if ctype_u.startswith("Pie"):
        text = _nlg_pie(items, caption, is_ocr, percent_sum, normalized)
        ctype_final = "Pie"
    elif ctype_u.startswith("Bar"):
        text = _nlg_bar(items, caption, is_ocr)
        ctype_final = "Bar"
    elif ctype_u.startswith("Line"):
        text = _nlg_line(items, caption, is_ocr)
        ctype_final = "Line"
    else:
        text = _nlg_generic(items, caption, is_ocr)
        ctype_final = "Unknown"

    # 6) Сформируем meta и вернём
    used_items = [
        {"label": it.label, "value": round(float(it.value), 6), "unit": it.unit}
        for it in items
    ]

    if getattr(Cfg, "FIG_STRICT", True):
        strict_passed = True if used_xml else bool(ok)
    else:
        strict_passed = bool(used_items)

    warnings: List[str] = []
    ignored_items = int(extra_meta.get("ignored_items") or 0)
    if ignored_items:
        warnings.append(
            f"Игнорировано {ignored_items} подозрительных точек (низкая уверенность или некорректные метки)."
        )
    if normalized and percent_sum is not None:
        warnings.append(
            f"Доли диаграммы были нормализованы до 100 % (сумма исходных значений ≈ {percent_sum})."
        )
    if is_ocr and Cfg.VISION_APPEND_CAVEAT_FOR_OCR:
        warnings.append("Числа считаны с изображения; возможна небольшая погрешность.")

    payload = {
        "kind": "chart",
        "text": text,
        "data": used_items,
        "exact_numbers": used_items,
        "exact_text": False,
        "warnings": warnings,
        "raw_text": [t for t in raw_fragments if t],
        "meta": {
            "chart_type": ctype_final,
            "source": extra_meta.get("source") or ("ooxml" if used_xml else "image_text"),
            "used_items": used_items,
            "ignored_items": ignored_items,
            "percent_sum": percent_sum,
            "normalized": bool(normalized),
            "caveat": (
                "Числа считаны с изображения; возможна небольшая погрешность."
                if (is_ocr and Cfg.VISION_APPEND_CAVEAT_FOR_OCR)
                else None
            ),
            "caption": caption,
            "context": context,
            "strict_passed": strict_passed,
            "chart_hint": chart_hint,
        },
    }
    _cache_put(key, payload)
    return payload

    # 5) Сбор финального текста
    ctype_u = (ctype or "Unknown").strip().title()
    is_ocr = (extra_meta.get("source") == "image_text") and (not used_xml)

    # Если тип не распознан, но есть хинт — используем хинт для NLG-ветки
    if ctype_u == "Unknown" and chart_hint:
        map_hint = {
            "pie": "Pie",
            "bar": "Bar",
            "stacked_bar": "Bar",
            "line": "Line",
            "histogram": "Bar",
        }
        ctype_u = map_hint.get(chart_hint, "Unknown")

    # Отсортируем по value по убыванию для консистентности
    items = sorted(items, key=lambda it: it.value, reverse=True)

    if ctype_u.startswith("Pie"):
        text = _nlg_pie(items, caption, is_ocr, percent_sum, normalized)
        ctype_final = "Pie"
    elif ctype_u.startswith("Bar"):
        text = _nlg_bar(items, caption, is_ocr)
        ctype_final = "Bar"
    elif ctype_u.startswith("Line"):
        text = _nlg_line(items, caption, is_ocr)
        ctype_final = "Line"
    else:
        text = _nlg_generic(items, caption, is_ocr)
        ctype_final = "Unknown"

    # 6) Сформируем meta и вернём
    used_items = [
        {"label": it.label, "value": round(float(it.value), 6), "unit": it.unit}
        for it in items
    ]

    # strict_passed:
    #   - для OOXML считаем, что строгая проверка пройдена;
    #   - для vision используем результат _strict_validate;
    #   - при FIG_STRICT=False просто True, если есть данные.
    if getattr(Cfg, "FIG_STRICT", True):
        strict_passed = True if used_xml else bool(ok)
    else:
        strict_passed = bool(used_items)

    payload = {
        # явно помечаем, что это анализ диаграммы
        "kind": "chart",
        "text": text,
        "data": used_items,
        # exact_numbers — структурный список значений (вместо простого bool)
        "exact_numbers": used_items,
        # текст с картинки без OCR не считаем «строго точным»
        "exact_text": False,
        "meta": {
            "chart_type": ctype_final,
            "source": extra_meta.get("source") or ("ooxml" if used_xml else "image_text"),
            "used_items": used_items,
            "ignored_items": int(extra_meta.get("ignored_items") or 0),
            "percent_sum": percent_sum,
            "normalized": bool(normalized),
            "caveat": (
                "Числа считаны с изображения; возможна небольшая погрешность."
                if (is_ocr and Cfg.VISION_APPEND_CAVEAT_FOR_OCR)
                else None
            ),
            "caption": caption,
            "context": context,
            "strict_passed": strict_passed,
            "chart_hint": chart_hint,
        },
    }
    _cache_put(key, payload)
    return payload



def analyze_text_figure(
    image_path: str,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    intent: Optional[str] = "describe",
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Анализ «нечисловых» рисунков: схемы, оргструктуры, текстовые блоки, иллюстрации.

    Используем только vision_describe, без гарантий побуквенной точности текста.
    Возвращаем ответы в том же формате, что и для диаграмм:
      text, data, meta, kind, exact_numbers, exact_text, warnings, raw_text.
    """
    cap_hint = kwargs.get("caption_hint")
    if cap_hint and not caption:
        caption = cap_hint

    key = "text|" + _cache_key(image_path, caption, context, None)
    cached = _cache_get(key)
    if cached:
        return cached

    try:
        desc = vision_describe(image_or_images=image_path, lang=getattr(Cfg, "VISION_LANG", "ru"))
        base = (desc.get("description") or "").strip()
        if caption:
            text = f"{caption}: {base}" if base else caption
        else:
            text = base or "Описание изображения недоступно."
    except Exception:
        text = caption or "Описание изображения недоступно."

    raw_fragments: List[str] = []
    if caption:
        raw_fragments.append(_clean(caption))
    if context:
        raw_fragments.append(_clean(context))

    payload = {
        "kind": "text_figure",
        "text": text,
        "data": [],
        "exact_numbers": None,
        "exact_text": False,
        "warnings": [],
        "raw_text": [t for t in raw_fragments if t],
        "meta": {
            "source": "vision_describe",
            "caption": caption,
            "context": context,
            "exact_text": False,
        },
    }
    _cache_put(key, payload)
    return payload


def analyze_figure(
    image_path: str,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    intent: Optional[str] = "describe-with-numbers",
    chart_xml: Optional[bytes] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Универсальный вход: картинка → либо числовой анализ диаграммы, либо структурное описание.

    Логика:
      1) лёгкая классификация по caption/context/chart_xml (classify_figure);
      2) если рисунок числовой (диаграмма) → analyze_chart();
      3) иначе → analyze_text_figure();
      4) на выходе ВСЕГДА один и тот же формат:
         {text, data, meta, kind, exact_numbers, exact_text, warnings, raw_text}
         + meta.figure_kind / meta.figure_classification.

    При FIG_ANALYZER_ENABLED = False поведение эквивалентно старому analyze_figure (всегда как диаграмму).
    """
    # Классификация: что за рисунок
    cls = classify_figure(
        image_path=image_path,
        caption=caption,
        context=context,
        chart_xml=chart_xml,
    )

    numeric_intent = intent in {"describe-with-numbers", "chart", "numbers", "values"}

    # Если универсальный анализатор выключен — ведём себя как старый режим (диаграммы)
    if not getattr(Cfg, "FIG_ANALYZER_ENABLED", True):
        result = analyze_chart(
            image_path=image_path,
            caption=caption,
            context=context,
            intent=intent,
            chart_xml=chart_xml,
            **kwargs,
        )
    else:
        if cls.get("numeric") or numeric_intent:
            result = analyze_chart(
                image_path=image_path,
                caption=caption,
                context=context,
                intent=intent,
                chart_xml=chart_xml,
                **kwargs,
            )
        else:
            result = analyze_text_figure(
                image_path=image_path,
                caption=caption,
                context=context,
                intent=intent,
                **kwargs,
            )

    # Обогащаем результат метаданными универсального анализатора
    meta = result.setdefault("meta", {})
    meta.setdefault("figure_kind", cls.get("kind"))
    meta.setdefault("figure_classification", cls)

    # Верхнеуровневые поля, удобные для записи в БД
    # kind: что это за фигура с точки зрения универсальной классификации
    result.setdefault("kind", cls.get("kind"))

    # exact_numbers:
    #   - для числовых фигур: структурный список значений (если analyze_chart его уже положил),
    #     иначе fallback на result["data"];
    #   - для прочих: None.
    if cls.get("numeric"):
        if "exact_numbers" not in result:
            nums = result.get("data") or []
            result["exact_numbers"] = nums if nums else None
    else:
        result.setdefault("exact_numbers", None)

    # exact_text: сейчас без OCR нигде не гарантируем буквальной точности текста,
    # поэтому верхнеуровневое поле оставляем False, если его явно не выставили.
    result.setdefault("exact_text", False)

    # warnings/raw_text: если не выставлены анализаторами — подставим дефолты,
    # чтобы структура была стабильной.
    result.setdefault("warnings", [])
    result.setdefault("raw_text", [])

    return result
