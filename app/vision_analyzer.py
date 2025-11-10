# app/vision_analyzer.py
# -*- coding: utf-8 -*-
"""
Единая «мозговая» точка анализа изображений с числами.

API:
    analyze_figure(image_path, caption, context, intent, chart_xml=None) -> dict
        Возвращает:
        {
          "text": "<связный абзац с числами>",
          "meta": {
              "chart_type": "Pie|Bar|Line|Unknown",
              "source": "xml|vision",
              "used_items": [{"label":..., "value":..., "unit":"%|..."}],
              "ignored_items": int,
              "percent_sum": float|None,
              "normalized": bool,
              "caveat": str|None,
              "caption": str|None,
              "context": str|None,
          }
        }

Внутри:
  1) (опционально) парсит DOCX chart XML → точные данные.
  2) Vision Extract (OCR+визуальная разметка) → пары label→value (+unit, conf).
  3) Сведение/валидации: фильтр по conf, согласование единиц, нормализация Pie.
  4) Генерация связного текста (NLG) с числами.
  5) Кэш по хэшу изображения + контексту.

Зависимости:
  - .config.Cfg
  - .polza_client.vision_extract_values (и опционально vision_describe)
  - lxml (если есть) либо xml.etree.ElementTree
"""

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
        "v3",  # версия алгоритма (увеличь при изменениях логики)
        _sha256_file(image_path) if os.path.exists(image_path) else image_path,
        _sha256_bytes((caption or "").encode("utf-8")),
        _sha256_bytes((context or "").encode("utf-8")),
        _sha256_bytes(chart_xml or b""),
        f"conf={Cfg.VISION_EXTRACT_CONF_MIN}",
        f"pie_tol={Cfg.VISION_PIE_SUM_TOLERANCE_PP}",
        f"pct_dec={Cfg.VISION_PERCENT_DECIMALS}",
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
    s = s.replace(" ", " ").replace(" ", "")
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
    # округление: до целых для «штук»
    n = float(x)
    s = f"{int(round(n)):,}"
    # заменяем запятые на пробелы в качестве разрядного разделителя
    return s.replace(",", " ")


def _delta_fmt(a: float, b: float, unit: str) -> str:
    d = a - b
    sign = ""  # по тексту мы пишем «на X ... больше», там знак не нужен
    if unit == "%":
        return f"{sign}{_fmt_percent(abs(d), Cfg.VISION_PERCENT_DECIMALS)}".replace(" %", " п.п.")
    if unit:
        return f"{sign}{_fmt_number(abs(d))} {unit}"
    return f"{sign}{_fmt_number(abs(d))}"


def _maybe_normalize_pie(values: List[Tuple[str, float, str]]) -> Tuple[List[Tuple[str, float, str]], float, bool]:
    """
    Если это Pie и сумма 95..105% — пропорционально домножим до 100.
    Возвращает (values, sum_before, normalized)
    """
    if not values:
        return values, 0.0, False
    if not all((_norm_unit(u) in {"%", ""}) for (_, _, u) in values):
        return values, 0.0, False
    s = sum(v for (_, v, _) in values if v is not None)
    if s <= 0:
        return values, s, False
    target = float(Cfg.VISION_PIE_SUM_TARGET or 100.0)
    tol = float(Cfg.VISION_PIE_SUM_TOLERANCE_PP or 5.0)
    if abs(s - target) <= tol:
        # нормализуем
        k = target / s
        normed = [(lbl, v * k, "%") for (lbl, v, _) in values]
        return normed, s, True
    return values, s, False


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
    best = {}
    for u in norm:
        best[u] = best.get(u, 0) + 1
    winner = max(best.items(), key=lambda kv: kv[1])[0]
    return winner


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

    for r in rows:
        lbl = _clean(r.get("label"))
        unit = _norm_unit(r.get("unit")) or default_unit
        val = _to_float(r.get("value"))
        if val is None:
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
        meta = {"source": "xml", "ignored_items": 0}
        return ctype_xml, items_xml, meta, True

    # 2) Vision Extract
    if extract_obj:
        ctype, items, ignored = _from_extract(extract_obj)
        meta = {"source": "vision", "ignored_items": ignored}
        return ctype, items, meta, False

    return "Unknown", [], {"source": "none", "ignored_items": 0}, False


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


# ----------------------------- public API -----------------------------

def analyze_figure(
    image_path: str,
    caption: Optional[str] = None,
    context: Optional[str] = None,
    intent: Optional[str] = "describe-with-numbers",
    chart_xml: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Главная функция: картинка (+ подпись/контекст, опционально chart.xml) → связный абзац с числами.

    Порядок:
      1) кэш по хэшу изображения+контексту;
      2) если есть chart_xml → берём его данные как «истину»;
      3) иначе vision_extract_values();
      4) сведение/валидации/форматирование;
      5) NLG по типу графика.
    """
    # Ключ кэша
    key = _cache_key(image_path, caption, context, chart_xml)
    cached = _cache_get(key)
    if cached:
        return cached

    # 1) Получаем данные
    extract_obj: Optional[Dict[str, Any]] = None
    ctype: str = "Unknown"
    items: List[Item] = []
    extra_meta: Dict[str, Any] = {}
    used_xml = chart_xml is not None and len(chart_xml) > 0

    if used_xml:
        ctype, items, extra_meta, _ = _reconcile(chart_xml, None)
    else:
        if Cfg.VISION_EXTRACT_VALUES_ENABLED and Cfg.vision_active():
            try:
                extract_obj = vision_extract_values(
                    image_or_images=image_path,
                    caption_hint=_clean(caption),
                    ocr_hint=_clean(context),
                    temperature=0.0,
                    max_tokens=1400,
                    lang=Cfg.VISION_LANG,
                )
            except Exception as e:
                log.warning("vision_extract_values failed: %s", e)
                extract_obj = None
        ctype, items, extra_meta, _ = _reconcile(None, extract_obj)

    # 2) Фильтрация по уверенности — в extract_obj модель уже дала conf, мы их учитывали при парсинге значений.
    # (Если понадобится более строгий фильтр — можно дополнить здесь.)

    # 3) Нормализация для Pie
    percent_sum = None
    normalized = False
    if (ctype or "").lower().startswith("pie") or (items and all(it.unit in {"%", ""} for it in items)):
        # если в items unit пустой — назначим '%' (vision может не прислать)
        items = [Item(it.label, it.value, it.unit or "%") for it in items]
        items, s, normalized = _maybe_normalize_pie(items)
        percent_sum = round(s, Cfg.VISION_PERCENT_DECIMALS)

    # Если после сводки нет данных — попробуем хотя бы «описание без чисел»
    if not items:
        # Попробуем краткое описание (не критично, просто улучшает UX)
        try:
            desc = vision_describe(image_or_images=image_path, lang=Cfg.VISION_LANG)
            text = (desc.get("description") or "").strip()
            if caption:
                text = f"{caption}: {text}" if text else caption
        except Exception:
            text = caption or "Описание изображения недоступно."
        payload = {
            "text": text,
            "meta": {
                "chart_type": ctype or "Unknown",
                "source": extra_meta.get("source") or ("xml" if used_xml else "vision"),
                "used_items": [],
                "ignored_items": int(extra_meta.get("ignored_items") or 0),
                "percent_sum": None,
                "normalized": False,
                "caveat": ("Числа считаны с изображения; возможна погрешность." if (not used_xml and Cfg.VISION_APPEND_CAVEAT_FOR_OCR) else None),
                "caption": caption,
                "context": context,
            },
        }
        _cache_put(key, payload)
        return payload

    # 4) Сбор финального текста
    ctype_u = (ctype or "Unknown").strip().title()
    is_ocr = (extra_meta.get("source") == "vision") and (not used_xml)

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

    # 5) Сформируем meta и вернём
    used_items = [{"label": it.label, "value": round(float(it.value), 6), "unit": it.unit} for it in items]
    payload = {
        "text": text,
        "meta": {
            "chart_type": ctype_final,
            "source": extra_meta.get("source") or ("xml" if used_xml else "vision"),
            "used_items": used_items,
            "ignored_items": int(extra_meta.get("ignored_items") or 0),
            "percent_sum": percent_sum,
            "normalized": bool(normalized),
            "caveat": ("Числа считаны с изображения; возможна небольшая погрешность." if (is_ocr and Cfg.VISION_APPEND_CAVEAT_FOR_OCR) else None),
            "caption": caption,
            "context": context,
        },
    }
    _cache_put(key, payload)
    return payload
