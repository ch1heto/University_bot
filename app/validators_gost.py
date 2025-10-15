from __future__ import annotations
import re
from typing import List, Dict, Any, Optional, Tuple

# --------- Настройки профиля (упрощённый ГОСТ-профиль) ---------

EXPECTED_FONT_NAMES = {
    "times new roman",
    "timesnewromanpsmt",
    "timesnewroman",
    "tnr",
}
SIZE_PT_RANGE = (13.5, 14.5)        # ~ 14 pt
LINE_SPACING_TARGET = 1.5            # стандарт 1.5
LINE_SPACING_TOL = 0.2               # допустимое отклонение
FIRST_LINE_INDENT_PT_RANGE = (28, 42)  # ~1.0–1.5 см (часто требуют 1.25 см ≈ 35.4 pt)
REQUIRE_JUSTIFY = True

# Разделы, которые обычно ждут в ВКР (проверяется «похоже/приблизительно»)
REQUIRED_SECTIONS = [
    "введение",
    "заключение",
    "список литературы",  # также проверим «источники»/«библиография»
]

CAPTION_RE_TABLE = re.compile(
    r"^\s*(табл(?:ица)?|table)\.?\s*\d+(?:[.,]\d+)?(?:\s*[-—:]\s*.*)?\s*$",
    re.IGNORECASE
)
CAPTION_RE_FIG = re.compile(
    r"^\s*(рис(?:унок)?|figure)\.?\s*\d+(?:[.,]\d+)?(?:\s*[-—:]\s*.*)?\s*$",
    re.IGNORECASE
)

# --------- Вспомогалки ---------

def _norm(s: str | None) -> str:
    return (s or "").replace("\xa0", " ").strip()

def _is_paragraph(sec: Dict[str, Any]) -> bool:
    return (sec.get("element_type") or "").lower() in ("paragraph",)

def _is_heading(sec: Dict[str, Any]) -> bool:
    return (sec.get("element_type") or "").lower() in ("heading",)

def _is_table(sec: Dict[str, Any]) -> bool:
    return (sec.get("element_type") or "").lower() in ("table",)

def _is_figure(sec: Dict[str, Any]) -> bool:
    return (sec.get("element_type") or "").lower() in ("figure",)

def _fonts_list(attrs: Dict[str, Any]) -> List[str]:
    fonts = (attrs or {}).get("fonts") or []
    return [(_norm(f)).lower() for f in fonts if _norm(f)]

def _sizes_list(attrs: Dict[str, Any]) -> List[float]:
    return list((attrs or {}).get("font_sizes_pt") or [])

def _alignment(attrs: Dict[str, Any]) -> Optional[str]:
    a = (attrs or {}).get("alignment")
    return (a or "").lower() if isinstance(a, str) else a

def _first_line_indent_pt(attrs: Dict[str, Any]) -> Optional[float]:
    v = (attrs or {}).get("first_line_indent_pt")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _line_spacing(attrs: Dict[str, Any]) -> Optional[float]:
    v = (attrs or {}).get("line_spacing")
    try:
        return float(v) if v is not None else None
    except Exception:
        return None

def _title_text(sec: Dict[str, Any]) -> str:
    return _norm(sec.get("title"))

def _table_name(sec: Dict[str, Any]) -> str:
    # ожидаем «Таблица N» или подписанный титул
    return _norm(sec.get("title") or sec.get("section_path") or "Таблица")

def _figure_name(sec: Dict[str, Any]) -> str:
    return _norm(sec.get("title") or sec.get("section_path") or "Рисунок")

# --------- Проверки ---------

def check_structure(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Проверка наличия базовых разделов (введение/заключение/список литературы)."""
    titles = []
    for s in sections:
        if _is_heading(s):
            t = _norm(_title_text(s)).lower()
            if t:
                titles.append(t)

    issues: List[Dict[str, Any]] = []
    def present(name: str) -> bool:
        n = name.lower()
        for t in titles:
            if n in t:
                return True
        return False

    for req in REQUIRED_SECTIONS:
        if not present(req):
            issues.append({
                "type": "heading_presence",
                "severity": "warn",
                "message": f"Не найден раздел, похожий на «{req}».",
                "where": None,
            })

    # альтернативные формулировки для списка литературы
    if not any(k in " ".join(titles) for k in ["список литературы", "библиограф", "источн"]):
        issues.append({
            "type": "heading_presence",
            "severity": "warn",
            "message": "Не найден «Список литературы» (или «Библиография», «Источники»).",
            "where": None,
        })

    return issues

def check_paragraph_style(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Проверки по абзацам (евристики):
      — шрифт ≈ Times New Roman,
      — размер ≈ 14 pt,
      — межстрочный ≈ 1.5,
      — выравнивание по ширине,
      — красная строка ≈ 1.25 см.
    Возвращает (issues, stats)
    """
    issues: List[Dict[str, Any]] = []
    total = 0
    ok_font = ok_size = ok_line = ok_align = ok_indent = 0

    for s in sections:
        if not _is_paragraph(s):
            continue
        total += 1
        attrs = s.get("attrs") or {}

        # font
        fonts = _fonts_list(attrs)
        if fonts:
            if any(f in EXPECTED_FONT_NAMES for f in fonts):
                ok_font += 1
            else:
                issues.append({
                    "type": "font",
                    "severity": "warn",
                    "message": f"Необычный шрифт: {', '.join(fonts)} (ожидается Times New Roman).",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })
        # size
        sizes = _sizes_list(attrs)
        if sizes:
            in_range = any(SIZE_PT_RANGE[0] <= sz <= SIZE_PT_RANGE[1] for sz in sizes)
            if in_range:
                ok_size += 1
            else:
                issues.append({
                    "type": "size",
                    "severity": "warn",
                    "message": f"Кегль вне диапазона {SIZE_PT_RANGE[0]}–{SIZE_PT_RANGE[1]} pt: {sizes}",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })
        # line spacing
        ls = _line_spacing(attrs)
        if ls is not None:
            if abs(ls - LINE_SPACING_TARGET) <= LINE_SPACING_TOL:
                ok_line += 1
            else:
                issues.append({
                    "type": "line_spacing",
                    "severity": "warn",
                    "message": f"Межстрочный интервал {ls} (ожидается ~{LINE_SPACING_TARGET}).",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })
        # alignment
        al = _alignment(attrs)
        if REQUIRE_JUSTIFY and al is not None:
            if str(al).lower() == "justify":
                ok_align += 1
            else:
                issues.append({
                    "type": "alignment",
                    "severity": "warn",
                    "message": f"Выравнивание: {al} (ожидается по ширине).",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })
        # first line indent
        fi = _first_line_indent_pt(attrs)
        if fi is not None:
            if FIRST_LINE_INDENT_PT_RANGE[0] <= fi <= FIRST_LINE_INDENT_PT_RANGE[1]:
                ok_indent += 1
            else:
                issues.append({
                    "type": "first_line_indent",
                    "severity": "warn",
                    "message": f"Красная строка {round(fi,1)} pt (ожидается ~1.25 см ≈ 35 pt).",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })

    stats = {
        "paragraphs_total": total,
        "ok_font": ok_font,
        "ok_size": ok_size,
        "ok_line_spacing": ok_line,
        "ok_alignment": ok_align,
        "ok_first_line_indent": ok_indent,
    }
    return issues, stats

def check_tables_and_figures(sections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Подписи и базовая форма таблиц/рисунков."""
    issues: List[Dict[str, Any]] = []
    t_count = f_count = 0
    t_ok_caption = f_ok_caption = 0
    t_nonempty = 0

    for s in sections:
        if _is_table(s):
            t_count += 1
            name = _table_name(s)
            if CAPTION_RE_TABLE.match(name):
                t_ok_caption += 1
            else:
                issues.append({
                    "type": "table_caption_format",
                    "severity": "warn",
                    "message": f"Подпись таблицы не похожа на ГОСТ: «{name}». Ожидается «Таблица N.М — ...».",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })
            txt = _norm(s.get("text") or "")
            if txt and txt != "(пустая таблица)":
                t_nonempty += 1
            else:
                issues.append({
                    "type": "table_empty",
                    "severity": "warn",
                    "message": f"Похоже, таблица пустая или не распознана.",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")}
                })

        elif _is_figure(s):
            f_count += 1
            name = _figure_name(s)
            if CAPTION_RE_FIG.match(name):
                f_ok_caption += 1
            else:
                issues.append({
                    "type": "figure_caption_format",
                    "severity": "warn",
                    "message": f"Подпись рисунка не похожа на ГОСТ: «{name}». Ожидается «Рисунок N.М — ...».",
                    "where": {"section_path": s.get("section_path"), "page": s.get("page")},
                    "sample": (_norm(s.get("text") or "")[:180])
                })

    stats = {
        "tables_total": t_count,
        "tables_with_caption_ok": t_ok_caption,
        "tables_nonempty": t_nonempty,
        "figures_total": f_count,
        "figures_with_caption_ok": f_ok_caption,
    }
    return issues, stats

def check_heading_levels(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Грубая проверка уровней заголовков: не должно быть скачков уровня > 1.
    (работает, если в секциях присутствует поле 'level' и 'element_type'='heading').
    """
    issues: List[Dict[str, Any]] = []
    last_level: Optional[int] = None
    for s in sections:
        if not _is_heading(s):
            continue
        lvl = s.get("level")
        if isinstance(lvl, int):
            if last_level is not None and (lvl - last_level) > 1:
                issues.append({
                    "type": "heading_level_jump",
                    "severity": "warn",
                    "message": f"Скачок уровня заголовка: {last_level} → {lvl}.",
                    "where": {"title": _title_text(s), "page": s.get("page")}
                })
            last_level = lvl
    return issues

def check_bibliography_presence(sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Простейшая проверка наличия блока «Список литературы/Источники/Библиография».
    Для PDF/минимальных секций — может не сработать (тогда просто игнор).
    """
    issues: List[Dict[str, Any]] = []
    found_head = False
    for s in sections:
        if _is_heading(s):
            t = _norm(_title_text(s)).lower()
            if any(k in t for k in ["список литературы", "источники", "библиограф"]):
                found_head = True
                break
    if not found_head:
        issues.append({
            "type": "bibliography_missing",
            "severity": "warn",
            "message": "Не найден заголовок раздела «Список литературы/Источники/Библиография».",
            "where": None
        })
    return issues

# --------- Публичный API ---------

def validate_gost(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Основная функция валидации по «ГОСТ-профилю».
    Возвращает словарь:
    {
      "summary": {...},             # агрегаты/проценты
      "issues": [ {...}, ... ]      # список замечаний (type, severity, message, where?, sample?)
    }
    """
    all_issues: List[Dict[str, Any]] = []

    # Структура (введение/заключение/список литературы)
    all_issues += check_structure(sections)

    # Уровни заголовков
    all_issues += check_heading_levels(sections)

    # Абзацы: стиль
    para_issues, para_stats = check_paragraph_style(sections)
    all_issues += para_issues

    # Таблицы/рисунки: подписи и пустые таблицы
    tf_issues, tf_stats = check_tables_and_figures(sections)
    all_issues += tf_issues

    # Библиография (наличие заголовка)
    all_issues += check_bibliography_presence(sections)

    # Сводка
    summary = {
        **para_stats,
        **tf_stats,
        "issues_total": len(all_issues),
    }
    return {"summary": summary, "issues": all_issues}


def render_report(report: Dict[str, Any], *, max_issues: int = 20) -> str:
    """
    Человекочитаемый отчёт для ответа студенту в чате.
    """
    if not report:
        return "Не удалось построить отчёт по ГОСТ."

    s = report.get("summary") or {}
    lines: List[str] = []

    # Краткая сводка
    lines.append("Проверка оформления по базовому профилю ГОСТ (эвристики):")
    if s:
        lines.append(
            f"- Абзацев: {s.get('paragraphs_total', 0)}; "
            f"шрифт ОК: {s.get('ok_font', 0)}, кегль ОК: {s.get('ok_size', 0)}, "
            f"интервал ОК: {s.get('ok_line_spacing', 0)}, выравнивание ОК: {s.get('ok_alignment', 0)}, "
            f"красная строка ОК: {s.get('ok_first_line_indent', 0)}."
        )
        lines.append(
            f"- Таблиц: {s.get('tables_total', 0)} (подпись ОК: {s.get('tables_with_caption_ok', 0)}, "
            f"с непустым содержимым: {s.get('tables_nonempty', 0)}); "
            f"Рисунков: {s.get('figures_total', 0)} (подпись ОК: {s.get('figures_with_caption_ok', 0)})."
        )

    issues = report.get("issues") or []
    if not issues:
        lines.append("\nЗамечаний не найдено — базовые требования соблюдены.")
        return "\n".join(lines)

    lines.append(f"\nНайдено замечаний: {len(issues)}. Ключевые ({min(len(issues), max_issues)}):")
    for i, it in enumerate(issues[:max_issues], 1):
        tp = it.get("type") or "issue"
        sev = it.get("severity") or "info"
        msg = it.get("message") or ""
        where = it.get("where")
        place = ""
        if isinstance(where, dict):
            sec = where.get("section_path")
            pg = where.get("page")
            if sec:
                place += f" • {sec}"
            if pg:
                place += f" (стр. {pg})"
        sample = it.get("sample")
        if sample:
            lines.append(f"{i}. [{sev}] {msg}{place}\n    «{sample}»")
        else:
            lines.append(f"{i}. [{sev}] {msg}{place}")

    lines.append("\nВажно: проверки эвристические — требования могут отличаться в вашем вузе/кафедре.")
    return "\n".join(lines)
