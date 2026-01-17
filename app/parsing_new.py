# app/parsing_new.py
"""
Новый парсер документов на базе docx2python.
Заменяет старый parsing.py (2249 строк) и ooxml_lite.py (1521 строка).
Всего ~200 строк вместо 3770!
"""
from __future__ import annotations
import hashlib
from pathlib import Path

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from docx2python import docx2python
import json

# Импортируем утилиты из старого кода
try:
    from .utils import save_bytes, safe_filename, sha256_bytes, ensure_dir
    from .config import Cfg
except:
    # Fallback для тестирования
    def save_bytes(data, filename, dirpath): 
        return str(Path(dirpath) / filename)
    def safe_filename(name, max_len=180): 
        return name
    def sha256_bytes(data): 
        return "hash"
    def ensure_dir(path):
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        return p
    class Cfg:
        UPLOAD_DIR = "./uploads"


# ==================== КОНСТАНТЫ ====================

FIG_CAP_RE = re.compile(
    r"^(?:Рис\.?|Рисунок|Figure|Fig\.?)\s*\.?\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)",
    re.IGNORECASE
)

TAB_CAP_RE = re.compile(
    r"^(?:Таблица|Table|Табл\.?)\s*\.?\s*(?:№\s*)?([A-Za-zА-Яа-я]?\s*\d+(?:[.,]\d+)*)",
    re.IGNORECASE
)

HEAD_STYLE_RE = re.compile(r"(heading|заголовок)\s*([1-6])", re.IGNORECASE)


# ==================== ОСНОВНЫЕ ФУНКЦИИ ====================

def parse_docx(file_path: str) -> Dict[str, Any]:
    """
    Парсинг .docx файла через python-docx (более надёжный).
    Возвращает словарь с секциями, таблицами, рисунками.
    """
    file_path = str(file_path)
    
    try:
        from docx import Document
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.table import _Cell, Table
        from docx.text.paragraph import Paragraph
    except ImportError:
        return {
            'sections': [{'text': 'Установите python-docx: pip install python-docx', 'level': 0}],
            'tables': [],
            'figures': [],
            'metadata': {},
            'raw_text': '',
        }
    
    try:
        doc = Document(file_path)
        
        # Директория для сохранения изображений
        upload_dir = getattr(Cfg, 'UPLOAD_DIR', './uploads')
        images_dir = Path(upload_dir) / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # ===================================================================
        # ИЗВЛЕЧЕНИЕ СЕКЦИЙ (ПАРАГРАФЫ)
        # ===================================================================
        sections = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            # Определяем уровень заголовка
            level = 0
            title = ""
            if para.style.name.startswith('Heading'):
                try:
                    level = int(para.style.name.replace('Heading ', ''))
                    title = text
                except:
                    level = 1
                    title = text
            
            sections.append({
                'text': text,
                'level': level,
                'title': title,
            })
        
        # ===================================================================
        # ИЗВЛЕЧЕНИЕ ТАБЛИЦ
        # ===================================================================
        tables = []
        for table_idx, table in enumerate(doc.tables, start=1):
            rows_data = []
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells]
                rows_data.append(row_data)
            
            if rows_data:
                tables.append({
                    'id': table_idx,
                    'num': str(table_idx),
                    'caption': f'Таблица {table_idx}',
                    'data': rows_data,
                    'rows': len(rows_data),
                    'cols': len(rows_data[0]) if rows_data else 0,
                })
        
        # ===================================================================
        # ИЗВЛЕЧЕНИЕ РИСУНКОВ (ИЗОБРАЖЕНИЙ)
        # ===================================================================
        figures = []
        figure_counter = 0
        
        # Извлекаем все изображения из relationships
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                figure_counter += 1
                
                try:
                    # Получаем бинарные данные изображения
                    image_part = rel.target_part
                    image_data = image_part.blob
                    
                    # Определяем расширение
                    ext = rel.target_ref.split('.')[-1]
                    if not ext.startswith('.'):
                        ext = '.' + ext
                    
                    # Генерируем имя файла
                    img_hash = hashlib.sha256(image_data).hexdigest()[:8]
                    safe_name = f"fig_{figure_counter}_{img_hash}{ext}"
                    img_path = images_dir / safe_name
                    
                    # Сохраняем изображение
                    with open(img_path, 'wb') as f:
                        f.write(image_data)
                    
                    # Ищем подпись рисунка в тексте
                    caption = _find_figure_caption_in_sections(sections, figure_counter)
                    
                    # Извлекаем номер рисунка из подписи
                    fig_num = str(figure_counter)
                    if caption:
                        import re
                        match = re.search(r'[Рр]исунок\s+(\d+)', caption)
                        if match:
                            fig_num = match.group(1)
                    
                    figures.append({
                        'id': figure_counter,
                        'num': fig_num,
                        'caption': caption or f'Рисунок {figure_counter}',
                        'image_path': str(img_path.resolve()),
                        'original_name': rel.target_ref,
                        'kind': 'image',
                    })
                    
                except Exception as e:
                    print(f"Ошибка извлечения рисунка {figure_counter}: {e}")
        
        # ===================================================================
        # МЕТАДАННЫЕ
        # ===================================================================
        core_props = doc.core_properties
        metadata = {
            'author': core_props.author or '',
            'title': core_props.title or '',
            'subject': core_props.subject or '',
            'created': str(core_props.created) if core_props.created else '',
            'modified': str(core_props.modified) if core_props.modified else '',
        }
        
        # ===================================================================
        # ПОЛНЫЙ ТЕКСТ
        # ===================================================================
        raw_text = '\n'.join([s['text'] for s in sections])
        
        return {
            'sections': sections,
            'tables': tables,
            'figures': figures,
            'metadata': metadata,
            'raw_text': raw_text,
        }
    
    except Exception as e:
        print(f"Ошибка парсинга DOCX: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            'sections': [{'text': f'Ошибка парсинга: {e}', 'level': 0}],
            'tables': [],
            'figures': [],
            'metadata': {},
            'raw_text': '',
        }


def _find_figure_caption_in_sections(sections: list, figure_num: int) -> str:
    """Ищет подпись рисунка в секциях"""
    import re
    
    # Паттерны для поиска подписей
    patterns = [
        rf'[Рр]исунок\s+{figure_num}[\.:\s—–-]+(.+)',
        rf'[Ff]igure\s+{figure_num}[\.:\s—–-]+(.+)',
        rf'[Рр]ис\.\s*{figure_num}[\.:\s—–-]+(.+)',
    ]
    
    for section in sections:
        text = section.get('text', '')
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                caption = match.group(0).strip()
                # Ограничиваем длину подписи (обычно не более 200 символов)
                if len(caption) > 200:
                    caption = caption[:200] + '...'
                return caption
    
    return f'Рисунок {figure_num}'


def parse_doc(file_path: str) -> Dict[str, Any]:
    """
    Парсинг старого формата .doc (через конвертацию в .docx или antiword).
    Fallback: если не можем сконвертировать, возвращаем минимальную структуру.
    """
    file_path = str(file_path)
    
    # Попытка 1: конвертация через LibreOffice (если установлен)
    try:
        docx_path = _convert_doc_to_docx(file_path)
        if docx_path and os.path.exists(docx_path):
            return parse_docx(docx_path)
    except Exception as e:
        print(f"Не удалось сконвертировать .doc: {e}")
    
    # Попытка 2: antiword (если установлен)
    try:
        import subprocess
        text = subprocess.check_output(['antiword', file_path], text=True)
        return {
            'sections': [{'text': text, 'level': 0, 'title': ''}],
            'tables': [],
            'figures': [],
            'metadata': {},
            'raw_text': text,
        }
    except:
        pass
    
    # Fallback: минимальная структура
    return {
        'sections': [{'text': 'Не удалось распарсить .doc файл', 'level': 0, 'title': ''}],
        'tables': [],
        'figures': [],
        'metadata': {},
        'raw_text': '',
    }


def parse_pdf(file_path: str) -> Dict[str, Any]:
    """
    Парсинг PDF документа.
    Использует pdfplumber для извлечения текста и таблиц.
    
    Примечание: PDF парсинг сложнее, чем DOCX, потому что PDF - это
    "набор инструкций для рендеринга", а не структурированный документ.
    """
    file_path = str(file_path)
    
    try:
        import pdfplumber
    except ImportError:
        return {
            'sections': [{'text': 'Для парсинга PDF установите: pip install pdfplumber', 'level': 0}],
            'tables': [],
            'figures': [],
            'metadata': {},
            'raw_text': '',
        }
    
    sections = []
    tables = []
    all_text = []
    
    try:
        with pdfplumber.open(file_path) as pdf:
            # Метаданные
            metadata = {
                'pages': len(pdf.pages),
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
            }
            
            for page_num, page in enumerate(pdf.pages, start=1):
                # Извлекаем текст страницы
                text = page.extract_text() or ""
                all_text.append(text)
                
                if text.strip():
                    sections.append({
                        'text': text,
                        'level': 0,
                        'title': '',
                        'page': page_num,
                    })
                
                # Извлекаем таблицы
                page_tables = page.extract_tables()
                for table_idx, table in enumerate(page_tables or []):
                    if table:
                        # Очищаем пустые значения
                        cleaned_table = [
                            [cell.strip() if cell else '' for cell in row]
                            for row in table
                        ]
                        
                        tables.append({
                            'id': len(tables) + 1,
                            'num': f"{page_num}.{table_idx + 1}",
                            'caption': f"Таблица на странице {page_num}",
                            'data': cleaned_table,
                            'rows': len(cleaned_table),
                            'cols': len(cleaned_table[0]) if cleaned_table else 0,
                            'page': page_num,
                        })
        
        return {
            'sections': sections,
            'tables': tables,
            'figures': [],  # PDF изображения требуют PyMuPDF (fitz)
            'metadata': metadata,
            'raw_text': '\n\n'.join(all_text),
        }
    
    except Exception as e:
        print(f"Ошибка парсинга PDF: {e}")
        return {
            'sections': [{'text': f'Ошибка парсинга PDF: {e}', 'level': 0}],
            'tables': [],
            'figures': [],
            'metadata': {},
            'raw_text': '',
        }


def save_upload(data: bytes, filename: str, dirpath: str = None) -> str:
    """
    Сохраняет загруженный файл на диск.
    Совместимо со старым API.
    """
    if dirpath is None:
        dirpath = getattr(Cfg, 'UPLOAD_DIR', './uploads')
    
    ensure_dir(dirpath)
    safe_name = safe_filename(filename)
    filepath = Path(dirpath) / safe_name
    
    filepath.write_bytes(data or b"")
    return str(filepath.resolve())


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def _safe_get_text(doc_content) -> str:
    """
    Безопасное извлечение текста из документа.
    doc_content.text иногда падает из-за проблем с заголовками/footer.
    """
    try:
        return doc_content.text or ""
    except Exception as e:
        # Fallback: собираем текст из body вручную
        try:
            all_text = []
            for section in doc_content.body:
                for para in section:
                    if isinstance(para, list):
                        text = _flatten_list_to_text(para)
                    else:
                        text = str(para).strip()
                    if text:
                        all_text.append(text)
            return '\n'.join(all_text)
        except:
            return ""

def _extract_sections(doc_content, file_path: str) -> List[Dict[str, Any]]:
    """
    Извлекает секции (параграфы) документа.
    """
    sections = []
    
    # doc_content.body - это вложенная структура:
    # [раздел][параграф/таблица][ячейка (если таблица)][строка]
    for section_idx, section in enumerate(doc_content.body):
        for para_idx, para in enumerate(section):
            # Пропускаем таблицы (они обрабатываются отдельно)
            if isinstance(para, list) and len(para) > 0 and isinstance(para[0], list):
                continue
            
            # Получаем текст параграфа
            if isinstance(para, list):
                text = _flatten_list_to_text(para)
            else:
                text = str(para).strip()
            
            if not text:
                continue
            
            # Определяем уровень заголовка (если есть)
            level = 0
            title = ""
            if _is_heading(text):
                level, title = _parse_heading(text)
            
            sections.append({
                'text': text,
                'level': level,
                'title': title,
                'section_idx': section_idx,
                'para_idx': para_idx,
            })
    
    return sections


def _extract_tables(doc_content) -> List[Dict[str, Any]]:
    """
    Извлекает таблицы из документа.
    """
    tables = []
    table_counter = 0
    
    for section_idx, section in enumerate(doc_content.body):
        for item_idx, item in enumerate(section):
            # Проверяем, является ли элемент таблицей
            if isinstance(item, list) and len(item) > 0 and isinstance(item[0], list):
                # Это таблица!
                table_data = []
                
                for row in item:
                    if isinstance(row, list):
                        # Преобразуем каждую ячейку в строку
                        row_data = [_flatten_list_to_text(cell) if isinstance(cell, list) else str(cell).strip() 
                                    for cell in row]
                        table_data.append(row_data)
                
                # Ищем подпись таблицы (обычно в предыдущем или следующем параграфе)
                caption = _find_table_caption(doc_content.body, section_idx, item_idx)
                
                # Извлекаем номер таблицы из подписи
                table_num = None
                if caption:
                    match = TAB_CAP_RE.match(caption)
                    if match:
                        table_num = match.group(1).strip()
                
                table_counter += 1
                tables.append({
                    'id': table_counter,
                    'num': table_num,
                    'caption': caption,
                    'data': table_data,
                    'rows': len(table_data),
                    'cols': len(table_data[0]) if table_data else 0,
                    'section_idx': section_idx,
                    'item_idx': item_idx,
                })
    
    return tables


def _extract_figures(doc_content, file_path: str) -> List[Dict[str, Any]]:
    """
    Извлекает рисунки (изображения) из документа.
    """
    figures = []
    figure_counter = 0
    
    # Директория для сохранения изображений
    upload_dir = getattr(Cfg, 'UPLOAD_DIR', './uploads')
    images_dir = ensure_dir(Path(upload_dir) / 'images')
    
    # docx2python автоматически извлекает изображения
    for img_name, img_data in doc_content.images.items():
        figure_counter += 1
        
        # Сохраняем изображение
        img_hash = sha256_bytes(img_data)
        ext = Path(img_name).suffix or '.png'
        safe_name = f"fig_{figure_counter}_{img_hash[:8]}{ext}"
        img_path = images_dir / safe_name
        img_path.write_bytes(img_data)
        
        # Ищем подпись рисунка в тексте документа
        caption = _find_figure_caption(doc_content.text, figure_counter)
        
        # Извлекаем номер рисунка из подписи
        fig_num = None
        if caption:
            match = FIG_CAP_RE.match(caption)
            if match:
                fig_num = match.group(1).strip()
        
        figures.append({
            'id': figure_counter,
            'num': fig_num,
            'caption': caption,
            'image_path': str(img_path.resolve()),
            'original_name': img_name,
            'kind': 'image',  # может быть 'chart', 'diagram', 'photo'
        })
    
    return figures


def _extract_metadata(doc_content) -> Dict[str, Any]:
    """
    Извлекает метаданные документа (автор, дата создания и т.д.)
    """
    metadata = {}
    
    try:
        # docx2python предоставляет properties
        props = getattr(doc_content, 'core_properties', None) or getattr(doc_content, 'properties', None)
        if props:
            metadata = {
                'author': props.get('author', ''),
                'title': props.get('title', ''),
                'subject': props.get('subject', ''),
                'created': props.get('created', ''),
                'modified': props.get('modified', ''),
            }
    except:
        pass
    
    return metadata


# ==================== УТИЛИТЫ ====================

def _flatten_list_to_text(lst) -> str:
    """
    Преобразует вложенный список в плоский текст.
    """
    if isinstance(lst, str):
        return lst.strip()
    elif isinstance(lst, list):
        return ' '.join(_flatten_list_to_text(item) for item in lst).strip()
    else:
        return str(lst).strip()


def _is_heading(text: str) -> bool:
    """
    Проверяет, является ли текст заголовком.
    """
    text = text.strip()
    # Эвристика: заголовки обычно короткие и начинаются с цифры или заглавной буквы
    if len(text) < 5 or len(text) > 200:
        return False
    
    # Проверяем паттерны заголовков
    if re.match(r'^\d+\.?\s+[А-ЯЁA-Z]', text):  # "1. Введение" или "1 ВВЕДЕНИЕ"
        return True
    
    if text.isupper() and len(text.split()) <= 10:  # "ГЛАВА 1. ТЕОРЕТИЧЕСКИЕ ОСНОВЫ"
        return True
    
    return False


def _parse_heading(text: str) -> Tuple[int, str]:
    """
    Определяет уровень заголовка и его текст.
    """
    text = text.strip()
    
    # Паттерн: "1.2.3 Название"
    match = re.match(r'^(\d+(?:\.\d+)*)\s+(.+)', text)
    if match:
        num_parts = match.group(1).split('.')
        level = len(num_parts)
        title = match.group(2).strip()
        return level, title
    
    # Паттерн: "ГЛАВА 1. НАЗВАНИЕ"
    if text.isupper():
        return 1, text
    
    return 0, text


def _find_table_caption(body, section_idx: int, item_idx: int) -> Optional[str]:
    """
    Ищет подпись таблицы в соседних параграфах.
    """
    # Проверяем предыдущий параграф
    if item_idx > 0:
        prev_item = body[section_idx][item_idx - 1]
        if isinstance(prev_item, str) or (isinstance(prev_item, list) and not isinstance(prev_item[0], list)):
            text = _flatten_list_to_text(prev_item)
            if TAB_CAP_RE.match(text):
                return text
    
    # Проверяем следующий параграф
    if item_idx < len(body[section_idx]) - 1:
        next_item = body[section_idx][item_idx + 1]
        if isinstance(next_item, str) or (isinstance(next_item, list) and not isinstance(next_item[0], list)):
            text = _flatten_list_to_text(next_item)
            if TAB_CAP_RE.match(text):
                return text
    
    return None


def _find_figure_caption(full_text: str, figure_num: int) -> Optional[str]:
    """
    Ищет подпись рисунка по тексту документа.
    """
    # Ищем строки вида "Рисунок 1 - Название"
    pattern = rf"(?:Рис\.?|Рисунок|Figure)\s*\.?\s*(?:№\s*)?{figure_num}\s*[-–—]\s*(.+)"
    match = re.search(pattern, full_text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    
    return None


def _convert_doc_to_docx(doc_path: str) -> Optional[str]:
    """
    Конвертирует .doc в .docx через LibreOffice (если установлен).
    """
    import subprocess
    import tempfile
    
    try:
        # Создаём временную директорию
        temp_dir = tempfile.mkdtemp()
        
        # Конвертируем через soffice
        subprocess.run([
            'soffice',
            '--headless',
            '--convert-to', 'docx',
            '--outdir', temp_dir,
            doc_path
        ], check=True, timeout=30)
        
        # Находим сконвертированный файл
        docx_name = Path(doc_path).stem + '.docx'
        docx_path = Path(temp_dir) / docx_name
        
        if docx_path.exists():
            return str(docx_path)
    except:
        pass
    
    return None


# ==================== СОВМЕСТИМОСТЬ СО СТАРЫМ API ====================

def build_index(doc_path: str, output_dir: str = None) -> Dict[str, Any]:
    """
    Создаёт индекс документа (аналог oox_build_index).
    Для совместимости со старым кодом.
    """
    parsed = parse_docx(doc_path)
    
    index = {
        'figures': parsed.get('figures', []),
        'tables': parsed.get('tables', []),
        'sections': parsed.get('sections', []),
        'metadata': parsed.get('metadata', {}),
    }
    
    # Сохраняем индекс в JSON (если нужно)
    if output_dir:
        ensure_dir(output_dir)
        index_path = Path(output_dir) / 'index.json'
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
    
    return index


def figure_lookup(index: Dict, figure_num: str) -> Optional[Dict]:
    """
    Ищет рисунок по номеру (аналог oox_fig_lookup).
    """
    for fig in index.get('figures', []):
        if fig.get('num') == figure_num:
            return fig
    return None


def table_lookup(index: Dict, table_num: str) -> Optional[Dict]:
    """
    Ищет таблицу по номеру (аналог oox_tbl_lookup).
    """
    for tbl in index.get('tables', []):
        if tbl.get('num') == table_num:
            return tbl
    return None


# ==================== ЭКСПОРТ ====================

__all__ = [
    'parse_docx',
    'parse_doc',
    'parse_pdf',  # ← Добавили
    'save_upload',
    'build_index',
    'figure_lookup',
    'table_lookup',
]