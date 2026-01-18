# chart_extractor.py
"""
Модуль для извлечения диаграмм (charts) из DOCX файлов.
Диаграммы в DOCX хранятся как XML в word/charts/chartN.xml,
а не как изображения в word/media/.

Этот модуль:
1. Парсит XML диаграмм и извлекает данные (категории, значения)
2. Рендерит диаграммы в PNG с помощью matplotlib
3. Сохраняет изображения и возвращает структуру для индексации
"""

import os
import re
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from xml.etree import ElementTree as ET

# Попытка импорта matplotlib для рендеринга
try:
    import matplotlib
    matplotlib.use('Agg')  # Без GUI
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False
    print("WARNING: matplotlib не установлен. Диаграммы не будут рендериться в PNG.")


# Namespaces для парсинга OOXML
NAMESPACES = {
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
}


def extract_charts_from_docx(docx_path: str, output_dir: str) -> List[Dict[str, Any]]:
    """
    Извлекает все диаграммы из DOCX файла.
    
    Args:
        docx_path: Путь к DOCX файлу
        output_dir: Директория для сохранения PNG изображений
        
    Returns:
        Список словарей с информацией о диаграммах:
        [
            {
                'id': 1,
                'num': '1',
                'caption': 'Рисунок 1 - ...',
                'image_path': '/path/to/chart1.png',
                'kind': 'pie',  # pie, bar, line, etc.
                'chart_data': {
                    'categories': ['A', 'B', 'C'],
                    'values': [10, 20, 30],
                    'series': [...]
                }
            },
            ...
        ]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = []
    
    try:
        with zipfile.ZipFile(docx_path, 'r') as zf:
            # Находим все chart*.xml файлы
            chart_files = [f for f in zf.namelist() if f.startswith('word/charts/chart') and f.endswith('.xml')]
            chart_files.sort(key=lambda x: _extract_chart_number(x))
            
            print(f"DEBUG: Найдено {len(chart_files)} диаграмм в DOCX")
            
            # Читаем document.xml для поиска подписей
            doc_xml = None
            if 'word/document.xml' in zf.namelist():
                doc_xml = zf.read('word/document.xml').decode('utf-8')
            
            for idx, chart_file in enumerate(chart_files, start=1):
                try:
                    chart_xml = zf.read(chart_file).decode('utf-8')
                    chart_data = _parse_chart_xml(chart_xml)
                    
                    if not chart_data:
                        print(f"DEBUG: Не удалось распарсить {chart_file}")
                        continue
                    
                    # Ищем подпись рисунка
                    caption = _find_figure_caption(doc_xml, idx) if doc_xml else None
                    
                    # Генерируем PNG
                    png_path = None
                    if MATPLOTLIB_OK:
                        png_name = f"chart_{idx}_{hashlib.md5(chart_xml.encode()).hexdigest()[:8]}.png"
                        png_path = output_dir / png_name
                        _render_chart_to_png(chart_data, str(png_path))
                        print(f"DEBUG: Сохранена диаграмма #{idx}: {png_path}")
                    
                    figures.append({
                        'id': idx,
                        'num': str(idx),
                        'caption': caption or f'Рисунок {idx}',
                        'image_path': str(png_path) if png_path and png_path.exists() else None,
                        'kind': chart_data.get('type', 'chart'),
                        'chart_data': chart_data,
                        'original_name': chart_file,
                    })
                    
                except Exception as e:
                    print(f"DEBUG: Ошибка обработки {chart_file}: {e}")
                    continue
                    
    except Exception as e:
        print(f"ERROR: Не удалось открыть DOCX: {e}")
        
    return figures


def _extract_chart_number(filename: str) -> int:
    """Извлекает номер из chart1.xml -> 1"""
    match = re.search(r'chart(\d+)\.xml', filename)
    return int(match.group(1)) if match else 0


def _parse_chart_xml(xml_content: str) -> Optional[Dict[str, Any]]:
    """
    Парсит XML диаграммы и извлекает данные.
    
    Returns:
        {
            'type': 'pie' | 'bar' | 'line' | 'area' | ...,
            'title': 'Заголовок',
            'categories': ['Кат1', 'Кат2', ...],
            'series': [
                {'name': 'Серия1', 'values': [1, 2, 3]},
                ...
            ],
            'format': '0%' | '#,##0' | ...
        }
    """
    try:
        # Убираем BOM и лишние пробелы
        xml_content = xml_content.strip()
        if xml_content.startswith('\ufeff'):
            xml_content = xml_content[1:]
            
        root = ET.fromstring(xml_content)
        
        result = {
            'type': 'unknown',
            'title': None,
            'categories': [],
            'series': [],
            'format': None,
        }
        
        # Определяем тип диаграммы
        chart_types = {
            'pieChart': 'pie',
            'pie3DChart': 'pie',
            'barChart': 'bar',
            'bar3DChart': 'bar',
            'lineChart': 'line',
            'line3DChart': 'line',
            'areaChart': 'area',
            'area3DChart': 'area',
            'doughnutChart': 'doughnut',
            'radarChart': 'radar',
            'scatterChart': 'scatter',
        }
        
        for chart_type, type_name in chart_types.items():
            if root.find(f'.//c:{chart_type}', NAMESPACES) is not None:
                result['type'] = type_name
                break
        
        # Извлекаем заголовок
        title_el = root.find('.//c:title//a:t', NAMESPACES)
        if title_el is not None and title_el.text:
            result['title'] = title_el.text.strip()
        
        # Извлекаем серии данных
        for ser in root.findall('.//c:ser', NAMESPACES):
            series_data = {'name': None, 'values': [], 'categories': []}
            
            # Имя серии
            ser_name = ser.find('.//c:tx//c:v', NAMESPACES)
            if ser_name is not None and ser_name.text:
                series_data['name'] = ser_name.text.strip()
            
            # Категории (обычно одинаковые для всех серий)
            for pt in ser.findall('.//c:cat//c:pt', NAMESPACES):
                v = pt.find('c:v', NAMESPACES)
                if v is not None and v.text:
                    cat_text = v.text.strip()
                    series_data['categories'].append(cat_text)
                    if cat_text not in result['categories']:
                        result['categories'].append(cat_text)
            
            # Значения
            for pt in ser.findall('.//c:val//c:pt', NAMESPACES):
                v = pt.find('c:v', NAMESPACES)
                if v is not None and v.text:
                    try:
                        val = float(v.text.strip())
                        series_data['values'].append(val)
                    except ValueError:
                        series_data['values'].append(0)
            
            # Формат чисел
            fmt = ser.find('.//c:val//c:formatCode', NAMESPACES)
            if fmt is not None and fmt.text:
                result['format'] = fmt.text.strip()
            
            if series_data['values']:
                result['series'].append(series_data)
        
        return result if result['series'] else None
        
    except Exception as e:
        print(f"DEBUG: Ошибка парсинга chart XML: {e}")
        return None


def _render_chart_to_png(chart_data: Dict[str, Any], output_path: str) -> bool:
    """
    Рендерит диаграмму в PNG файл с помощью matplotlib.
    """
    if not MATPLOTLIB_OK:
        return False
        
    try:
        plt.figure(figsize=(8, 6))
        plt.rcParams['font.family'] = 'DejaVu Sans'  # Поддержка кириллицы
        
        chart_type = chart_data.get('type', 'bar')
        categories = chart_data.get('categories', [])
        series = chart_data.get('series', [])
        fmt = chart_data.get('format', '')
        
        if not series or not series[0].get('values'):
            plt.close()
            return False
        
        values = series[0]['values']
        
        # Убеждаемся, что categories и values одной длины
        if len(categories) < len(values):
            categories.extend([f'Кат{i+1}' for i in range(len(categories), len(values))])
        elif len(values) < len(categories):
            values.extend([0] * (len(categories) - len(values)))
        
        # Рендерим в зависимости от типа
        if chart_type == 'pie':
            # Для круговой диаграммы
            colors = plt.cm.Set3(range(len(values)))
            
            # Форматируем подписи
            def autopct_format(pct):
                if '0%' in (fmt or ''):
                    return f'{pct:.0f}%'
                return f'{pct:.1f}%'
            
            plt.pie(values, labels=categories, autopct=autopct_format, colors=colors, startangle=90)
            plt.axis('equal')
            
        elif chart_type in ('bar', 'column'):
            # Столбчатая диаграмма
            x = range(len(categories))
            bars = plt.bar(x, values, color=plt.cm.Set2(range(len(values))))
            plt.xticks(x, categories, rotation=45, ha='right')
            
            # Добавляем значения над столбцами
            for bar, val in zip(bars, values):
                height = bar.get_height()
                label = f'{val:.0%}' if '0%' in (fmt or '') else f'{val:.1f}'
                plt.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
                           
        elif chart_type == 'line':
            # Линейный график
            plt.plot(categories, values, marker='o', linewidth=2, markersize=8)
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
        else:
            # Дефолт: столбчатая
            x = range(len(categories))
            plt.bar(x, values, color=plt.cm.Set2(range(len(values))))
            plt.xticks(x, categories, rotation=45, ha='right')
        
        # Заголовок
        if chart_data.get('title'):
            plt.title(chart_data['title'], fontsize=14)
        
        # Легенда для серий
        if len(series) > 1:
            plt.legend([s.get('name', f'Серия {i+1}') for i, s in enumerate(series)])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return os.path.exists(output_path)
        
    except Exception as e:
        print(f"DEBUG: Ошибка рендеринга диаграммы: {e}")
        plt.close()
        return False


def _find_figure_caption(doc_xml: str, figure_num: int) -> Optional[str]:
    """
    Ищет подпись рисунка в document.xml.
    Паттерны: "Рисунок 1 - Описание", "Рис. 1. Описание"
    """
    if not doc_xml:
        return None
        
    patterns = [
        rf'[Рр]исунок\s*{figure_num}\s*[-–—\.]\s*([^<\n]+)',
        rf'[Рр]ис\.\s*{figure_num}\s*[-–—\.]\s*([^<\n]+)',
        rf'Figure\s*{figure_num}\s*[-–—\.]\s*([^<\n]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, doc_xml)
        if match:
            # Очищаем от XML-тегов
            caption = re.sub(r'<[^>]+>', '', match.group(0))
            return caption.strip()
    
    return None


def get_chart_data_as_text(chart_data: Dict[str, Any]) -> str:
    """
    Преобразует данные диаграммы в текстовое описание для LLM.
    """
    if not chart_data:
        return ""
    
    lines = []
    
    chart_type = chart_data.get('type', 'диаграмма')
    type_names = {
        'pie': 'Круговая диаграмма',
        'bar': 'Столбчатая диаграмма',
        'line': 'Линейный график',
        'area': 'Диаграмма с областями',
        'doughnut': 'Кольцевая диаграмма',
    }
    lines.append(f"Тип: {type_names.get(chart_type, chart_type)}")
    
    if chart_data.get('title'):
        lines.append(f"Заголовок: {chart_data['title']}")
    
    fmt = chart_data.get('format', '')
    is_percent = '0%' in fmt or '%' in fmt
    
    lines.append("\nДанные диаграммы:")
    
    for series in chart_data.get('series', []):
        if series.get('name'):
            lines.append(f"\nСерия: {series['name']}")
        
        categories = series.get('categories') or chart_data.get('categories', [])
        values = series.get('values', [])
        
        for cat, val in zip(categories, values):
            if is_percent:
                lines.append(f"  - {cat}: {val*100:.1f}%")
            else:
                lines.append(f"  - {cat}: {val}")
    
    return '\n'.join(lines)


# ==================== ТЕСТ ====================

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python chart_extractor.py <path_to_docx>")
        sys.exit(1)
    
    docx_path = sys.argv[1]
    output_dir = './extracted_charts'
    
    print(f"Извлекаю диаграммы из: {docx_path}")
    
    figures = extract_charts_from_docx(docx_path, output_dir)
    
    print(f"\nНайдено диаграмм: {len(figures)}")
    
    for fig in figures:
        print(f"\n--- Рисунок {fig['num']} ---")
        print(f"Тип: {fig['kind']}")
        print(f"Подпись: {fig['caption']}")
        print(f"Путь: {fig['image_path']}")
        
        if fig.get('chart_data'):
            print("\nДанные:")
            print(get_chart_data_as_text(fig['chart_data']))