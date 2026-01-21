# app/unified_pipeline.py
"""
Единая точка входа для ВСЕХ типов вопросов.
Решает проблему дублирования запросов к LLM API.

БЫЛО:
- answer_builder.py делает запрос
- document_calc_agent.py делает запрос  
- document_semantic_planner.py делает запрос
= 3 запроса на один вопрос!

СТАЛО:
- unified_pipeline.py делает ОДИН запрос
= 1 запрос на один вопрос!
"""

from __future__ import annotations
import re
import json
import asyncio
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache

logger = logging.getLogger(__name__)

# Импорты
try:
    from .config import Cfg
    from .retrieval import (
        retrieve,
        build_context,
        retrieve_coverage,
        build_context_coverage,
        get_table_context_for_numbers,
        get_figure_context_for_numbers,
        # НОВОЕ: для мультимодальности
        build_multimodal_context,
        get_figure_record_with_image,
    )
    from .polza_client import (
        chat_with_gpt, 
        chat_with_gpt_stream,
        # НОВОЕ: для мультимодальности
        chat_with_gpt_multimodal,
        chat_with_gpt_stream_multimodal,
    )
    from .db import get_user_active_doc
    from .figures import analyze_figure_with_vision
except Exception as e:
    logger.warning(f"Импорты не загрузились (возможно, это тест): {e}")
    # Заглушки для тестирования
    def retrieve(*args, **kwargs): return []
    def build_context(*args, **kwargs): return ""
    def chat_with_gpt(*args, **kwargs): return "Заглушка"
    async def chat_with_gpt_stream(*args, **kwargs): yield "Заглушка"


# ==================== ОПРЕДЕЛЕНИЕ ТИПА ВОПРОСА ====================

class QuestionType:
    """Типы вопросов"""
    CALC = "calc"           # Вычислительный вопрос (работа с числами)
    TABLE = "table"         # Вопрос про таблицу
    FIGURE = "figure"       # Вопрос про рисунок/график
    MIXED = "mixed"         # Вопрос про таблицу И рисунок (сравнение)
    SECTION = "section"     # Вопрос про конкретный раздел
    SEMANTIC = "semantic"   # Обычный смысловой вопрос
    GOST = "gost"          # Вопрос про оформление/ГОСТ


def detect_question_type(question: str) -> str:
    """
    Определяет тип вопроса по ключевым словам.
    """
    q_lower = question.lower()
    
    # 1. ВЫЧИСЛИТЕЛЬНЫЕ вопросы
    calc_keywords = [
        'сколько', 'посчитай', 'вычисли', 'сумма', 'среднее', 
        'процент', 'доля', 'количество', 'всего', 'итого'
    ]
    if any(kw in q_lower for kw in calc_keywords):
        return QuestionType.CALC
    
    # 2. СМЕШАННЫЕ вопросы (таблица + рисунок) — проверяем ПЕРВЫМИ!
    has_table = any(w in q_lower for w in ['таблиц', 'табл.', 'table'])
    has_figure = any(kw in q_lower for kw in ['рис', 'график', 'диаграмм', 'figure', 'chart'])
    
    if has_table and has_figure:
        return QuestionType.MIXED
    
    # 3. ТАБЛИЦЫ
    if has_table:
        return QuestionType.TABLE
    
    # 4. РИСУНКИ/ГРАФИКИ
    if has_figure:
        return QuestionType.FIGURE
    
    # 4. РАЗДЕЛЫ
    section_keywords = ['глав', 'раздел', 'параграф', 'пункт', 'введение', 'заключение']
    if any(kw in q_lower for kw in section_keywords):
        return QuestionType.SECTION
    
    # 5. ГОСТ/ОФОРМЛЕНИЕ
    gost_keywords = ['гост', 'оформлени', 'шрифт', 'межстроч', 'поля', 'кегл']
    if any(kw in q_lower for kw in gost_keywords):
        return QuestionType.GOST
    
    # 6. По умолчанию - семантический вопрос
    return QuestionType.SEMANTIC


def extract_numbers_from_question(question: str, entity_type: str = "figure") -> List[str]:
    """
    Извлекает номера (рисунков/таблиц/разделов) из вопроса.
    Поддерживает номера с точкой: 1.1, 2.3, 3.1
    
    entity_type: "figure", "table", "section"
    """
    q_lower = question.lower()
    numbers = []
    
    if entity_type == "figure":
        # Паттерны для рисунков во ВСЕХ падежах и с номерами типа 1.1
        patterns = [
            r'рисун[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',  # рисунок 1.1, рисунке 2.3
            r'рис\.?\s*(?:№\s*)?(\d+(?:\.\d+)?)',         # рис. 1.1, рис 2
            r'график[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',  # график 1.1
            r'диаграмм[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',# диаграмма 2.1
            r'figure\s*(?:№\s*)?(\d+(?:\.\d+)?)',         # figure 1.1
            r'fig\.?\s*(?:№\s*)?(\d+(?:\.\d+)?)',         # fig. 1.1
        ]
    elif entity_type == "table":
        # Паттерны для таблиц с номерами типа 2.1
        patterns = [
            r'таблиц[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',  # таблица 2.1, таблице 3
            r'табл\.?\s*(?:№\s*)?(\d+(?:\.\d+)?)',        # табл. 2.1
            r'table\s*(?:№\s*)?(\d+(?:\.\d+)?)',          # table 2.1
        ]
    else:  # section
        patterns = [
            r'глав[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',
            r'раздел[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',
            r'пункт[а-яё]*\s*(?:№\s*)?(\d+(?:\.\d+)?)',
        ]
    
    for pattern in patterns:
        matches = re.findall(pattern, q_lower)
        numbers.extend(matches)
    
    # Fallback: ищем номер после предлога
    if not numbers:
        fallback_patterns = {
            "figure": r'(?:на|про|в|о|об)\s+рис\w*\s+(\d+(?:\.\d+)?)',
            "table": r'(?:в|из|на|про)\s+табл\w*\s+(\d+(?:\.\d+)?)',
        }
        if entity_type in fallback_patterns:
            fallback_match = re.search(fallback_patterns[entity_type], q_lower)
            if fallback_match:
                numbers.append(fallback_match.group(1))
    
    print(f"[DEBUG] extract_numbers('{question[:40]}...', '{entity_type}') -> {numbers}")
    
    return list(set(numbers))


async def _search_figure_in_text(doc_id: int, figure_num: str) -> str:
    """
    Ищет текстовое описание рисунка в документе.
    Используется когда:
    1. Vision API не смог проанализировать изображение
    2. Изображение вообще не найдено (рисунок — это текстовая схема/таблица)
    3. Vision API показывает не тот рисунок
    
    Стратегия: ищем подпись рисунка и возвращаем текст ПОСЛЕ неё,
    а также ищем раздел с соответствующим номером (например 1.2 → раздел 1.2).
    """
    try:
        from .db import get_conn
        con = get_conn()
        cur = con.cursor()
        
        result_text = ""
        
        # Стратегия 1: Ищем раздел с номером рисунка (1.2 → раздел 1.2)
        # Это работает для случаев когда рисунок описывает содержимое раздела
        section_num = figure_num  # например "1.2"
        
        cur.execute("""
            SELECT text FROM chunks 
            WHERE doc_id = ? AND (
                section_path LIKE ? OR 
                text LIKE ?
            )
            ORDER BY id ASC
            LIMIT 20
        """, (doc_id, f'%{section_num}%', f'%{section_num}%'))
        
        section_chunks = cur.fetchall()
        for chunk in section_chunks:
            chunk_text = chunk['text']
            # Фильтруем только содержательные чанки
            if len(chunk_text) > 50 and 'Рис.' not in chunk_text[:20]:
                result_text += chunk_text + "\n\n"
        
        # Стратегия 2: Ищем подпись рисунка и берём контекст после неё
        if len(result_text) < 500:  # Если мало нашли
            search_patterns = [
                f'%Рис%{figure_num}%',
                f'%рис%{figure_num}%',
            ]
            
            for pattern in search_patterns:
                cur.execute("""
                    SELECT id, text FROM chunks 
                    WHERE doc_id = ? AND text LIKE ?
                    ORDER BY id ASC
                    LIMIT 1
                """, (doc_id, pattern))
                
                row = cur.fetchone()
                if row:
                    caption_chunk_id = row['id']
                    
                    # Берём следующие 15 чанков (больше контекста!)
                    cur.execute("""
                        SELECT text FROM chunks 
                        WHERE doc_id = ? AND id > ?
                        ORDER BY id ASC
                        LIMIT 15
                    """, (doc_id, caption_chunk_id))
                    
                    following_chunks = cur.fetchall()
                    for chunk in following_chunks:
                        chunk_text = chunk['text']
                        if len(chunk_text) > 30 and chunk_text not in result_text:
                            result_text += chunk_text + "\n\n"
                    
                    break
        
        # Стратегия 3: Ищем ключевые слова связанные с темой рисунка
        # Например для "Методы анализа" ищем упоминания методов
        if 'метод' in figure_num.lower() or len(result_text) < 500:
            cur.execute("""
                SELECT text FROM chunks 
                WHERE doc_id = ? AND (
                    text LIKE '%метод%анализ%' OR
                    text LIKE '%ABC%анализ%' OR
                    text LIKE '%вертикальн%анализ%' OR
                    text LIKE '%горизонтальн%анализ%' OR
                    text LIKE '%структурн%анализ%'
                )
                ORDER BY id ASC
                LIMIT 10
            """, (doc_id,))
            
            method_chunks = cur.fetchall()
            for chunk in method_chunks:
                chunk_text = chunk['text']
                if len(chunk_text) > 50 and chunk_text not in result_text:
                    result_text += chunk_text + "\n\n"
        
        con.close()
        
        # Ограничиваем размер чтобы не переполнить контекст
        if len(result_text) > 8000:
            result_text = result_text[:8000] + "\n\n[... текст сокращён ...]"
        
        if result_text:
            logger.info(f"Найдено текстовое описание рисунка {figure_num}: {len(result_text)} символов")
        
        return result_text
        
    except Exception as e:
        logger.warning(f"Ошибка поиска текстового описания рисунка: {e}")
        return ""


# ==================== ПОЛУЧЕНИЕ КОНТЕКСТА ====================

async def get_context_for_question(
    doc_id: int,
    owner_id: int,
    question: str,
    question_type: str,
) -> str:
    """
    Получает релевантный контекст в зависимости от типа вопроса.
    ОДИН вызов для каждого вопроса (не множественные!).
    """
    
    # 1. Для вопросов про таблицы ИЛИ вычислительных с упоминанием таблицы
    # Объединяем логику, чтобы "В таблице 2.1..." всегда находило данные
    is_table_question = (
        question_type == QuestionType.TABLE or 
        (question_type == QuestionType.CALC and any(w in question.lower() for w in ['таблиц', 'табл']))
    )
    
    if is_table_question:
        table_nums = extract_numbers_from_question(question, "table")
        print(f"[DEBUG TABLE] is_table_question=True, table_nums={table_nums}, doc_id={doc_id}")
        
        if table_nums:
            # Получаем базовый контекст
            text_context = await asyncio.to_thread(
                get_table_context_for_numbers,
                owner_id, doc_id, table_nums
            )
            
            # Пытаемся найти точные данные таблицы в чанках
            table_data_text = ""
            try:
                from .db import get_conn
                con = get_conn()
                cur = con.cursor()
                
                for table_num in table_nums:
                    search_patterns = [
                        f'%Таблица {table_num}%',
                        f'%таблица {table_num}%',
                        f'%Таблица{table_num}%',
                    ]
                    
                    for pattern in search_patterns:
                        cur.execute("""
                            SELECT text, element_type, section_path FROM chunks 
                            WHERE doc_id = ? AND (
                                element_type = 'table' OR
                                section_path LIKE ? OR 
                                text LIKE ?
                            )
                            ORDER BY 
                                CASE WHEN element_type = 'table' THEN 0 ELSE 1 END,
                                LENGTH(text) DESC
                            LIMIT 5
                        """, (doc_id, pattern, pattern))
                        
                        rows = cur.fetchall()
                        print(f"[DEBUG TABLE] Поиск '{pattern}': найдено {len(rows)} чанков")
                        
                        if rows:
                            for row in rows:
                                chunk_text = row['text']
                                print(f"[DEBUG TABLE]   element_type={row['element_type']}, len={len(chunk_text)}")
                                if len(chunk_text) > 100:
                                    table_data_text += chunk_text + "\n\n"
                            
                            if table_data_text:
                                print(f"[DEBUG TABLE] Итого данных таблицы {table_num}: {len(table_data_text)} символов")
                                break
                    
                    if table_data_text:
                        break
                
                con.close()
            except Exception as e:
                print(f"[DEBUG TABLE] Ошибка поиска таблицы: {e}")
            
            if table_data_text:
                combined = f"""ДАННЫЕ ТАБЛИЦЫ:
{table_data_text}

КОНТЕКСТ ИЗ ДОКУМЕНТА:
{text_context}

ВАЖНО: Используй данные из таблицы для точного ответа!"""
                return combined
            
            return text_context
    
    # 2. Для вычислительных вопросов БЕЗ явного указания таблицы
    if question_type == QuestionType.CALC:
        table_nums = extract_numbers_from_question(question, "table")
        if table_nums:
            return await asyncio.to_thread(
                get_table_context_for_numbers,
                owner_id, doc_id, table_nums
            )
    
    # 3. Для СМЕШАННЫХ вопросов (таблица + рисунок) — собираем оба контекста
    if question_type == QuestionType.MIXED:
        table_nums = extract_numbers_from_question(question, "table")
        fig_nums = extract_numbers_from_question(question, "figure")
        
        combined_parts = []
        
        # Получаем данные таблицы (копируем логику из TABLE)
        if table_nums:
            table_data_text = ""
            try:
                from .db import get_conn
                con = get_conn()
                cur = con.cursor()
                
                for table_num in table_nums:
                    search_patterns = [
                        f'%Таблица {table_num}%',
                        f'%таблица {table_num}%',
                        f'%Таблица{table_num}%',
                    ]
                    
                    for pattern in search_patterns:
                        cur.execute("""
                            SELECT text, element_type, section_path FROM chunks 
                            WHERE doc_id = ? AND (
                                element_type = 'table' OR
                                section_path LIKE ? OR 
                                text LIKE ?
                            )
                            ORDER BY 
                                CASE WHEN element_type = 'table' THEN 0 ELSE 1 END,
                                LENGTH(text) DESC
                            LIMIT 5
                        """, (doc_id, pattern, pattern))
                        
                        rows = cur.fetchall()
                        print(f"[DEBUG MIXED TABLE] Поиск '{pattern}': найдено {len(rows)} чанков")
                        
                        if rows:
                            for row in rows:
                                chunk_text = row['text']
                                print(f"[DEBUG MIXED TABLE]   element_type={row['element_type']}, len={len(chunk_text)}")
                                if len(chunk_text) > 100:
                                    table_data_text += chunk_text + "\n\n"
                            
                            if table_data_text:
                                print(f"[DEBUG MIXED TABLE] Итого данных таблицы {table_num}: {len(table_data_text)} символов")
                                break
                    
                    if table_data_text:
                        break
                
                con.close()
                
                if table_data_text:
                    combined_parts.append(f"ДАННЫЕ ТАБЛИЦЫ {table_nums[0]}:\n{table_data_text}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки таблицы для MIXED: {e}")
                print(f"[DEBUG MIXED TABLE] Ошибка: {e}")
        
        # Получаем данные рисунка через Vision API
        if fig_nums:
            try:
                vision_result = await asyncio.to_thread(
                    analyze_figure_with_vision,
                    owner_id, doc_id, fig_nums[0], question
                )
                
                if vision_result:
                    combined_parts.append(f"ВИЗУАЛЬНЫЙ АНАЛИЗ РИСУНКА {fig_nums[0]} (Vision API):\n{vision_result}")
            except Exception as e:
                logger.warning(f"Vision API failed для MIXED: {e}")
        
        # Получаем общий текстовый контекст
        text_context = await asyncio.to_thread(
            retrieve, owner_id, doc_id, question, top_k=5
        )
        text_context_str = build_context(text_context) if text_context else ""
        
        if combined_parts:
            combined_context = "\n\n---\n\n".join(combined_parts)
            if text_context_str:
                combined_context += f"\n\nДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:\n{text_context_str}"
            return combined_context
        
        return text_context_str
    
    # 4. Для вопросов про рисунки - сначала chart_data, потом Vision API
    if question_type == QuestionType.FIGURE:
        fig_nums = extract_numbers_from_question(question, "figure")
        if fig_nums:
            # Получаем текстовый контекст (подписи, упоминания)
            text_context = await asyncio.to_thread(
                get_figure_context_for_numbers,
                owner_id, doc_id, fig_nums
            )
            
            # НОВОЕ: Пытаемся получить chart_data из БД (точные данные диаграммы!)
            chart_data_text = ""
            try:
                from .db import get_figures_for_doc
                figures = await asyncio.to_thread(get_figures_for_doc, doc_id)
                
                logger.info(f"Найдено {len(figures)} рисунков в БД для doc_id={doc_id}")
                
                for fig in figures:
                    fig_label = str(fig.get('label') or fig.get('figure_label') or '')
                    fig_num_from_db = fig_label.strip()
                    
                    # Проверяем, совпадает ли номер
                    for requested_num in fig_nums:
                        if requested_num in fig_num_from_db or fig_num_from_db.endswith(requested_num) or fig_num_from_db == requested_num:
                            logger.info(f"Проверяем рисунок {fig_label}")
                            
                            # chart_data хранится в ATTRS, а не в data!
                            attrs = fig.get('attrs') or {}
                            if isinstance(attrs, str):
                                try:
                                    attrs = json.loads(attrs)
                                except:
                                    attrs = {}
                            
                            chart_data = attrs.get('chart_data')
                            logger.info(f"chart_data в attrs: {bool(chart_data)}")
                            
                            if chart_data:
                                try:
                                    from .chart_extractor import get_chart_data_as_text
                                    chart_data_text = get_chart_data_as_text(chart_data)
                                    logger.info(f"Извлечены данные диаграммы: {len(chart_data_text)} символов")
                                except ImportError:
                                    chart_data_text = _format_chart_data_simple(chart_data)
                                break
                    if chart_data_text:
                        break
                        
            except Exception as e:
                logger.warning(f"Не удалось получить chart_data из БД: {e}")
            
            # Если есть данные диаграммы - используем их (НЕ нужен Vision API!)
            if chart_data_text:
                combined_context = f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ТОЧНЫЕ ДАННЫЕ ДИАГРАММЫ (извлечены из DOCX):
{chart_data_text}

ВАЖНО: Используй ТОЛЬКО эти точные данные при ответе!
"""
                logger.info("Используем chart_data для ответа (Vision API не нужен)")
                return combined_context
            
            # Fallback: Vision API (если нет chart_data)
            try:
                vision_result = await asyncio.to_thread(
                    analyze_figure_with_vision,
                    owner_id, doc_id, fig_nums[0], question
                )
                
                # Проверяем качество ответа Vision API
                vision_is_valid = True
                if vision_result:
                    vision_lower = vision_result.lower()
                    # Признаки ПОЛНОГО провала (изображение не загрузилось или пустое)
                    # Эти фразы должны быть в НАЧАЛЕ ответа, чтобы не ловить их в середине текста
                    critical_failures = [
                        'изображение полностью чёрное',
                        'изображение пустое',
                        'не удалось загрузить',
                        'изображение не загружено',
                        'контент отсутствует',
                        'изображение не отображается',
                    ]
                    # Проверяем только если эти фразы в первых 200 символах (в начале ответа)
                    vision_start = vision_lower[:200]
                    if any(sign in vision_start for sign in critical_failures):
                        logger.warning(f"Vision API: изображение не загрузилось, используем текстовый контекст")
                        vision_is_valid = False
                
                if vision_result and vision_is_valid:
                    # Дополнительно: ищем текстовое описание рисунка
                    # (полезно когда Vision видит не тот рисунок или рисунок — текстовая схема)
                    figure_text_context = await _search_figure_in_text(doc_id, fig_nums[0])
                    
                    combined_context = f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ВИЗУАЛЬНЫЙ АНАЛИЗ РИСУНКА (Vision API):
{vision_result}
"""
                    # Добавляем текстовое описание если оно существенное
                    if figure_text_context and len(figure_text_context) > 200:
                        combined_context += f"""

ТЕКСТОВОЕ ОПИСАНИЕ РИСУНКА {fig_nums[0]} (из документа):
{figure_text_context}

КРИТИЧЕСКИ ВАЖНО: 
1. Визуальный анализ показывает ДРУГОЕ изображение (не соответствует подписи рисунка).
2. Используй ТЕКСТОВОЕ ОПИСАНИЕ как ОСНОВНОЙ и ЕДИНСТВЕННЫЙ источник для ответа!
3. НЕ описывай визуальные элементы из Vision API — они относятся к другому рисунку.
4. Отвечай на вопрос ТОЛЬКО на основе текстового описания выше.
"""
                    return combined_context
                else:
                    # Vision не сработал — ищем описание рисунка в тексте
                    logger.info(f"Vision API не дал результата, ищем описание рисунка {fig_nums[0]} в тексте")
                    figure_text_context = await _search_figure_in_text(doc_id, fig_nums[0])
                    if figure_text_context:
                        combined_context = f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ТЕКСТОВОЕ ОПИСАНИЕ РИСУНКА {fig_nums[0]} (из документа):
{figure_text_context}

ВАЖНО: Визуальный анализ рисунка недоступен. Отвечай на основе текстового описания из документа.
"""
                        return combined_context
                    
            except Exception as e:
                logger.warning(f"Vision API failed: {e}, using text context only")
            
            # Последний fallback: только текстовый контекст
            return text_context
        
        else:
            # Номера рисунков не указаны — проверяем, спрашивают ли про ВСЕ рисунки
            all_figures_keywords = [
                'все рисунки', 'всех рисунков', 'всем рисункам', 'рисунки в работе', 
                'какие рисунки', 'сколько рисунков', 'перечисли рисунки', 
                'список рисунков', 'про рисунки', 'о рисунках',
                'на рисунках', 'рисунках в', 'значения на рисунках',
                'данные на рисунках', 'диаграммы в работе', 'все диаграммы',
                'графики в работе', 'все графики', 'опиши рисунки',
                'расскажи о диаграммах', 'расскажи про диаграммы',
            ]
            
            q_lower = question.lower()
            is_all_figures_question = any(kw in q_lower for kw in all_figures_keywords)
            
            if is_all_figures_question:
                logger.info("Вопрос про ВСЕ рисунки — собираем информацию обо всех")
                
                try:
                    from .db import get_figures_for_doc
                    figures = await asyncio.to_thread(get_figures_for_doc, doc_id)
                    
                    if figures:
                        all_figures_text = f"В документе найдено {len(figures)} рисунков:\n\n"
                        
                        for i, fig in enumerate(figures, 1):
                            fig_label = str(fig.get('label') or fig.get('figure_label') or f'#{i}')
                            caption = fig.get('caption') or 'без подписи'
                            kind = fig.get('kind') or 'image'
                            
                            all_figures_text += f"### Рисунок {fig_label}: {caption}\n"
                            all_figures_text += f"Тип: {kind}\n"
                            
                            # Добавляем ПОЛНЫЕ данные если есть chart_data
                            attrs = fig.get('attrs') or {}
                            if isinstance(attrs, str):
                                try:
                                    attrs = json.loads(attrs)
                                except:
                                    attrs = {}
                            
                            chart_data = attrs.get('chart_data')
                            if chart_data:
                                # Извлекаем полные данные диаграммы
                                categories = chart_data.get('categories', [])
                                series = chart_data.get('series', [])
                                chart_type = chart_data.get('chart_type', '')
                                
                                if chart_type:
                                    all_figures_text += f"Тип диаграммы: {chart_type}\n"
                                
                                if categories and series:
                                    all_figures_text += "**Данные:**\n"
                                    for s in series:
                                        series_name = s.get('name', 'Значения')
                                        values = s.get('values', [])
                                        all_figures_text += f"  {series_name}:\n"
                                        for j, cat in enumerate(categories):
                                            if j < len(values):
                                                val = values[j]
                                                # Форматируем проценты
                                                if isinstance(val, (int, float)):
                                                    all_figures_text += f"    - {cat}: {val}%\n"
                                                else:
                                                    all_figures_text += f"    - {cat}: {val}\n"
                            
                            all_figures_text += "\n"
                        
                        # Добавляем RAG-контекст для дополнительной информации
                        rag_context = await asyncio.to_thread(
                            retrieve, owner_id, doc_id, question, top_k=5
                        )
                        rag_text = build_context(rag_context) if rag_context else ""
                        
                        return f"""ПОЛНЫЕ ДАННЫЕ ВСЕХ РИСУНКОВ В ДОКУМЕНТЕ:
{all_figures_text}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ДОКУМЕНТА:
{rag_text}

ИНСТРУКЦИЯ: Используй ТОЧНЫЕ ЧИСЛОВЫЕ ДАННЫЕ из списка выше для ответа. Опиши значения, тренды, максимумы и минимумы на основе этих данных."""
                    
                except Exception as e:
                    logger.warning(f"Ошибка получения списка рисунков: {e}")
    
    # 5. Для вопросов про РАЗДЕЛЫ — расширенный поиск
    if question_type == QuestionType.SECTION:
        section_context = ""
        
        # Извлекаем номера из вопроса (поддержка "глава 1", "1.1", "раздел 2")
        q_lower = question.lower()

        chapter_nums = re.findall(r'глав[а-яё]*\s*(?:№\s*)?(\d+)', q_lower)
        # любые уровни: 1.1, 1.1.1, 2.3.4.5
        section_nums = re.findall(r'\b(\d+(?:\.\d+)+)\b', question)
        # поддержка "раздел 2", "пункт 3.1", "подраздел 1.2.3"
        named_nums = re.findall(r'(?:раздел|подраздел|пункт|параграф)\s*(?:№\s*)?(\d+(?:\.\d+)*)', q_lower)

        all_nums = sorted(set(chapter_nums + section_nums + named_nums), key=len, reverse=True)

        # Проверяем ключевые слова для введения
        q_lower = question.lower()
        asks_intro = any(kw in q_lower for kw in ['введени', 'цель', 'объект', 'предмет', 'задач', 'гипотез', 'актуальн'])
        asks_structure = any(kw in q_lower for kw in ['структур', 'содержани', 'оглавлен', 'все главы', 'все подглав', 'план'])
        asks_conclusion = 'заключен' in q_lower or 'вывод' in q_lower
        
        try:
            from .db import get_conn, get_document_ai_meta
            
            # Если спрашивают про введение/цель/задачи — используем сохранённые метаданные
            if asks_intro:
                meta = get_document_ai_meta(doc_id)
                print(f"[DEBUG META] doc_id={doc_id}, meta={meta}")  # Дебаг
                
                if meta:
                    intro_parts = []
                    if meta.get('relevance'):
                        intro_parts.append(f"АКТУАЛЬНОСТЬ ИССЛЕДОВАНИЯ:\n{meta['relevance']}")
                    if meta.get('object'):
                        intro_parts.append(f"ОБЪЕКТ ИССЛЕДОВАНИЯ: {meta['object']}")
                    if meta.get('subject'):
                        intro_parts.append(f"ПРЕДМЕТ ИССЛЕДОВАНИЯ: {meta['subject']}")
                    if meta.get('goal'):
                        intro_parts.append(f"ЦЕЛЬ ИССЛЕДОВАНИЯ: {meta['goal']}")
                    if meta.get('tasks') and isinstance(meta['tasks'], list):
                        tasks_formatted = "\n".join([f"  {i+1}. {t}" for i, t in enumerate(meta['tasks'])])
                        intro_parts.append(f"ЗАДАЧИ ИССЛЕДОВАНИЯ:\n{tasks_formatted}")
                    if meta.get('hypothesis'):
                        intro_parts.append(f"ГИПОТЕЗА: {meta['hypothesis']}")
                    
                    if intro_parts:
                        section_context = "=== ДАННЫЕ ИЗ ВВЕДЕНИЯ ===\n\n" + "\n\n".join(intro_parts) + "\n\n"
                        print(f"[DEBUG] Добавлен контекст из метаданных: {len(section_context)} символов")
            
            # Если спрашивают про структуру — ищем все заголовки
            if asks_structure or 'подглав' in q_lower:
                con = get_conn()
                cur = con.cursor()
                cur.execute("""
                    SELECT text, section_path, element_type FROM chunks
                    WHERE owner_id = ? AND doc_id = ? AND element_type = 'heading'
                    ORDER BY id ASC
                    LIMIT 200
                """, (owner_id, doc_id))

                
                headings = cur.fetchall()
                con.close()
                
                if headings:
                    heading_texts = [row['text'] for row in headings]
                    section_context += "СТРУКТУРА ДОКУМЕНТА (ЗАГОЛОВКИ):\n\n"
                    section_context += "\n".join(heading_texts) + "\n\n"
            
            # Поиск по конкретным номерам глав
            if all_nums:
                con = get_conn()
                cur = con.cursor()
                
                for num in all_nums:
                    # Расширенный поиск: по тексту И по section_path
                    # Используем более гибкие паттерны
                    
                    # 1) Сначала ищем заголовок главы
                    patterns = []

                    is_chapter_request = bool(re.search(r'\bглава\b', q_lower))

                    if '.' not in str(num):
                        if is_chapter_request:
                            # ЖЁСТКО только глава, чтобы не цеплять списки "1. ..." и пункты
                            patterns += [
                                f'ГЛАВА {num}%',
                                f'Глава {num}%',
                            ]
                        else:
                            # общий случай (раздел/пункт), можно шире
                            patterns += [
                                f'ГЛАВА {num}%',
                                f'Глава {num}%',
                                f'{num} %',
                                f'{num}. %',
                                f'{num}.%',
                                f'%> {num} %',
                            ]


                    # Ищем И по text, И по section_path
                    # --- before execute: guard ---
                    if not patterns:
                        continue  # или: patterns = [f"%{num}%"] как ultra-fallback

                    text_clause = " OR ".join(["text LIKE ?"] * len(patterns))
                    path_clause = " OR ".join(["section_path LIKE ?"] * len(patterns))

                    sql = f"""
                        SELECT text, section_path, element_type
                        FROM chunks
                        WHERE owner_id = ? AND doc_id = ? AND element_type = 'heading'
                        AND ( ({text_clause}) OR ({path_clause}) )
                        ORDER BY id ASC
                        LIMIT 10
                    """
                    cur.execute(sql, (owner_id, doc_id, *patterns, *patterns))

                    
                    headings = cur.fetchall()
                    found_section_paths = []
                    
                    for row in headings:
                        chunk_text = row['text']
                        sp = row['section_path']
                        if sp:
                            found_section_paths.append(sp)

                    
                    # 2) Теперь ищем ВСЁ содержимое этой главы по section_path
                    for sp in found_section_paths:
                        if not sp:
                            continue
                        cur.execute("""
                            SELECT text, element_type FROM chunks
                            WHERE owner_id = ? AND doc_id = ? AND section_path LIKE ?
                            ORDER BY id ASC
                            LIMIT 500
                        """, (owner_id, doc_id, f'{sp}%'))
                        
                        content_rows = cur.fetchall()
                        for crow in content_rows:
                            ctext = crow['text']
                            if ctext not in section_context and len(ctext) > 50:
                                section_context += ctext + "\n\n"
                    
                    # 3) Fallback: поиск подглав (1.1, 1.2, 1.3...)
                    # Fallback, если заголовок главы не найден, но пользователь просит "глава N":
                    # ищем любые подзаголовки/фрагменты вида "N." / "N.x" в тексте и section_path без ограничения 1..9
                    if '.' not in str(num) and not found_section_paths:
                        fallback_patterns = [
                            f'{num}.%',      # "1. ..." или "1.1 ..."
                            f'{num} %',      # "1 Введение"
                        ]
                        cur.execute(f"""
                            SELECT text FROM chunks
                            WHERE owner_id = ? AND doc_id = ? AND (
                                {" OR ".join(["section_path LIKE ?"] * len(fallback_patterns))} OR
                                {" OR ".join(["text LIKE ?"] * len(fallback_patterns))}
                            )
                            ORDER BY id ASC
                            LIMIT 200
                        """, (owner_id, doc_id, *fallback_patterns, *fallback_patterns))

                        for row in cur.fetchall():
                            stext = row['text']
                            if stext not in section_context and len(stext) > 30:
                                section_context += stext + "\n\n"
                
                con.close()
            
            # Для заключения
            if asks_conclusion:
                con = get_conn()
                cur = con.cursor()
                cur.execute("""
                    SELECT text FROM chunks 
                    WHERE doc_id = ? AND (
                        text LIKE '%ЗАКЛЮЧЕНИЕ%' OR 
                        text LIKE '%Заключение%' OR
                        section_path LIKE '%заключен%'
                    )
                    ORDER BY id ASC
                    LIMIT 20
                """, (doc_id,))
                
                rows = cur.fetchall()
                for row in rows:
                    if row['text'] not in section_context:
                        section_context += row['text'] + "\n\n"
                con.close()
            
            if section_context:
                print(f"[DEBUG SECTION] Собран контекст: {len(section_context)} символов")
                
                # Дополняем RAG-поиском
                rag_context = await asyncio.to_thread(
                    retrieve, owner_id, doc_id, question, top_k=5
                )
                rag_text = build_context(rag_context) if rag_context else ""
                
                return f"""ИНФОРМАЦИЯ ПО ЗАПРОСУ:

{section_context}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ:
{rag_text}

ВАЖНО: Отвечай на основе представленных данных. Если информации недостаточно — укажи это."""
            
            else:
                # Fallback: если специфичный поиск не дал результатов
                print(f"[DEBUG SECTION] Контекст пустой, используем RAG fallback")
                # Не возвращаем — пусть код идёт дальше в RAG поиск
                pass
                
        except Exception as e:
            logger.warning(f"Ошибка поиска разделов: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Для остальных - обычный RAG поиск
    # Используем retrieve_coverage для лучшего качества
    try:
        # Async retrieve
        result = await asyncio.to_thread(
            retrieve_coverage,
            owner_id, doc_id, question,
            per_item_k=3,  # По 3 чанка на каждый найденный элемент
            backfill_k=5,  # + 5 дополнительных чанков
        )
        
        if result and result.get('snippets'):
            return build_context_coverage(result['snippets'])
    except Exception as e:
        logger.warning(f"retrieve_coverage failed: {e}, fallback to basic retrieve")
    
    # Fallback: базовый retrieve
    snippets = await asyncio.to_thread(
        retrieve, owner_id, doc_id, question, top_k=10
    )
    return build_context(snippets) if snippets else ""
    

# ==================== ПОСТРОЕНИЕ ПРОМПТА ====================

def build_system_prompt(question_type: str) -> str:
    """
    Создаёт системный промпт в зависимости от типа вопроса.
    """
    
    base = (
        "Ты помощник по анализу выпускных квалификационных работ (ВКР).\n"
        "Отвечай СТРОГО на основе предоставленного контекста из документа.\n"
        "Не используй общие знания, догадки или типовые определения.\n"
        "Если в контексте нет информации для ответа - честно скажи об этом.\n"
        "\n"
        "ВАЖНО: ВСЕГДА завершай ответ полностью. Не обрывай на полуслове.\n"
        "Если ответ получается длинным - сократи его, но ОБЯЗАТЕЛЬНО напиши вывод.\n"
    )
    
    if question_type == QuestionType.CALC:
        return base + (
            "\nТип вопроса: ВЫЧИСЛИТЕЛЬНЫЙ.\n"
            "- Извлекай числа из таблиц и выполняй точные вычисления.\n"
            "- Показывай формулы и промежуточные результаты.\n"
            "- Не придумывай значения, которых нет в таблице.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.TABLE:
        return base + (
            "\nТип вопроса: ТАБЛИЦА.\n"
            "- Анализируй структуру и данные таблицы.\n"
            "- Ссылайся на конкретные строки и столбцы.\n"
            "- Выделяй ключевые значения и тренды.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.FIGURE:
        return base + (
            "\nТип вопроса: РИСУНОК/ГРАФИК.\n"
            "- Описывай визуальные элементы графика.\n"
            "- Указывай тренды, максимумы, минимумы.\n"
            "- Ссылайся на подпись и легенду рисунка.\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
        )
    
    elif question_type == QuestionType.SECTION:
        return base + (
            "\nТип вопроса: РАЗДЕЛ/ГЛАВА/СТРУКТУРА.\n"
            "ПРАВИЛА:\n"
            "- Отвечай на конкретный запрос пользователя (описать главу, перечислить подглавы, раскрыть содержание раздела).\n"
            "- Если пользователь просит 'опиши главу N' — дай краткое содержание главы:\n"
            "  1) тема и назначение главы,\n"
            "  2) ключевые положения и определения,\n"
            "  3) что рассматривается по подпунктам,\n"
            "  4) выводы по главе (если есть в тексте).\n"
            "- Если пользователь просит 'структуру/оглавление/план' — перечисли заголовки и подзаголовки.\n"
            "- Не навязывай пункты 'цель/объект/предмет/задачи/методы', если пользователь об этом не спрашивал.\n"
            "- Если информации в контексте нет — честно скажи: 'В документе не указано'.\n"
            "- В конце: краткий вывод.\n"
        )


    else:  # SEMANTIC или SECTION
        return base + (
            "\nТип вопроса: СМЫСЛОВОЙ.\n"
            "ПРАВИЛА ОТВЕТА:\n"
            "- Отвечай ТОЛЬКО на основе контекста из документа.\n"
            "- Приводи прямые цитаты в кавычках «...».\n"
            "- НЕ используй ссылки [S1], [S2] — они не видны пользователю.\n"
            "- Если информации нет в контексте — честно скажи: 'В документе не указано'.\n"
            "- НЕ выдумывай 'предполагаемые формулировки' — только факты из документа.\n"
            "- Структурируй ответ: пункты с конкретными данными.\n"
            "- В конце — краткий вывод из 1-2 предложений.\n"
        )


# ==================== ГЛАВНЫЙ ПАЙПЛАЙН ====================

class UnifiedPipeline:
    """
    Единый пайплайн для обработки ВСЕХ типов вопросов.
    """
    
    def __init__(self):
        self._cache = {}  # Простой кэш ответов
        self._cache_size = 100  # Максимум элементов в кэше
    
    def _get_cache_key(self, doc_id: int, question: str) -> str:
        """Создаёт ключ для кэша"""
        import hashlib
        q_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"{doc_id}:{q_hash}"
    
    async def answer(
        self,
        owner_id: int,
        doc_id: int,
        question: str,
        *,
        use_cache: bool = True,
        stream: bool = False,
    ) -> str:
        """
        Главная функция: отвечает на вопрос по документу.
        
        Args:
            owner_id: ID пользователя
            doc_id: ID документа
            question: Вопрос пользователя
            use_cache: Использовать кэш
            stream: Стриминговый режим (генератор чанков)
        
        Returns:
            Ответ на вопрос (или генератор для стрима)
        """
        
        # 1. Проверяем кэш
        if use_cache:
            cache_key = self._get_cache_key(doc_id, question)
            if cache_key in self._cache:
                logger.info(f"Cache HIT for: {question[:50]}...")
                return self._cache[cache_key]
        
        # 2. Определяем тип вопроса
        question_type = detect_question_type(question)
        logger.info(f"Question type: {question_type}")
        
        # 3. Получаем контекст ОДИН раз
        context = await get_context_for_question(
            doc_id, owner_id, question, question_type
        )
        
        # Проверка: если контекст пустой или слишком короткий
        if not context or len(context) < 100:
            return (
                "В документе не найдено информации для ответа на этот вопрос.\n"
                "Попробуйте:\n"
                "- Уточнить формулировку\n"
                "- Указать номер главы/раздела\n"
                "- Спросить про другой аспект работы"
            )
        
        # 4. Строим системный промпт
        system_prompt = build_system_prompt(question_type)
        
        # 5. Вызываем LLM ОДИН раз
        if stream and chat_with_gpt_stream:
            # Стриминговый режим
            return self._answer_stream(system_prompt, question, context)
        else:
            # Обычный режим
            answer = await self._answer_sync(system_prompt, question, context)
            
            # 6. Кэшируем результат
            if use_cache:
                self._update_cache(cache_key, answer)
            
            return answer
    
    async def _answer_sync(self, system: str, question: str, context) -> str:
        """Синхронный вызов LLM (поддерживает мультимодальность)"""
        try:
            # Проверяем, мультимодальный ли контекст
            images = []
            text_context = context
            
            if isinstance(context, dict) and context.get("type") == "multimodal":
                # Мультимодальный режим: есть картинки
                text_context = context.get("text", "")
                images = context.get("images", [])
                logger.info(f"Multimodal request with {len(images)} image(s)")
            elif isinstance(context, str):
                # Обычный текстовый режим
                text_context = context
                logger.info("Text-only request")
            else:
                # Странный формат - пытаемся извлечь текст
                text_context = str(context)
            
            # Формируем сообщения
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Контекст из документа:\n{text_context}\n\nВопрос: {question}"}
            ]
            
            # Вызываем API (с картинками или без)
            if images:
                # Мультимодальный запрос
                answer = await asyncio.to_thread(
                    chat_with_gpt_multimodal,
                    messages=messages,
                    images=images,
                    temperature=0.2,  # Ниже температура для точности
                    max_tokens=4000,
                )
            else:
                # Обычный текстовый запрос
                answer = await asyncio.to_thread(
                    chat_with_gpt,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=4000,
                )
            
            return answer
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"Ошибка при обращении к модели: {e}"
    
    async def _answer_stream(self, system: str, question: str, context: str):
        """Стриминговый вызов LLM (генератор)"""
        try:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Контекст:\n{context}\n\nВопрос: {question}"}
            ]
            
            # Генератор чанков
            async for chunk in chat_with_gpt_stream(
                messages=messages,
                temperature=0.3,
                max_tokens=4000,
            ):
                yield chunk
        except Exception as e:
            logger.error(f"LLM stream failed: {e}")
            yield f"Ошибка: {e}"
    
    def _update_cache(self, key: str, value: str):
        """Обновляет кэш (с ограничением размера)"""
        if len(self._cache) >= self._cache_size:
            # Удаляем самый старый элемент (FIFO)
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        
        self._cache[key] = value


# ==================== ГЛОБАЛЬНЫЙ ЭКЗЕМПЛЯР ====================

# Создаём единственный экземпляр для всего приложения
pipeline = UnifiedPipeline()

# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

async def answer_question(
    user_id: int,
    question: str,
    *,
    stream: bool = False,
) -> str:
    """
    Удобная обёртка для использования в bot.py
    
    Пример использования:
        answer = await answer_question(user_id=123, question="Что такое ВКР?")
    """
    # Получаем активный документ пользователя
    doc_id = await asyncio.to_thread(get_user_active_doc, user_id)
    
    if not doc_id:
        return (
            "У вас нет активного документа.\n"
            "Загрузите файл ВКР командой /upload или отправьте его сюда."
        )
    
    # Вызываем unified pipeline
    return await pipeline.answer(
        owner_id=user_id,
        doc_id=doc_id,
        question=question,
        stream=stream,
    )


# ==================== ЭКСПОРТ ====================

__all__ = [
    'UnifiedPipeline',
    'pipeline',
    'answer_question',
    'detect_question_type',
    'QuestionType',
]

def _format_chart_data_simple(chart_data: dict) -> str:
    """Простое форматирование chart_data без chart_extractor"""
    if not chart_data:
        return ""
    
    lines = []
    chart_type = chart_data.get('type', 'диаграмма')
    lines.append(f"Тип: {chart_type}")
    
    if chart_data.get('title'):
        lines.append(f"Заголовок: {chart_data['title']}")
    
    lines.append("\nДанные:")
    
    for series in chart_data.get('series', []):
        if series.get('name'):
            lines.append(f"\nСерия: {series['name']}")
        
        categories = series.get('categories') or chart_data.get('categories', [])
        values = series.get('values', [])
        
        for cat, val in zip(categories, values):
            if isinstance(val, float) and val < 1:
                lines.append(f"  - {cat}: {val*100:.1f}%")
            else:
                lines.append(f"  - {cat}: {val}")
    
    return '\n'.join(lines)