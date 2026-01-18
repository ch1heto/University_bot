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
    )
    from .polza_client import chat_with_gpt, chat_with_gpt_stream
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
    
    # 2. ТАБЛИЦЫ
    if any(w in q_lower for w in ['таблиц', 'табл.', 'table']):
        return QuestionType.TABLE
    
    # 3. РИСУНКИ/ГРАФИКИ
    fig_keywords = ['рис', 'график', 'диаграмм', 'схем', 'figure', 'chart']
    if any(kw in q_lower for kw in fig_keywords):
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
    
    # 3. Для вопросов про рисунки - сначала chart_data, потом Vision API
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
                
                if vision_result:
                    combined_context = f"""
ОПИСАНИЕ ИЗ ДОКУМЕНТА:
{text_context}

ВИЗУАЛЬНЫЙ АНАЛИЗ РИСУНКА (Vision API):
{vision_result}
"""
                    return combined_context
            except Exception as e:
                logger.warning(f"Vision API failed: {e}, using text context only")
            
            # Последний fallback: только текстовый контекст
            return text_context
    
    # 4. Для остальных - обычный RAG поиск
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
    
    elif question_type == QuestionType.GOST:
        return base + (
            "\nТип вопроса: ОФОРМЛЕНИЕ/ГОСТ.\n"
            "- Проверяй соответствие требованиям ГОСТа.\n"
            "- Указывай конкретные нарушения, если они есть.\n"
            "- Давай рекомендации по исправлению.\n"
        )
    
    else:  # SEMANTIC или SECTION
        return base + (
            "\nТип вопроса: СМЫСЛОВОЙ.\n"
            "- Давай содержательные, развёрнутые ответы.\n"
            "- Используй цитаты из документа для подтверждения.\n"
            "- Структурируй ответ логично (введение, основная часть, вывод).\n"
            "- В конце ОБЯЗАТЕЛЬНО напиши краткий вывод.\n"
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
    
    async def _answer_sync(self, system: str, question: str, context: str) -> str:
        """Синхронный вызов LLM"""
        try:
            # Формируем сообщения
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Контекст из документа:\n{context}\n\nВопрос: {question}"}
            ]
            
            # Вызываем API
            answer = await asyncio.to_thread(
                chat_with_gpt,
                messages=messages,
                temperature=0.3,  # Низкая температура для точности
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