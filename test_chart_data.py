#!/usr/bin/env python3
"""
Диагностика второго документа (ВКР_Филин.docx)
Проверяет таблицы и рисунки
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_parsing():
    """Тест парсинга документа"""
    print("="*60)
    print("ТЕСТ: Парсинг ВКР_Филин.docx")
    print("="*60)
    
    try:
        from app.parsing_new import parse_docx
        
        # Путь к документу (подставь свой)
        docx_path = None
        for p in ["uploads/ВКР_Филин.docx", "uploads/781477708_ВКР_Филин.docx"]:
            if os.path.exists(p):
                docx_path = p
                break
        
        if not docx_path:
            # Ищем любой docx с "Филин" в названии
            for root, dirs, files in os.walk("uploads"):
                for f in files:
                    if "Филин" in f and f.endswith(".docx"):
                        docx_path = os.path.join(root, f)
                        break
        
        if not docx_path:
            print("❌ Файл ВКР_Филин.docx не найден в uploads/")
            return
        
        print(f"Файл: {docx_path}")
        
        result = parse_docx(docx_path)
        
        sections = result.get("sections", [])
        tables = result.get("tables", [])
        figures = result.get("figures", [])
        
        print(f"\nРезультат парсинга:")
        print(f"  Секций: {len(sections)}")
        print(f"  Таблиц: {len(tables)}")
        print(f"  Рисунков: {len(figures)}")
        
        # Проверяем таблицы
        print(f"\n--- ТАБЛИЦЫ ---")
        for i, tbl in enumerate(tables[:5], 1):
            print(f"\nТаблица {i}:")
            print(f"  num: {tbl.get('num')}")
            print(f"  caption: {tbl.get('caption', '')[:50]}...")
            print(f"  rows: {tbl.get('rows')}, cols: {tbl.get('cols')}")
            data = tbl.get('data', [])
            if data:
                print(f"  Первая строка: {data[0][:3] if len(data[0]) > 3 else data[0]}")
        
        # Проверяем рисунки
        print(f"\n--- РИСУНКИ ---")
        for i, fig in enumerate(figures[:5], 1):
            print(f"\nРисунок {i}:")
            print(f"  num: {fig.get('num')}")
            print(f"  kind: {fig.get('kind')}")
            print(f"  caption: {fig.get('caption', '')[:50]}...")
            print(f"  image_path: {fig.get('image_path')}")
            print(f"  chart_data: {bool(fig.get('chart_data'))}")
        
        return {"tables": tables, "figures": figures}
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_db_for_doc():
    """Проверяем что в БД для последнего документа"""
    print("\n" + "="*60)
    print("ТЕСТ: Проверка БД")
    print("="*60)
    
    try:
        from app.db import get_conn, get_figures_for_doc
        import json
        
        con = get_conn()
        cur = con.cursor()
        
        # Получаем последний документ
        cur.execute("SELECT id, path FROM documents ORDER BY id DESC LIMIT 1")
        doc = cur.fetchone()
        
        if not doc:
            print("❌ Нет документов в БД")
            return
        
        doc_id = doc["id"]
        print(f"Последний документ: id={doc_id}, path={doc['path']}")
        
        # Проверяем таблицы в chunks
        cur.execute("""
            SELECT COUNT(*) as cnt FROM chunks 
            WHERE doc_id = ? AND (
                element_type = 'table' OR 
                element_type = 'table_row' OR
                section_path LIKE '%Таблица%'
            )
        """, (doc_id,))
        table_chunks = cur.fetchone()["cnt"]
        print(f"Чанков с таблицами: {table_chunks}")
        
        # Проверяем рисунки
        cur.execute("SELECT COUNT(*) as cnt FROM figures WHERE doc_id = ?", (doc_id,))
        figures_cnt = cur.fetchone()["cnt"]
        print(f"Рисунков в figures: {figures_cnt}")
        
        # Детали по рисункам
        figures = get_figures_for_doc(doc_id)
        print(f"\nДетали рисунков:")
        for fig in figures[:5]:
            attrs = fig.get('attrs') or {}
            has_chart = 'chart_data' in attrs if isinstance(attrs, dict) else False
            print(f"  label={fig.get('label')}, kind={fig.get('figure_kind')}, chart_data={has_chart}")
        
        con.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


def test_table_search():
    """Тест поиска по таблицам"""
    print("\n" + "="*60)
    print("ТЕСТ: Поиск по таблицам")
    print("="*60)
    
    try:
        from app.db import get_conn
        
        con = get_conn()
        cur = con.cursor()
        
        # Получаем последний документ
        cur.execute("SELECT id FROM documents ORDER BY id DESC LIMIT 1")
        doc = cur.fetchone()
        if not doc:
            print("❌ Нет документов")
            return
        
        doc_id = doc["id"]
        
        # Ищем чанки с упоминанием "Таблица 2.1"
        cur.execute("""
            SELECT id, section_path, element_type, text 
            FROM chunks 
            WHERE doc_id = ? AND (
                text LIKE '%Таблица 2.1%' OR
                section_path LIKE '%Таблица 2.1%' OR
                text LIKE '%финансов%результат%'
            )
            LIMIT 5
        """, (doc_id,))
        
        rows = cur.fetchall()
        print(f"Найдено чанков с 'Таблица 2.1' или 'финансов': {len(rows)}")
        
        for row in rows:
            print(f"\n  ID: {row['id']}")
            print(f"  section_path: {row['section_path'][:80]}...")
            print(f"  element_type: {row['element_type']}")
            print(f"  text: {row['text'][:100]}...")
        
        con.close()
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parsing()
    test_db_for_doc()
    test_table_search()