import sqlite3, pathlib
db = pathlib.Path("vkr.sqlite").as_posix()
with open("sql/init_sql.sql", "r", encoding="utf-8") as f:
    sql = f.read()
con = sqlite3.connect(db)
con.executescript(sql)
con.close()
print("SQLite готов:", db)
