-- migrations/xxxx_user_state.sql
-- Создание user_state для хранения активного документа и прочих пользовательских настроек.
-- Совместимо с существующей схемой (users.active_doc_id остаётся источником истины для старого кода).

PRAGMA foreign_keys=ON;

BEGIN;

-- 1) Таблица пользовательского состояния
CREATE TABLE IF NOT EXISTS user_state (
    user_id        INTEGER NOT NULL,                 -- users.id
    active_doc_id  INTEGER,                          -- documents.id (может быть NULL)
    last_seen      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    extras         TEXT,                             -- JSON с дополнительными настройками (опционально)
    PRIMARY KEY (user_id),
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (active_doc_id) REFERENCES documents(id) ON DELETE SET NULL ON UPDATE CASCADE
);

-- 2) Индексы
CREATE INDEX IF NOT EXISTS idx_user_state_active_doc ON user_state(active_doc_id);
CREATE INDEX IF NOT EXISTS idx_user_state_last_seen  ON user_state(last_seen);

-- 3) Инициализация записей состояния для уже существующих пользователей (если их нет)
INSERT OR IGNORE INTO user_state(user_id)
SELECT id FROM users;

-- 4) Бэкоф активного документа из users.active_doc_id → user_state.active_doc_id
--    Переносим ТОЛЬКО валидные ссылки на существующие документы, чтобы не ломать FK.
UPDATE user_state
SET active_doc_id = (
  SELECT u.active_doc_id
  FROM users u
  WHERE u.id = user_state.user_id
)
WHERE active_doc_id IS NULL
  AND EXISTS (
    SELECT 1
    FROM users u
    JOIN documents d ON d.id = u.active_doc_id
    WHERE u.id = user_state.user_id
  );

-- 5) Триггеры синхронизации

-- 5.1 Insert пользователя → создаём строку в user_state и переносим активный документ (если есть и валиден)
CREATE TRIGGER IF NOT EXISTS user_state_after_users_insert
AFTER INSERT ON users
BEGIN
  INSERT OR IGNORE INTO user_state(user_id, active_doc_id, last_seen)
  VALUES (
    NEW.id,
    (SELECT CASE
              WHEN NEW.active_doc_id IS NOT NULL
                   AND EXISTS (SELECT 1 FROM documents d WHERE d.id = NEW.active_doc_id)
                THEN NEW.active_doc_id
              ELSE NULL
            END),
    CURRENT_TIMESTAMP
  );
END;

-- 5.2 Обновление users.active_doc_id → отражаем в user_state (и обновляем last_seen), только если реально изменилось
CREATE TRIGGER IF NOT EXISTS user_state_after_users_update_active
AFTER UPDATE OF active_doc_id ON users
WHEN OLD.active_doc_id IS NOT NEW.active_doc_id
BEGIN
  UPDATE user_state
  SET active_doc_id = NEW.active_doc_id,
      last_seen = CURRENT_TIMESTAMP
  WHERE user_id = NEW.id
    AND (active_doc_id IS NOT NEW.active_doc_id);
END;

-- 5.3 Обновление user_state.active_doc_id → отражаем в users (чтобы старый код видел актуальное значение),
--     только если реально изменилось; защищено от зацикливания условием WHERE.
CREATE TRIGGER IF NOT EXISTS users_after_user_state_update_active
AFTER UPDATE OF active_doc_id ON user_state
WHEN OLD.active_doc_id IS NOT NEW.active_doc_id
BEGIN
  UPDATE users
  SET active_doc_id = NEW.active_doc_id
  WHERE id = NEW.user_id
    AND (active_doc_id IS NOT NEW.active_doc_id);
END;

-- 5.4 Любое изменение extras → обновляем last_seen (без рекурсии, т.к. last_seen не входит в список обновлённых колонок)
CREATE TRIGGER IF NOT EXISTS user_state_touch_on_extras
AFTER UPDATE OF extras ON user_state
BEGIN
  UPDATE user_state
  SET last_seen = CURRENT_TIMESTAMP
  WHERE user_id = NEW.user_id;
END;

COMMIT;
