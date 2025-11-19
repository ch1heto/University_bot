from aiogram import types, F, Dispatcher
from aiogram.filters import Command
from aiogram.utils.keyboard import InlineKeyboardBuilder

from .db import (
    ensure_user,
    list_user_documents,
    set_user_active_doc,
)

DOC_SELECT_PREFIX = "doc_select:"


def register_docs_handlers(dp: Dispatcher) -> None:
    """
    Регистрируем хендлеры для работы с документами:
    - команда /docs
    - выбор документа по кнопке
    """

    @dp.message(Command("docs"))
    async def cmd_docs(message: types.Message):
        tg_id = str(message.from_user.id)
        user_id = ensure_user(tg_id)

        docs = list_user_documents(user_id, limit=10)
        if not docs:
            await message.answer(
                "У вас пока нет загруженных документов.\n"
                "Сначала отправьте файл, а потом попробуйте /docs."
            )
            return

        # Используем InlineKeyboardBuilder вместо row_width
        builder = InlineKeyboardBuilder()

        for d in docs:
            # Берём только «хвост» пути, чтобы не показывать полный путь на диске
            path = d["path"] or ""
            name = path.split("/")[-1].split("\\")[-1] or f"Документ {d['id']}"
            prefix = "✅ " if d.get("is_active") else "📄 "
            text = f"{prefix}{name}"

            builder.button(
                text=text,
                callback_data=f"{DOC_SELECT_PREFIX}{d['id']}",
            )

        # по одной кнопке в строке
        builder.adjust(1)

        await message.answer(
            "Выберите документ, по которому задавать вопросы:",
            reply_markup=builder.as_markup(),
        )

    @dp.callback_query(F.data.startswith(DOC_SELECT_PREFIX))
    async def on_doc_select(callback: types.CallbackQuery):
        tg_id = str(callback.from_user.id)
        user_id = ensure_user(tg_id)

        payload = callback.data[len(DOC_SELECT_PREFIX):]
        try:
            doc_id = int(payload)
        except ValueError:
            await callback.answer("Некорректный документ.", show_alert=True)
            return

        # Сохраняем выбор пользователя
        set_user_active_doc(user_id, doc_id)

        await callback.answer("Активный документ изменён ✅", show_alert=False)

        # Обновляем текст сообщения
        await callback.message.edit_text(
            f"Текущий активный документ: ID {doc_id}.\n"
            f"Теперь все вопросы будут относиться к этому файлу."
        )
