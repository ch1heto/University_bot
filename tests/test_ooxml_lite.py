import pytest

from app.ooxml_lite import _norm_label_num, _split_caption_num_tail


@pytest.mark.parametrize(
    "caption,is_table,expected_num,expected_tail",
    [
        ("Рисунок 2.3 — Структура", False, "2.3", "Структура"),
        ("Table 1.2: Data", True, "1.2", "Data"),
        ("Figure 5", False, "5", None),
    ],
)
def test_split_caption_num_tail(caption, is_table, expected_num, expected_tail):
    num, tail = _split_caption_num_tail(caption, is_table=is_table)

    assert num == expected_num
    assert tail == expected_tail


@pytest.mark.parametrize(
    "raw,expected",
    [
        (" 2 , 3 ", "2.3"),
        ("A  10, 4", "10.4"),
        ("Б 7,8", "7.8"),
    ],
)
def test_norm_label_num_handles_spaces_and_commas(raw, expected):
    assert _norm_label_num(raw) == expected
