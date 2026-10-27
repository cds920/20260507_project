"""월별 제출 현황판: OJT 운영일, 조회 월, 데이터 보존."""
from __future__ import annotations

import datetime
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_import_stubs() -> None:
    if "streamlit" not in sys.modules:
        st = ModuleType("streamlit")

        def _deco(*args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def wrap(fn):
                return fn

            return wrap

        st.cache_data = _deco
        st.cache_resource = _deco
        st.secrets = {}
        st.error = lambda *a, **k: None
        st.stop = lambda: None
        st.cache_data.clear = lambda: None  # type: ignore[attr-defined]
        sys.modules["streamlit"] = st
    if "gspread" not in sys.modules:
        gspread = ModuleType("gspread")

        class _Client:
            pass

        gspread.Client = _Client
        sys.modules["gspread"] = gspread
        exc = ModuleType("gspread.exceptions")

        class APIError(Exception):
            pass

        class WorksheetNotFound(Exception):
            pass

        exc.APIError = APIError
        exc.WorksheetNotFound = WorksheetNotFound
        gspread.exceptions = exc
        sys.modules["gspread.exceptions"] = exc


import zoneinfo

class _DummyTZ:
    def __init__(self, key):
        self.key = key


zoneinfo.ZoneInfo = _DummyTZ
_install_import_stubs()

from db import (  # noqa: E402
    attendance_month_choices,
    board_dates_for_month,
    is_ojt_operating_day,
    ojt_span_end,
    ojt_span_start,
    seed_demo_logs_if_empty,
    _purge_logs_outside_test_period,
)


class AttendancePeriodTests(unittest.TestCase):
    def test_september_weekdays_before_ojt_are_not_board_columns(self):
        days = board_dates_for_month(2026, 9)
        self.assertNotIn(datetime.date(2026, 9, 1), days)
        self.assertNotIn(datetime.date(2026, 9, 2), days)
        self.assertNotIn(datetime.date(2026, 9, 3), days)
        self.assertNotIn(datetime.date(2026, 9, 4), days)
        self.assertIn(datetime.date(2026, 9, 7), days)

    def test_extra_log_date_keeps_column_outside_ojt(self):
        extra = [datetime.date(2026, 9, 3)]
        days = board_dates_for_month(2026, 9, extra_dates=extra)
        self.assertIn(datetime.date(2026, 9, 3), days)
        self.assertIn(datetime.date(2026, 9, 7), days)

    def test_may_board_matches_first_term_display_window(self):
        days = board_dates_for_month(2026, 5)
        self.assertNotIn(datetime.date(2026, 5, 1), days)
        self.assertNotIn(datetime.date(2026, 5, 11), days)
        self.assertIn(datetime.date(2026, 5, 14), days)
        self.assertIn(datetime.date(2026, 5, 29), days)

    def test_october_ojt_weekdays_are_included(self):
        days = board_dates_for_month(2026, 10)
        self.assertIn(datetime.date(2026, 10, 5), days)
        self.assertIn(datetime.date(2026, 10, 12), days)

    def test_month_choices_come_from_logs_terms_and_today(self):
        today = datetime.date(2026, 10, 27)
        months = attendance_month_choices(
            [datetime.date(2026, 5, 20)],
            today=today,
        )
        self.assertIn((2026, 5), months)
        self.assertIn((2026, 9), months)
        self.assertIn((2026, 10), months)
        self.assertNotIn((2026, 6), months)
        self.assertNotIn((2026, 7), months)

    def test_current_month_is_always_selectable(self):
        today = datetime.date(2026, 11, 3)
        months = attendance_month_choices([], today=today)
        self.assertIn((2026, 11), months)

    def test_ojt_operating_day_rules(self):
        self.assertFalse(is_ojt_operating_day(datetime.date(2026, 9, 4)))
        self.assertTrue(is_ojt_operating_day(datetime.date(2026, 9, 7)))
        self.assertTrue(is_ojt_operating_day(datetime.date(2026, 5, 14)))
        self.assertFalse(is_ojt_operating_day(datetime.date(2026, 5, 13)))
        self.assertTrue(is_ojt_operating_day(datetime.date(2026, 11, 14)))
        self.assertTrue(is_ojt_operating_day(datetime.date(2027, 1, 9)))
        self.assertFalse(is_ojt_operating_day(datetime.date(2027, 1, 10)))

    def test_second_term_months_include_through_january(self):
        today = datetime.date(2026, 10, 27)
        months = attendance_month_choices([], today=today)
        self.assertIn((2026, 11), months)
        self.assertIn((2026, 12), months)
        self.assertIn((2027, 1), months)
        self.assertNotIn((2027, 2), months)

    def test_purge_and_demo_seed_do_not_touch_sheets(self):
        with patch("db.init_db") as init_db:
            self.assertEqual(_purge_logs_outside_test_period(), 0)
            self.assertEqual(seed_demo_logs_if_empty(force_refresh=True), 0)
            init_db.assert_not_called()

    def test_ojt_span_covers_may_through_january(self):
        self.assertEqual(ojt_span_start(), datetime.date(2026, 5, 11))
        self.assertEqual(ojt_span_end(), datetime.date(2027, 1, 9))

    def test_teacher_defines_board_start_and_end(self):
        src = (ROOT / "teacher_view.py").read_text(encoding="utf-8")
        self.assertIn("_ATTENDANCE_BOARD_START", src)
        self.assertIn("_ATTENDANCE_BOARD_END", src)
        self.assertIn("ojt_span_start()", src)
        self.assertIn("ojt_span_end()", src)
        self.assertNotRegex(src, r"(?<!_)ATTENDANCE_BOARD_START")

    def test_teacher_board_has_month_dropdown_not_fixed_caption(self):
        src = (ROOT / "teacher_view.py").read_text(encoding="utf-8")
        self.assertIn('"조회 월"', src)
        self.assertIn("board_dates_for_month", src)
        self.assertIn("attendance_reminder_date_col_{year}_{month}", src)
        self.assertNotIn("실전 테스트 기간", src)
        self.assertNotIn("_build_test_period_attendance", src)

    def test_holiday_and_absence_still_in_board_styles(self):
        src = (ROOT / "teacher_view.py").read_text(encoding="utf-8")
        self.assertIn('_ATTENDANCE_STATUS_HOLIDAY = "공휴일"', src)
        self.assertIn('_ATTENDANCE_STATUS_ABSENCE = "결석"', src)
        self.assertIn('_ATTENDANCE_STATUS_MISSING = "미제출"', src)
        self.assertIn("결석 및 예외 처리", src)


if __name__ == "__main__":
    unittest.main()
