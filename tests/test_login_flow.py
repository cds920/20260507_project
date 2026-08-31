"""로그인 경로: 캐시 중첩 방지, 스피너 문구, 인증 성공/실패."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_import_stubs() -> None:
    """테스트 러너에 streamlit/gspread가 없어도 db.authenticate를 검증한다."""
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


zoneinfo.ZoneInfo = _DummyTZ  # Windows 테스트 러너에 tzdata가 없을 수 있음
_install_import_stubs()

from db import (  # noqa: E402
    STUDENTS_HEADERS,
    authenticate,
    get_user,
    hash_password,
)


def _cells(uid: str, password: str, role: str) -> list[str]:
    row = [""] * len(STUDENTS_HEADERS)
    hi = {h: i for i, h in enumerate(STUDENTS_HEADERS)}
    row[hi["uid"]] = uid
    row[hi["password"]] = password
    row[hi["role"]] = role
    return row


class LoginFlowTests(unittest.TestCase):
    @patch("db.init_db")
    @patch("db._find_student_row")
    def test_a_student_login(self, find, _init):
        find.return_value = (2, _cells("yongsan1", hash_password("1234"), "student"))
        user = authenticate("yongsan1", "1234")
        self.assertEqual(user["uid"], "yongsan1")
        self.assertEqual(user["role"], "student")
        self.assertNotIn("password", user)

    @patch("db.init_db")
    @patch("db._find_student_row")
    def test_b_teacher_login(self, find, _init):
        find.return_value = (2, _cells("teacher", hash_password("1234"), "teacher"))
        user = authenticate("teacher", "1234")
        self.assertEqual(user["uid"], "teacher")
        self.assertEqual(user["role"], "teacher")

    @patch("db.init_db")
    @patch("db._find_student_row")
    def test_c_wrong_password(self, find, _init):
        find.return_value = (2, _cells("yongsan1", hash_password("1234"), "student"))
        self.assertIsNone(authenticate("yongsan1", "wrong"))

    @patch("db.init_db")
    @patch("db._find_student_row")
    def test_d_unknown_id(self, find, _init):
        find.return_value = None
        self.assertIsNone(authenticate("nobody", "1234"))
        self.assertIsNone(get_user("nobody"))

    def test_e_get_user_is_not_cache_data(self):
        self.assertFalse(hasattr(get_user, "clear"))
        src = (ROOT / "db.py").read_text(encoding="utf-8")
        idx = src.find("def get_user(")
        self.assertGreater(idx, 0)
        prelude = src[max(0, idx - 120) : idx]
        self.assertNotIn("@st.cache_data", prelude)

    def test_f_login_ui_routes_without_internal_spinner_name(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("show_login()", app)
        self.assertIn("authenticate(uid_norm", app)
        self.assertIn('st.session_state.user = user["uid"]', app)
        self.assertIn("로그인 정보를 확인하고 있습니다", app)
        self.assertIn("사용자 정보를 불러오는 중 문제가 발생했습니다", app)
        self.assertIn("show_student(uid)", app)
        self.assertIn("show_teacher()", app)
        self.assertNotIn("Running get_user", app)

    @patch("db.init_db")
    @patch("db._find_student_row", side_effect=RuntimeError("sheets down"))
    def test_sheets_error_is_not_swallowed_as_bad_password(self, _find, _init):
        with self.assertRaises(RuntimeError):
            authenticate("yongsan1", "1234")

    def test_sheets_retry_is_finite(self):
        src = (ROOT / "db.py").read_text(encoding="utf-8")
        self.assertIn("delays = (0.8, 1.6, 3.0)", src)
        self.assertIn('kwargs.setdefault("timeout", (8, 20))', src)
        self.assertIn("show_spinner=False", src)
        self.assertIn("@st.cache_data(ttl=60, show_spinner=False)", src)


if __name__ == "__main__":
    unittest.main()
