"""NCS 포트폴리오 — Google 스프레드시트 전용 저장소 (sqlite 미사용).

원 SQLite 스키마와의 대응
------------------------
* **logs** 테이블 → ``logs`` 시트 1행 헤더·열 순서 동일:
  ``id``, ``uid``, ``date``, ``ncs_unit``, ``bsr``, ``image_note``, ``image_b64``,
  ``audio_note``, ``ncs_term_ratio``, ``created_at``
* **users**, **student_profiles**, **portfolio_comments**, **progress** 는
  한 학생(또는 교사)당 한 행으로 집약해 ``students`` 시트에 저장
  (``uid``, ``password``, ``role``, 이력서·포트폴리오·``progress_json`` 등).
* **researcher_logs** → ``researcher_logs`` 시트 (교사 화면 호환).

연결 URL: https://docs.google.com/spreadsheets/d/1TrqWys6ZVVYfN0Pi6vitZ255rilYNxHoLw3AWvDJI_Y/edit
"""
from __future__ import annotations

import streamlit as st
import gspread
import json
import datetime
import math
import random
import re
import secrets
from typing import Any

from gspread.exceptions import WorksheetNotFound

# ───────────────────────────────────────────────────────────────────
# 스프레드시트
# ───────────────────────────────────────────────────────────────────
SPREADSHEET_ID: str = "1TrqWys6ZVVYfN0Pi6vitZ255rilYNxHoLw3AWvDJI_Y"

STUDENTS_HEADERS: list[str] = [
    "uid",
    "password",
    "role",
    "full_name",
    "birth_date",
    "email",
    "phone",
    "motto",
    "photo_b64",
    "educations_json",
    "careers_json",
    "certificates_json",
    "awards_json",
    "tech_stack_json",
    "profile_updated_at",
    "portfolio_comment_text",
    "portfolio_reflection_level",
    "portfolio_updated_at",
    "portfolio_is_confirmed",
    "progress_json",
]

LOGS_HEADERS: list[str] = [
    "id",
    "uid",
    "date",
    "ncs_unit",
    "bsr",
    "image_note",
    "image_b64",
    "audio_note",
    "ncs_term_ratio",
    "created_at",
]

RESEARCHER_HEADERS: list[str] = ["id", "log_date", "note", "created_at"]

_db_initialized: bool = False


def _invalidate_read_caches() -> None:
    """시트 쓰기 후 read 캐시를 비운다 (429 방지와 데이터 정합)."""
    st.cache_data.clear()


@st.cache_resource
def get_gspread_client() -> gspread.Client:
    try:
        creds_data = st.secrets["GOOGLE_CREDENTIALS"]
        if isinstance(creds_data, str):
            creds_info = json.loads(creds_data, strict=False)
        else:
            creds_info = dict(creds_data)
        return gspread.service_account_from_dict(creds_info)
    except Exception as e:
        st.error(f"구글 시트 연결 실패: {str(e)}")
        st.stop()
        raise AssertionError("unreachable") from e


@st.cache_resource
def _cached_open_spreadsheet() -> gspread.Spreadsheet:
    return get_gspread_client().open_by_key(SPREADSHEET_ID)


@st.cache_resource
def _cached_worksheet(title: str) -> gspread.Worksheet:
    return _cached_open_spreadsheet().worksheet(title)


def _get_spreadsheet() -> gspread.Spreadsheet:
    return _cached_open_spreadsheet()


def _invalidate_connection_caches() -> None:
    st.cache_resource.clear()
    st.cache_data.clear()


def _ensure_worksheet(title: str, rows: int = 2000, cols: int = 30) -> gspread.Worksheet:
    """WorksheetNotFound 방지: 없으면 add_worksheet 후 리소스 캐시 초기화."""
    sh = _cached_open_spreadsheet()
    try:
        return sh.worksheet(title)
    except WorksheetNotFound:
        sh.add_worksheet(title=title, rows=int(rows), cols=int(cols))
        _invalidate_connection_caches()
        return _cached_open_spreadsheet().worksheet(title)


def _col_letter(n: int) -> str:
    """1-based column index → A1 letters."""
    s = ""
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def _header_range(headers: list[str]) -> str:
    end = _col_letter(len(headers))
    return f"A1:{end}1"


def _strip_bom(s: str) -> str:
    return str(s or "").strip().lstrip("\ufeff")


def _headers_match(row1: list[str], expected: list[str]) -> bool:
    if len(row1) < len(expected):
        return False
    return all(_strip_bom(str(row1[i])) == expected[i] for i in range(len(expected)))


def _sheet_has_body_rows(all_values: list[list[str]]) -> bool:
    for row in all_values[1:]:
        if row and any(_strip_bom(str(c)) for c in row):
            return True
    return False


# Google Sheets 단일 셀 문자 수 상한(공식 50,000자; 여유를 둔다).
GOOGLE_SHEETS_MAX_CELL_CHARS: int = 49000


def _truncate_sheet_cell(s: str, *, tail: str = "…[셀 한도 초과로 잘림]") -> str:
    """시트 API 오류(400 등)를 막기 위해 셀당 길이를 제한한다."""
    s = str(s or "")
    if len(s) <= GOOGLE_SHEETS_MAX_CELL_CHARS:
        return s
    budget = GOOGLE_SHEETS_MAX_CELL_CHARS - len(tail)
    if budget < 1:
        return s[:GOOGLE_SHEETS_MAX_CELL_CHARS]
    return s[:budget] + tail


def _coerce_log_image_for_sheet(
    image_b64: str | None,
    image_note: str | None,
) -> tuple[str, str]:
    """data URI 등 긴 이미지는 셀 한도 초과 시 비우고 메모에 안내(저장은 계속)."""
    b64 = _cell_str(image_b64)
    note = _truncate_sheet_cell(_cell_str(image_note))
    if not b64:
        return "", note
    if len(b64) <= GOOGLE_SHEETS_MAX_CELL_CHARS:
        return b64, note
    warn = "(증거 사진: 스프레드시트 셀 한도로 저장 생략)"
    if note:
        return "", _truncate_sheet_cell(f"{warn} {note}".strip())
    return "", warn


def _normalize_log_sheet_row(cells: list[str]) -> list[str]:
    """기존 logs 행을 시트 셀 한도에 맞게 다듬어 update 시 API 오류를 줄인다."""
    row = list(cells[: len(LOGS_HEADERS)])
    while len(row) < len(LOGS_HEADERS):
        row.append("")
    bi = LOGS_HEADERS.index("image_b64")
    ni = LOGS_HEADERS.index("image_note")
    b64_s, note_s = _coerce_log_image_for_sheet(row[bi], row[ni])
    row[bi] = b64_s
    row[ni] = note_s
    row[LOGS_HEADERS.index("bsr")] = _truncate_sheet_cell(row[LOGS_HEADERS.index("bsr")])
    row[LOGS_HEADERS.index("audio_note")] = _truncate_sheet_cell(row[LOGS_HEADERS.index("audio_note")])
    for key in ("uid", "date", "ncs_unit", "created_at"):
        i = LOGS_HEADERS.index(key)
        row[i] = _truncate_sheet_cell(row[i])
    return row


def _cell_str(v: Any) -> str:
    """시트에 쓰기 위한 셀 값: 항상 str, None·NaN은 빈 문자열."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        if v == int(v):
            return str(int(v))
        return format(v, "f").rstrip("0").rstrip(".") or "0"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, datetime.datetime):
        return v.isoformat(timespec="seconds")
    if isinstance(v, datetime.date):
        return v.isoformat()
    return str(v)


def _parse_int_cell(val: Any, *, default: int = 0) -> int:
    s = str(val or "").strip().replace(",", "")
    if not s:
        return default
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return default


def _parse_float_cell(val: Any) -> float | None:
    s = str(val or "").strip().replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _ensure_sheet_headers(ws: gspread.Worksheet, headers: list[str]) -> None:
    """1행이 비었거나 본문이 없으면 헤더를 쓴다. 본문이 있는데 헤더가 다르면 명시적 오류."""
    allv = ws.get_all_values()
    row1 = allv[0] if allv else []
    if not row1 or not any(_strip_bom(str(x)) for x in row1):
        ws.update(range_name=_header_range(headers), values=[headers], value_input_option="RAW")
        _invalidate_read_caches()
        return
    if _headers_match(row1, headers):
        return
    if _sheet_has_body_rows(allv):
        raise RuntimeError(
            f"워크시트 「{ws.title}」 1행 헤더가 앱이 기대하는 열과 다릅니다. "
            f"데이터가 있으면 백업 후 1행을 다음 순서로 맞추세요: {', '.join(headers)}"
        )
    ws.update(range_name=_header_range(headers), values=[headers], value_input_option="RAW")
    _invalidate_read_caches()


def init_db() -> None:
    """앱 시작 시 한 번: 시트 존재·헤더 보장."""
    global _db_initialized
    if _db_initialized:
        return
    get_gspread_client()
    stu = _ensure_worksheet("students", rows=500, cols=max(32, len(STUDENTS_HEADERS) + 2))
    log = _ensure_worksheet("logs", rows=8000, cols=max(16, len(LOGS_HEADERS) + 2))
    res = _ensure_worksheet("researcher_logs", rows=500, cols=8)
    _ensure_sheet_headers(stu, STUDENTS_HEADERS)
    _ensure_sheet_headers(log, LOGS_HEADERS)
    _ensure_sheet_headers(res, RESEARCHER_HEADERS)
    _db_initialized = True


def reset_connection() -> None:
    """다음 호출에서 스프레드시트 핸들을 다시 연다."""
    global _db_initialized
    _db_initialized = False
    _invalidate_connection_caches()


def _students_ws() -> gspread.Worksheet:
    init_db()
    return _cached_worksheet("students")


def _logs_ws() -> gspread.Worksheet:
    init_db()
    return _cached_worksheet("logs")


def _researcher_ws() -> gspread.Worksheet:
    init_db()
    return _cached_worksheet("researcher_logs")


@st.cache_data(ttl=60)
def _bulk_students_values() -> tuple[tuple[str, ...], ...]:
    init_db()
    rows = _cached_worksheet("students").get_all_values()
    return tuple(tuple("" if c is None else str(c) for c in r) for r in rows)


@st.cache_data(ttl=60)
def _bulk_logs_values() -> tuple[tuple[str, ...], ...]:
    init_db()
    rows = _cached_worksheet("logs").get_all_values()
    return tuple(tuple("" if c is None else str(c) for c in r) for r in rows)


@st.cache_data(ttl=60)
def _bulk_researcher_values() -> tuple[tuple[str, ...], ...]:
    init_db()
    rows = _cached_worksheet("researcher_logs").get_all_values()
    return tuple(tuple("" if c is None else str(c) for c in r) for r in rows)


def student_number(uid: str) -> int:
    m = re.search(r"(\d+)\s*$", str(uid or ""))
    if not m:
        return 999
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return 999


def student_label(uid: str) -> str:
    n = student_number(uid)
    return f"{n}번 도제생" if n != 999 else str(uid)


TEACHER_UID: str = "teacher"
DEFAULT_PASSWORD: str = "1234"
STUDENT_COUNT: int = 10
STUDENT_UIDS: tuple[str, ...] = tuple(f"yongsan{i}" for i in range(1, STUDENT_COUNT + 1))

_LEGACY_UID_MAP: dict[str, str] = {"admin": TEACHER_UID}
for _i in range(1, STUDENT_COUNT + 1):
    _LEGACY_UID_MAP[f"S{_i:02d}"] = f"yongsan{_i}"
del _i

TEST_PERIOD_START: datetime.date = datetime.date(2026, 5, 11)
TEST_PERIOD_END: datetime.date = datetime.date(2026, 5, 29)


def test_period_weekdays() -> list[datetime.date]:
    days: list[datetime.date] = []
    d = TEST_PERIOD_START
    while d <= TEST_PERIOD_END:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def app_today() -> datetime.date:
    real_today = datetime.date.today()
    if real_today < TEST_PERIOD_START:
        return TEST_PERIOD_START
    if real_today > TEST_PERIOD_END:
        return TEST_PERIOD_END
    return real_today


def _students_values() -> list[list[str]]:
    init_db()
    return [list(row) for row in _bulk_students_values()]


def _student_header_index() -> dict[str, int]:
    return {h: i for i, h in enumerate(STUDENTS_HEADERS)}


def _find_student_row(uid: str) -> tuple[int, list[str]] | None:
    """1-based data row index or None. Returns (row_index, row_values padded)."""
    rows = _students_values()
    if len(rows) < 2:
        return None
    hdr = rows[0]
    if not _headers_match(hdr, STUDENTS_HEADERS):
        return None
    want = str(uid).strip().lower()
    for r_i, row in enumerate(rows[1:], start=2):
        if not row:
            continue
        if str(row[0]).strip().lower() == want:
            while len(row) < len(STUDENTS_HEADERS):
                row.append("")
            return r_i, row
    return None


def _default_student_row(uid: str, password: str, role: str) -> list[str]:
    row = [""] * len(STUDENTS_HEADERS)
    hi = _student_header_index()
    row[hi["uid"]] = uid
    row[hi["password"]] = password
    row[hi["role"]] = role
    row[hi["progress_json"]] = "{}"
    row[hi["portfolio_is_confirmed"]] = "0"
    row[hi["profile_updated_at"]] = ""
    row[hi["portfolio_updated_at"]] = ""
    return row


def _write_student_row(row_1based: int, cells: list[str]) -> None:
    ws = _students_ws()
    end = _col_letter(len(STUDENTS_HEADERS))
    rng = f"A{row_1based}:{end}{row_1based}"
    pad = list(cells)
    while len(pad) < len(STUDENTS_HEADERS):
        pad.append("")
    row_out = [_cell_str(x) for x in pad[: len(STUDENTS_HEADERS)]]
    ws.update(range_name=rng, values=[row_out], value_input_option="RAW")
    _invalidate_read_caches()


def _append_student_row(cells: list[str]) -> None:
    ws = _students_ws()
    pad = list(cells)
    while len(pad) < len(STUDENTS_HEADERS):
        pad.append("")
    ws.append_row([_cell_str(x) for x in pad[: len(STUDENTS_HEADERS)]], value_input_option="RAW")
    _invalidate_read_caches()


def _delete_student_row(row_1based: int) -> None:
    _students_ws().delete_rows(row_1based)
    _invalidate_read_caches()


def _migrate_legacy_uids() -> None:
    if not _LEGACY_UID_MAP:
        return
    for old_uid, new_uid in _LEGACY_UID_MAP.items():
        old_r = _find_student_row(old_uid)
        if not old_r:
            continue
        new_r = _find_student_row(new_uid)
        if new_r:
            for tbl_uid in (old_uid,):
                _delete_logs_for_uid(tbl_uid)
            _delete_student_row(old_r[0])
            continue
        row_i, cells = old_r
        hi = _student_header_index()
        cells[hi["uid"]] = new_uid
        cells[hi["password"]] = DEFAULT_PASSWORD
        _write_student_row(row_i, cells)
        _rewrite_logs_uid(old_uid, new_uid)


def _rewrite_logs_uid(old_uid: str, new_uid: str) -> None:
    init_db()
    ws = _logs_ws()
    all_v = [list(r) for r in _bulk_logs_values()]
    if len(all_v) < 2:
        return
    hdr = all_v[0]
    if not _headers_match(hdr, LOGS_HEADERS):
        return
    ui = LOGS_HEADERS.index("uid")
    any_written = False
    for r_i, row in enumerate(all_v[1:], start=2):
        if len(row) > ui and str(row[ui]).strip().lower() == str(old_uid).strip().lower():
            row = list(row)
            while len(row) < len(LOGS_HEADERS):
                row.append("")
            row[ui] = _cell_str(new_uid)
            end = _col_letter(len(LOGS_HEADERS))
            row_norm = _normalize_log_sheet_row([_cell_str(x) for x in row[: len(LOGS_HEADERS)]])
            ws.update(
                range_name=f"A{r_i}:{end}{r_i}",
                values=[row_norm],
                value_input_option="RAW",
            )
            any_written = True
    if any_written:
        _invalidate_read_caches()


def _delete_logs_for_uid(uid: str) -> None:
    ws = _logs_ws()
    all_v = [list(r) for r in _bulk_logs_values()]
    if len(all_v) < 2:
        return
    hdr = all_v[0]
    if not _headers_match(hdr, LOGS_HEADERS):
        return
    ui = LOGS_HEADERS.index("uid")
    want = str(uid).strip().lower()
    to_del = [
        r_i for r_i, row in enumerate(all_v[1:], start=2) if len(row) > ui and str(row[ui]).strip().lower() == want
    ]
    if not to_del:
        return
    for r_i in sorted(to_del, reverse=True):
        ws.delete_rows(r_i)
    _invalidate_read_caches()


def ensure_default_users() -> None:
    init_db()
    _migrate_legacy_uids()
    keep_uids: tuple[str, ...] = STUDENT_UIDS + (TEACHER_UID,)

    if not _find_student_row(TEACHER_UID):
        _append_student_row(_default_student_row(TEACHER_UID, DEFAULT_PASSWORD, "teacher"))
    else:
        r = _find_student_row(TEACHER_UID)
        if r:
            hi = _student_header_index()
            cells = r[1]
            cells[hi["role"]] = "teacher"
            _write_student_row(r[0], cells)

    for uid in STUDENT_UIDS:
        if not _find_student_row(uid):
            _append_student_row(_default_student_row(uid, DEFAULT_PASSWORD, "student"))

    rows = _students_values()
    if len(rows) < 2:
        return
    hi = _student_header_index()
    for r_i, row in list(enumerate(rows[1:], start=2))[::-1]:
        if not row:
            continue
        u = str(row[0]).strip().lower()
        if u and u not in keep_uids:
            _delete_logs_for_uid(u)
            _delete_student_row(r_i)


@st.cache_data(ttl=60)
def get_user(uid: str) -> dict[str, Any] | None:
    if uid is None:
        return None
    norm = str(uid).strip().lower()
    if not norm:
        return None
    init_db()
    hit = _find_student_row(norm)
    if not hit:
        return None
    cells = hit[1]
    hi = _student_header_index()
    return {
        "uid": cells[hi["uid"]],
        "password": cells[hi["password"]],
        "role": cells[hi["role"]],
    }


def authenticate(uid: str, pw: str) -> dict[str, Any] | None:
    user = get_user(uid)
    if not user:
        return None
    stored = str(user.get("password") or "").strip()
    attempt = str(pw or "").strip()
    if not stored or not attempt:
        return None
    try:
        ok = secrets.compare_digest(stored, attempt)
    except (TypeError, ValueError):
        ok = stored == attempt
    if not ok:
        return None
    return user


def update_password(uid: str, new_password: str) -> bool:
    pwd = (new_password or "").strip()
    if not pwd:
        return False
    norm = str(uid).strip().lower()
    if not norm:
        return False
    init_db()
    hit = _find_student_row(norm)
    if not hit:
        return False
    hi = _student_header_index()
    cells = hit[1]
    cells[hi["password"]] = pwd
    _write_student_row(hit[0], cells)
    return True


@st.cache_data(ttl=60)
def list_users() -> list[dict[str, Any]]:
    init_db()
    out: list[dict[str, Any]] = []
    rows = _students_values()
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        hi = _student_header_index()
        while len(row) < len(STUDENTS_HEADERS):
            row.append("")
        out.append({"uid": row[hi["uid"]], "role": row[hi["role"]]})
    out.sort(key=lambda d: d["uid"])
    return out


@st.cache_data(ttl=60)
def list_user_credentials() -> list[dict[str, Any]]:
    init_db()
    rows = _students_values()
    hi = _student_header_index()
    acc: list[dict[str, Any]] = []
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        while len(row) < len(STUDENTS_HEADERS):
            row.append("")
        role = row[hi["role"]]
        acc.append(
            {
                "uid": row[hi["uid"]],
                "password": row[hi["password"]],
                "role": role,
            }
        )
    acc.sort(key=lambda d: (0 if d.get("role") == "teacher" else 1, d["uid"]))
    return acc


def _progress_from_row(cells: list[str]) -> dict[str, int]:
    hi = _student_header_index()
    raw = cells[hi["progress_json"]] if len(cells) > hi["progress_json"] else "{}"
    try:
        data = json.loads(raw or "{}")
    except (TypeError, ValueError):
        data = {}
    out: dict[str, int] = {}
    for k, v in data.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    return out


def _save_progress_on_row(row_1based: int, cells: list[str], prog: dict[str, int]) -> None:
    hi = _student_header_index()
    while len(cells) < len(STUDENTS_HEADERS):
        cells.append("")
    cells[hi["progress_json"]] = json.dumps(prog, ensure_ascii=False)
    _write_student_row(row_1based, cells)


def seed_progress_if_missing(uid: str, defaults: dict[str, int]) -> dict[str, int]:
    init_db()
    hit = _find_student_row(uid)
    hi = _student_header_index()
    if not hit:
        return dict(defaults)
    row_i, cells = hit
    cur = _progress_from_row(cells)
    changed = False
    for unit, value in defaults.items():
        if unit not in cur:
            cur[unit] = int(value)
            changed = True
    if changed:
        _save_progress_on_row(row_i, cells, cur)
    return cur


def update_progress(uid: str, ncs_unit: str, value: int) -> None:
    init_db()
    hit = _find_student_row(uid)
    if not hit:
        return
    row_i, cells = hit
    cur = _progress_from_row(cells)
    cur[str(ncs_unit)] = int(value)
    _save_progress_on_row(row_i, cells, cur)


def _next_log_id() -> int:
    all_v = [list(r) for r in _bulk_logs_values()]
    mx = 0
    if len(all_v) >= 2:
        ii = LOGS_HEADERS.index("id")
        for row in all_v[1:]:
            if len(row) > ii:
                mx = max(mx, _parse_int_cell(row[ii], default=0))
    return mx + 1


def add_log(
    *,
    uid: str,
    date: str,
    ncs_unit: str,
    bsr: str,
    image_note: str | None = None,
    image_b64: str | None = None,
    audio_note: str | None = None,
    ncs_term_ratio: float | None = None,
) -> int:
    init_db()
    uid = str(uid).strip().lower()
    new_id = _next_log_id()
    created = datetime.datetime.now().isoformat(timespec="seconds")
    ratio_s = "" if ncs_term_ratio is None else _cell_str(ncs_term_ratio)
    img_b64_safe, img_note_safe = _coerce_log_image_for_sheet(image_b64, image_note)
    row = [
        _cell_str(new_id),
        _truncate_sheet_cell(_cell_str(uid)),
        _truncate_sheet_cell(_cell_str(date)),
        _truncate_sheet_cell(_cell_str(ncs_unit)),
        _truncate_sheet_cell(_cell_str(bsr)),
        img_note_safe,
        img_b64_safe,
        _truncate_sheet_cell(_cell_str(audio_note)),
        ratio_s,
        _truncate_sheet_cell(_cell_str(created)),
    ]
    _logs_ws().append_row(row, value_input_option="RAW")
    _invalidate_read_caches()
    return new_id


def _row_to_log_dict(row: list[str]) -> dict[str, Any]:
    while len(row) < len(LOGS_HEADERS):
        row.append("")
    d: dict[str, Any] = {}
    for h in LOGS_HEADERS:
        raw = row[LOGS_HEADERS.index(h)]
        d[h] = "" if raw is None else str(raw)
    d["id"] = _parse_int_cell(d.get("id"), default=0)
    d["uid"] = str(d.get("uid") or "").strip().lower()
    d["date"] = str(d.get("date") or "").strip()[:32]
    d["ncs_unit"] = str(d.get("ncs_unit") or "").strip()
    d["bsr"] = str(d.get("bsr") or "")
    tr = _parse_float_cell(d.get("ncs_term_ratio"))
    d["ncs_term_ratio"] = tr
    if not str(d.get("image_note") or "").strip():
        d["image_note"] = None
    else:
        d["image_note"] = str(d["image_note"])
    if not str(d.get("image_b64") or "").strip():
        d["image_b64"] = None
    else:
        d["image_b64"] = str(d["image_b64"])
    if not str(d.get("audio_note") or "").strip():
        d["audio_note"] = None
    else:
        d["audio_note"] = str(d["audio_note"])
    d["created_at"] = str(d.get("created_at") or "").strip()
    return d


@st.cache_data(ttl=60)
def list_logs(uid: str) -> list[dict[str, Any]]:
    init_db()
    all_v = [list(r) for r in _bulk_logs_values()]
    if len(all_v) < 2:
        return []
    if not _headers_match(all_v[0], LOGS_HEADERS):
        return []
    want = str(uid).strip().lower()
    ui = LOGS_HEADERS.index("uid")
    out: list[dict[str, Any]] = []
    for row in all_v[1:]:
        if len(row) <= ui:
            continue
        if str(row[ui]).strip().lower() != want:
            continue
        out.append(_row_to_log_dict(list(row)))

    def sort_key(r: dict[str, Any]) -> tuple[str, int]:
        ca = str(r.get("created_at") or "").strip()
        if not ca:
            ca = "1970-01-01T00:00:00"
        return (ca, _parse_int_cell(r.get("id"), default=0))

    out.sort(key=sort_key, reverse=True)
    return out


def delete_log(uid: str, log_id: int) -> None:
    init_db()
    ws = _logs_ws()
    all_v = [list(r) for r in _bulk_logs_values()]
    if len(all_v) < 2:
        return
    if not _headers_match(all_v[0], LOGS_HEADERS):
        return
    ii = LOGS_HEADERS.index("id")
    ui = LOGS_HEADERS.index("uid")
    want_uid = str(uid).strip().lower()
    target = _parse_int_cell(log_id, default=-1)
    if target < 0:
        return
    for r_i, row in enumerate(all_v[1:], start=2):
        if len(row) <= max(ii, ui):
            continue
        rid = _parse_int_cell(row[ii], default=-1)
        if rid == target and str(row[ui]).strip().lower() == want_uid:
            ws.delete_rows(r_i)
            _invalidate_read_caches()
            return


def clear_logs(uid: str) -> None:
    init_db()
    _delete_logs_for_uid(uid)
    hit = _find_student_row(uid)
    if hit:
        hi = _student_header_index()
        cells = hit[1]
        cells[hi["progress_json"]] = "{}"
        _write_student_row(hit[0], cells)


def _next_researcher_id() -> int:
    all_v = [list(r) for r in _bulk_researcher_values()]
    mx = 0
    if len(all_v) >= 2:
        for row in all_v[1:]:
            if row and str(row[0]).strip():
                mx = max(mx, _parse_int_cell(row[0], default=0))
    return mx + 1


def add_researcher_log(*, log_date: str, note: str) -> int:
    init_db()
    new_id = _next_researcher_id()
    created = datetime.datetime.now().isoformat(timespec="seconds")
    _researcher_ws().append_row(
        [_cell_str(new_id), _cell_str(log_date), _cell_str(note), _cell_str(created)],
        value_input_option="RAW",
    )
    _invalidate_read_caches()
    return new_id


@st.cache_data(ttl=60)
def list_researcher_logs() -> list[dict[str, Any]]:
    init_db()
    all_v = [list(r) for r in _bulk_researcher_values()]
    if len(all_v) < 2:
        return []
    if not _headers_match(all_v[0], RESEARCHER_HEADERS):
        return []
    out: list[dict[str, Any]] = []
    for row in all_v[1:]:
        if not row or not row[0].strip():
            continue
        while len(row) < len(RESEARCHER_HEADERS):
            row.append("")
        rid = _parse_int_cell(row[0], default=0)
        out.append(
            {
                "id": rid,
                "log_date": str(row[1] or "").strip(),
                "note": str(row[2] or ""),
                "created_at": str(row[3] or "").strip(),
            }
        )
    out.sort(key=lambda r: (str(r.get("log_date") or ""), _parse_int_cell(r.get("id"), default=0)), reverse=True)
    return out


def save_portfolio_comment(
    uid: str, comment_text: str, reflection_level: str = "", *, confirmed: bool = True
) -> None:
    init_db()
    hit = _find_student_row(uid)
    if not hit:
        return
    row_i, cells = hit
    hi = _student_header_index()
    while len(cells) < len(STUDENTS_HEADERS):
        cells.append("")
    now = datetime.datetime.now().isoformat(timespec="seconds")
    cells[hi["portfolio_comment_text"]] = comment_text
    cells[hi["portfolio_reflection_level"]] = reflection_level or ""
    cells[hi["portfolio_updated_at"]] = now
    cells[hi["portfolio_is_confirmed"]] = "1" if confirmed else "0"
    _write_student_row(row_i, cells)


@st.cache_data(ttl=60)
def get_portfolio_comment(uid: str) -> dict[str, Any] | None:
    init_db()
    hit = _find_student_row(uid)
    if not hit:
        return None
    cells = hit[1]
    hi = _student_header_index()
    while len(cells) < len(STUDENTS_HEADERS):
        cells.append("")
    text = cells[hi["portfolio_comment_text"]] or ""
    if not str(text).strip() and not str(cells[hi["portfolio_updated_at"]] or "").strip():
        return None
    ic = _parse_int_cell(cells[hi["portfolio_is_confirmed"]], default=0)
    return {
        "uid": cells[hi["uid"]],
        "comment_text": text,
        "reflection_level": cells[hi["portfolio_reflection_level"]] or "",
        "updated_at": cells[hi["portfolio_updated_at"]] or "",
        "is_confirmed": ic,
    }


def get_confirmed_portfolio_comment(uid: str) -> dict[str, Any] | None:
    row = get_portfolio_comment(uid)
    if not row:
        return None
    if int(row.get("is_confirmed") or 0):
        return row
    return None


EMPTY_PROFILE: dict[str, Any] = {
    "full_name": "",
    "birth_date": "",
    "email": "",
    "phone": "",
    "motto": "",
    "photo_b64": "",
    "educations": [],
    "careers": [],
    "certificates": [],
    "awards": [],
    "tech_stack": [],
}


def _safe_json_loads(s: Any) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


@st.cache_data(ttl=60)
def get_student_profile(uid: str) -> dict[str, Any]:
    init_db()
    hit = _find_student_row(uid)
    if not hit:
        return {**EMPTY_PROFILE, "uid": uid, "updated_at": ""}
    cells = hit[1]
    hi = _student_header_index()
    while len(cells) < len(STUDENTS_HEADERS):
        cells.append("")
    return {
        "uid": cells[hi["uid"]],
        "full_name": cells[hi["full_name"]] or "",
        "birth_date": cells[hi["birth_date"]] or "",
        "email": cells[hi["email"]] or "",
        "phone": cells[hi["phone"]] or "",
        "motto": cells[hi["motto"]] or "",
        "photo_b64": cells[hi["photo_b64"]] or "",
        "educations": _safe_json_loads(cells[hi["educations_json"]]) or [],
        "careers": _safe_json_loads(cells[hi["careers_json"]]) or [],
        "certificates": _safe_json_loads(cells[hi["certificates_json"]]) or [],
        "awards": _safe_json_loads(cells[hi["awards_json"]]) or [],
        "tech_stack": _safe_json_loads(cells[hi["tech_stack_json"]]) or [],
        "updated_at": cells[hi["profile_updated_at"]] or "",
    }


def save_student_profile(uid: str, profile: dict[str, Any]) -> None:
    init_db()
    hit = _find_student_row(uid)
    hi = _student_header_index()
    now = datetime.datetime.now().isoformat(timespec="seconds")
    payload = {
        "full_name": (profile.get("full_name") or "").strip(),
        "birth_date": (profile.get("birth_date") or "").strip(),
        "email": (profile.get("email") or "").strip(),
        "phone": (profile.get("phone") or "").strip(),
        "motto": (profile.get("motto") or "").strip(),
        "photo_b64": profile.get("photo_b64") or "",
        "educations_json": json.dumps(profile.get("educations") or [], ensure_ascii=False),
        "careers_json": json.dumps(profile.get("careers") or [], ensure_ascii=False),
        "certificates_json": json.dumps(profile.get("certificates") or [], ensure_ascii=False),
        "awards_json": json.dumps(profile.get("awards") or [], ensure_ascii=False),
        "tech_stack_json": json.dumps(profile.get("tech_stack") or [], ensure_ascii=False),
    }
    if hit:
        row_i, cells = hit
        while len(cells) < len(STUDENTS_HEADERS):
            cells.append("")
    else:
        row_i = None
        cells = _default_student_row(uid, DEFAULT_PASSWORD, "student")

    cells[hi["full_name"]] = payload["full_name"]
    cells[hi["birth_date"]] = payload["birth_date"]
    cells[hi["email"]] = payload["email"]
    cells[hi["phone"]] = payload["phone"]
    cells[hi["motto"]] = payload["motto"]
    cells[hi["photo_b64"]] = str(payload["photo_b64"])
    cells[hi["educations_json"]] = payload["educations_json"]
    cells[hi["careers_json"]] = payload["careers_json"]
    cells[hi["certificates_json"]] = payload["certificates_json"]
    cells[hi["awards_json"]] = payload["awards_json"]
    cells[hi["tech_stack_json"]] = payload["tech_stack_json"]
    cells[hi["profile_updated_at"]] = now

    if row_i is not None:
        _write_student_row(row_i, cells)
    else:
        _append_student_row(cells)


def clear_student_profile(uid: str) -> None:
    init_db()
    hit = _find_student_row(uid)
    if not hit:
        return
    row_i, cells = hit
    hi = _student_header_index()
    while len(cells) < len(STUDENTS_HEADERS):
        cells.append("")
    for key in (
        "full_name",
        "birth_date",
        "email",
        "phone",
        "motto",
        "photo_b64",
        "educations_json",
        "careers_json",
        "certificates_json",
        "awards_json",
        "tech_stack_json",
        "profile_updated_at",
    ):
        cells[hi[key]] = ""
    _write_student_row(row_i, cells)


_DEMO_LOG_TEMPLATES: list[tuple[str, str]] = [
    (
        "전자부품장착",
        "[배경] 인두 온도 350°C에서 0805 SMD 저항·콘덴서 다수를 PCB에 부착하는 실습을 진행하였다.\n"
        "[해결] 솔더 윅으로 브리지 부위를 정리하고, 부품 극성과 정렬 상태를 멀티미터·확대경으로 점검하였다.\n"
        "[성과] 납땜 품질이 향상되었고, 쇼트 발생 시 원인을 인두 각도·플럭스 양으로 좁혀 분석할 수 있었음을 알게 됨.",
    ),
    (
        "전자회로조립",
        "[배경] 브레드보드 위에 OPAMP 비반전 증폭기를 구성해 1kHz 사인파 입력 시 출력 파형을 측정하였다.\n"
        "[해결] 증폭률 계산 후 R1·R2 저항값을 변경하며 오실로스코프로 파형을 비교하였다.\n"
        "[성과] 이론 게인과 실측 게인의 오차 원인을 OPAMP 슬루레이트·전원전압으로 추적함.",
    ),
    (
        "PCB설계",
        "[배경] OrCAD에서 5V 레귤레이터 PCB의 부품 배치와 GND 베타플레인을 검토하였다.\n"
        "[해결] DRC 위반(클리어런스·드릴) 항목을 항목별로 수정하고, 거버 출력으로 최종 검증함.\n"
        "[성과] 라우팅 오류 발생 시 인접 비아·패드 간 간격을 우선 점검해야 함을 이해함.",
    ),
    (
        "마이크로컨트롤러",
        "[배경] Arduino UNO에서 PWM 출력을 활용해 LED 밝기 제어를 구현하고, UART로 디버그 로그를 출력함.\n"
        "[해결] analogWrite 듀티비를 0~255 단계로 변경하며 시리얼 모니터로 측정값을 비교함.\n"
        "[성과] 듀티비 변화에 따른 평균 전압을 멀티미터로 확인했고, PWM 주파수 영향까지 고민하게 됨.",
    ),
    (
        "PLC제어",
        "[배경] LS산전 XGB PLC에서 정역 전동기 제어 래더 회로를 구성하고 인터록을 적용함.\n"
        "[해결] 정·역 접점이 동시에 ON되지 않도록 b접점 인터록을 배치하고 시운전으로 검증함.\n"
        "[성과] 모터 보호 차원에서 인터록·OLR 신호 연결의 중요성을 깨달음.",
    ),
    (
        "센서응용",
        "[배경] 근접센서(NPN)를 24V DC PLC 입력에 연결해 컨베이어 위치 감지 기능을 구현함.\n"
        "[해결] 센서 출력선·풀업 저항·차폐 케이블 사용 여부를 점검하고 노이즈 영향을 분석함.\n"
        "[성과] 4-20mA 아날로그 신호와 디지털 신호의 차이를 실측을 통해 이해함.",
    ),
    (
        "전기안전",
        "[배경] 전동기 정비 전 LOTO(잠금·표시) 절차를 적용하고 보호구를 착용한 뒤 작업함.\n"
        "[해결] 전원 차단·검전기 점검·절연저항계(메거)로 절연 상태를 측정함.\n"
        "[성과] 작업 전 위험요인 파악과 단계별 안전조치의 중요성을 다시 한번 되새김.",
    ),
    (
        "산업통신",
        "[배경] Modbus RTU(RS-485)로 PLC 마스터-인버터 슬레이브 통신을 구성하고 주소·전송속도를 설정함.\n"
        "[해결] 프레임 오류·타임아웃 발생 시 종단저항(120Ω)·접지·결선을 점검하며 원인을 좁힘.\n"
        "[성과] 프로토콜 분석기를 통한 프레임 검증의 효과를 확인함.",
    ),
    (
        "임베디드하드웨어설계",
        "[배경] STM32 보드의 전원·클럭·디버그 포트 회로를 회로도 단위로 검토함.\n"
        "[해결] 디커플링 캐패시터 위치·접지 폴리곤·크리스탈 매칭 회로를 보완함.\n"
        "[성과] MCU 안정 동작을 위한 전원 무결성·EMI 대책의 기본 원칙을 이해함.",
    ),
    (
        "전자회로설계",
        "[배경] LTspice 시뮬레이션으로 1차 RC 저역통과 필터의 컷오프 주파수를 검증함.\n"
        "[해결] 저항·콘덴서 값을 바꿔 보드(Bode) 플롯을 비교하고 -3dB 점을 측정함.\n"
        "[성과] 이론과 실험 결과의 일치 여부를 정량적으로 분석할 수 있게 됨.",
    ),
]


def _purge_logs_outside_test_period() -> int:
    init_db()
    ws = _logs_ws()
    all_v = [list(r) for r in _bulk_logs_values()]
    if len(all_v) < 2:
        return 0
    if not _headers_match(all_v[0], LOGS_HEADERS):
        return 0
    di = LOGS_HEADERS.index("date")
    start_s = TEST_PERIOD_START.isoformat()
    end_s = TEST_PERIOD_END.isoformat()
    to_del: list[int] = []
    for r_i, row in enumerate(all_v[1:], start=2):
        if len(row) <= di:
            continue
        ds = str(row[di]).strip()[:10]
        if ds < start_s[:10] or ds > end_s[:10]:
            to_del.append(r_i)
    for r_i in sorted(to_del, reverse=True):
        ws.delete_rows(r_i)
    if to_del:
        _invalidate_read_caches()
    return len(to_del)


def seed_demo_logs_if_empty(*, force_refresh: bool = False) -> int:
    init_db()
    _purge_logs_outside_test_period()
    today = app_today()
    weekdays = [d for d in test_period_weekdays() if d <= today]
    if not weekdays:
        return 0
    rnd = random.Random(20260511)
    created = 0
    start_s = TEST_PERIOD_START.isoformat()
    end_s = TEST_PERIOD_END.isoformat()
    for idx, uid in enumerate(STUDENT_UIDS):
        if force_refresh:
            _delete_logs_for_uid(uid)
            cur_n = 0
        else:
            cur_n = sum(
                1
                for r in list_logs(uid)
                if start_s <= str(r.get("date") or "")[:10] <= end_s
            )
        if cur_n > 0:
            continue
        n_logs = max(1, min(3, len(weekdays), 1 + (idx % 3)))
        chosen_dates = sorted(rnd.sample(weekdays, n_logs))
        chosen_units = rnd.sample(_DEMO_LOG_TEMPLATES, n_logs)
        for d, (unit, bsr) in zip(chosen_dates, chosen_units):
            add_log(
                uid=uid,
                date=d.isoformat(),
                ncs_unit=unit,
                bsr=bsr,
                image_note=("사진 업로드됨" if rnd.random() < 0.6 else None),
                audio_note=("음성 녹음됨" if rnd.random() < 0.25 else None),
                ncs_term_ratio=round(rnd.uniform(55.0, 92.0), 1),
            )
            hit = _find_student_row(uid)
            if hit:
                row_i, cells = hit
                cur_p = _progress_from_row(cells)
                bump = rnd.randint(8, 18)
                cur_p[unit] = min(100, int(cur_p.get(unit, 0)) + bump)
                _save_progress_on_row(row_i, cells, cur_p)
            created += 1
    return created
