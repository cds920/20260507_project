"""NCS 포트폴리오 - SQLite DB (앱 시작 시 init_db 1회 호출, check_same_thread 지원)"""
import datetime
import json
import os
import random
import re
import secrets
import sqlite3
import tempfile
from pathlib import Path
from typing import Any


def student_number(uid: str) -> int:
    """학생 UID에서 번호를 추출.

    - 'yongsan1' → 1, 'yongsan10' → 10
    - 'S01' → 1 (과거 호환)
    - 비학생/비표준 형식은 999
    """
    m = re.search(r"(\d+)\s*$", str(uid or ""))
    if not m:
        return 999
    try:
        return int(m.group(1))
    except (TypeError, ValueError):
        return 999


def student_label(uid: str) -> str:
    """학생용 표시 라벨. 'yongsan1' → '1번 도제생'."""
    n = student_number(uid)
    return f"{n}번 도제생" if n != 999 else str(uid)

# ───────────────────────────────────────────────────────────────────
# 사용자 계정 체계 (2026-05 개편)
#   - 교사: teacher (초기 비밀번호 1234)
#   - 학생: yongsan1 ~ yongsan10 (초기 비밀번호 1234)
#   - 모든 아이디는 소문자 영문/숫자. 로그인은 대소문자 무시(.lower())
# ───────────────────────────────────────────────────────────────────
TEACHER_UID: str = "teacher"
DEFAULT_PASSWORD: str = "1234"
STUDENT_COUNT: int = 10
STUDENT_UIDS: tuple[str, ...] = tuple(f"yongsan{i}" for i in range(1, STUDENT_COUNT + 1))

# 과거 운영에서 사용한 UID → 새로운 UID 마이그레이션 매핑
#   admin → teacher, S01..S10 → yongsan1..yongsan10
_LEGACY_UID_MAP: dict[str, str] = {"admin": TEACHER_UID}
for _i in range(1, STUDENT_COUNT + 1):
    _LEGACY_UID_MAP[f"S{_i:02d}"] = f"yongsan{_i}"
del _i

# ───────────────────────────────────────────────────────────────────
# 실전 테스트 기간 (2026-05-11 월 ~ 2026-05-29 금)
# ───────────────────────────────────────────────────────────────────
TEST_PERIOD_START: datetime.date = datetime.date(2026, 5, 11)
TEST_PERIOD_END: datetime.date = datetime.date(2026, 5, 29)


def test_period_weekdays() -> list[datetime.date]:
    """테스트 기간 내 평일(월~금) 목록을 날짜 오름차순으로 반환."""
    days: list[datetime.date] = []
    d = TEST_PERIOD_START
    while d <= TEST_PERIOD_END:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def app_today() -> datetime.date:
    """
    앱 전체에서 사용하는 '오늘'.
    실제 오늘이 테스트 기간 시작일 이전이면 시작일(2026-05-11)을 반환,
    종료일 이후면 종료일(2026-05-29)을 반환,
    기간 내라면 실제 오늘을 그대로 사용한다.
    """
    real_today = datetime.date.today()
    if real_today < TEST_PERIOD_START:
        return TEST_PERIOD_START
    if real_today > TEST_PERIOD_END:
        return TEST_PERIOD_END
    return real_today

# Streamlit Cloud 등 읽기 전용 환경: 임시 폴더 사용 (재시작 시 데이터 초기화)
_DEFAULT_PATH = Path(__file__).resolve().parent / "data.sqlite3"
try:
    _writable = os.access(_DEFAULT_PATH.parent, os.W_OK)
except OSError:
    _writable = False
if _writable:
    DB_PATH = _DEFAULT_PATH
else:
    DB_PATH = Path(tempfile.gettempdir()) / "ncs_portfolio_data.sqlite3"
_db_initialized = False


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    """앱 시작 시 한 번만 호출. 테이블 생성."""
    global _db_initialized
    if _db_initialized:
        return
    with _connect() as con:
        con.executescript(
            """
            PRAGMA journal_mode=WAL;
            PRAGMA synchronous=NORMAL;

            CREATE TABLE IF NOT EXISTS users (
              uid TEXT PRIMARY KEY,
              password TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'student'
            );

            CREATE TABLE IF NOT EXISTS logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              uid TEXT NOT NULL,
              date TEXT NOT NULL,
              ncs_unit TEXT NOT NULL,
              bsr TEXT NOT NULL,
              image_note TEXT,
              image_b64 TEXT,
              audio_note TEXT,
              ncs_term_ratio REAL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(uid) REFERENCES users(uid)
            );

            CREATE TABLE IF NOT EXISTS progress (
              uid TEXT NOT NULL,
              ncs_unit TEXT NOT NULL,
              value INTEGER NOT NULL,
              PRIMARY KEY(uid, ncs_unit),
              FOREIGN KEY(uid) REFERENCES users(uid)
            );

            CREATE TABLE IF NOT EXISTS researcher_logs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              log_date TEXT NOT NULL,
              note TEXT NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS portfolio_comments (
              uid TEXT PRIMARY KEY,
              comment_text TEXT NOT NULL,
              reflection_level TEXT,
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              is_confirmed INTEGER NOT NULL DEFAULT 0,
              FOREIGN KEY(uid) REFERENCES users(uid)
            );

            CREATE TABLE IF NOT EXISTS student_profiles (
              uid TEXT PRIMARY KEY,
              full_name TEXT,
              birth_date TEXT,
              email TEXT,
              phone TEXT,
              motto TEXT,
              photo_b64 TEXT,
              educations_json TEXT,
              careers_json TEXT,
              certificates_json TEXT,
              awards_json TEXT,
              tech_stack_json TEXT,
              updated_at TEXT NOT NULL DEFAULT (datetime('now')),
              FOREIGN KEY(uid) REFERENCES users(uid)
            );
            """
        )
        try:
            con.execute("SELECT ncs_term_ratio FROM logs LIMIT 1")
        except sqlite3.OperationalError:
            con.execute("ALTER TABLE logs ADD COLUMN ncs_term_ratio REAL")
        try:
            con.execute("SELECT image_b64 FROM logs LIMIT 1")
        except sqlite3.OperationalError:
            con.execute("ALTER TABLE logs ADD COLUMN image_b64 TEXT")
        try:
            con.execute("SELECT is_confirmed FROM portfolio_comments LIMIT 1")
        except sqlite3.OperationalError:
            con.execute("ALTER TABLE portfolio_comments ADD COLUMN is_confirmed INTEGER NOT NULL DEFAULT 0")
            con.execute("UPDATE portfolio_comments SET is_confirmed=1")

        # users: password 컬럼 보장 (구 스키마 pw → password 이전)
        _ensure_users_password_column(con)
    _db_initialized = True


def _users_table_columns(con: sqlite3.Connection) -> set[str]:
    return {str(r[1]) for r in con.execute("PRAGMA table_info(users)").fetchall()}


def _ensure_users_password_column(con: sqlite3.Connection) -> None:
    """password 컬럼이 없으면 추가하고, 기존 pw 값이 있으면 복사한다."""
    cols = _users_table_columns(con)
    if "password" not in cols:
        con.execute("ALTER TABLE users ADD COLUMN password TEXT")
        cols = _users_table_columns(con)
    if "pw" in cols:
        con.execute(
            """
            UPDATE users
            SET password = pw
            WHERE password IS NULL OR TRIM(COALESCE(password, '')) = ''
            """
        )
    con.execute(
        """
        UPDATE users
        SET password = ?
        WHERE password IS NULL OR TRIM(COALESCE(password, '')) = ''
        """,
        (DEFAULT_PASSWORD,),
    )


def _sync_legacy_pw_column(con: sqlite3.Connection, uid: str, password_plain: str) -> None:
    """구 컬럼 pw가 남아 있으면 password와 동기화."""
    if "pw" in _users_table_columns(con):
        con.execute("UPDATE users SET pw=? WHERE uid=?", (password_plain, uid))


def _migrate_legacy_uids(con: sqlite3.Connection) -> None:
    """과거 UID(admin / S01~S10)를 새 UID(teacher / yongsan1~yongsan10)로 일괄 이전.

    - users 테이블의 PK(uid)와 그를 참조하는 모든 테이블(logs, progress,
      portfolio_comments, student_profiles)의 uid 컬럼을 한꺼번에 업데이트한다.
    - 대상 새 UID가 이미 존재하면 충돌을 피하기 위해 그 행을 건너뛴다.
    - 이 작업은 멱등(idempotent)하며, 이미 마이그레이션된 환경에서는 아무것도 하지 않는다.
    """
    if not _LEGACY_UID_MAP:
        return
    fk_tables = ("logs", "progress", "portfolio_comments", "student_profiles")
    for old_uid, new_uid in _LEGACY_UID_MAP.items():
        old_row = con.execute("SELECT uid FROM users WHERE uid=?", (old_uid,)).fetchone()
        if not old_row:
            continue
        new_row = con.execute("SELECT uid FROM users WHERE uid=?", (new_uid,)).fetchone()
        if new_row:
            # 새 UID가 이미 존재 → 옛 UID 데이터를 폐기하여 PK 충돌 회피
            for tbl in fk_tables:
                con.execute(f"DELETE FROM {tbl} WHERE uid=?", (old_uid,))
            con.execute("DELETE FROM users WHERE uid=?", (old_uid,))
            continue
        # FK를 먼저 옮겨놓아야 PK 변경 시 고아 데이터가 생기지 않음
        for tbl in fk_tables:
            con.execute(f"UPDATE {tbl} SET uid=? WHERE uid=?", (new_uid, old_uid))
        # PK 변경과 동시에 초기 비밀번호를 강제 리셋한다.
        # (예: 옛 admin의 'admin123' → 새 teacher의 '1234')
        con.execute(
            "UPDATE users SET uid=?, password=? WHERE uid=?",
            (new_uid, DEFAULT_PASSWORD, old_uid),
        )
        _sync_legacy_pw_column(con, new_uid, DEFAULT_PASSWORD)


def ensure_default_users() -> None:
    init_db()
    with _connect() as con:
        # 1) 과거 UID(admin/S0X)을 새 UID(teacher/yongsanX)로 안전하게 이전
        _migrate_legacy_uids(con)

        # 2) 교사 계정 보장 (없으면 생성, 있으면 role 보정)
        teacher_row = con.execute(
            "SELECT uid FROM users WHERE uid=?", (TEACHER_UID,)
        ).fetchone()
        if teacher_row:
            con.execute(
                "UPDATE users SET role=? WHERE uid=?", ("teacher", TEACHER_UID)
            )
        else:
            con.execute(
                "INSERT INTO users(uid, password, role) VALUES(?,?,?)",
                (TEACHER_UID, DEFAULT_PASSWORD, "teacher"),
            )

        # 3) 학생 계정 보장 (없으면 생성)
        for uid in STUDENT_UIDS:
            con.execute(
                "INSERT OR IGNORE INTO users(uid, password, role) VALUES(?,?,?)",
                (uid, DEFAULT_PASSWORD, "student"),
            )

        # 4) 정원 외 학생(과거 S11 등) 및 그 데이터를 정리
        keep_uids: tuple[str, ...] = STUDENT_UIDS + (TEACHER_UID,)
        placeholders = ",".join(["?"] * len(keep_uids))
        con.execute(
            f"DELETE FROM users WHERE uid NOT IN ({placeholders})",
            keep_uids,
        )
        for tbl in ("logs", "progress", "portfolio_comments", "student_profiles"):
            con.execute(
                f"DELETE FROM {tbl} WHERE uid NOT IN ({placeholders})",
                keep_uids,
            )

    # 배포용: 더미/데모 일지 자동 시드는 사용하지 않습니다.
    # (이전에는 `seed_demo_logs_if_empty()`로 학생 일지가 비어 있으면 1~3건을 자동 생성했지만,
    #  실제 학생들이 깨끗한 상태에서 첫 일지를 직접 작성하도록 비활성화했습니다.)


def get_user(uid: str) -> dict[str, Any] | None:
    """UID로 사용자 조회. 입력은 자동으로 소문자·trim 처리하여 대소문자 구분을 제거한다."""
    if uid is None:
        return None
    norm = str(uid).strip().lower()
    if not norm:
        return None
    with _connect() as con:
        row = con.execute(
            "SELECT uid, password, role FROM users WHERE uid=?", (norm,)
        ).fetchone()
        return dict(row) if row else None


def authenticate(uid: str, pw: str) -> dict[str, Any] | None:
    """대소문자 무시 UID + 비밀번호 검증. 성공 시 사용자 dict, 실패 시 None."""
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
    """비밀번호를 갱신한다. 성공 시 True, 사용자 미존재 또는 빈 비밀번호면 False."""
    pwd = (new_password or "").strip()
    if not pwd:
        return False
    norm = str(uid).strip().lower()
    if not norm:
        return False
    with _connect() as con:
        cur = con.execute(
            "UPDATE users SET password=? WHERE uid=?",
            (pwd, norm),
        )
        if (cur.rowcount or 0) <= 0:
            return False
        _sync_legacy_pw_column(con, norm, pwd)
        return True


def list_users() -> list[dict[str, Any]]:
    with _connect() as con:
        rows = con.execute("SELECT uid, role FROM users ORDER BY uid").fetchall()
        return [dict(r) for r in rows]


def list_user_credentials() -> list[dict[str, Any]]:
    """교사용: 모든 사용자(특히 학생)의 UID/비밀번호/role 일괄 조회.

    교사가 학생의 분실된 비밀번호를 확인·안내할 수 있도록 평문 비밀번호를 노출한다.
    (교내 폐쇄망 운영 환경 가정. 외부 노출 시에는 비밀번호 정책 강화 필요.)
    """
    with _connect() as con:
        rows = con.execute(
            "SELECT uid, password, role FROM users ORDER BY role DESC, uid"
        ).fetchall()
        return [dict(r) for r in rows]


def seed_progress_if_missing(uid: str, defaults: dict[str, int]) -> dict[str, int]:
    with _connect() as con:
        for unit, value in defaults.items():
            con.execute(
                "INSERT OR IGNORE INTO progress(uid, ncs_unit, value) VALUES(?,?,?)",
                (uid, unit, int(value)),
            )
        rows = con.execute(
            "SELECT ncs_unit, value FROM progress WHERE uid=?",
            (uid,),
        ).fetchall()
        return {r["ncs_unit"]: int(r["value"]) for r in rows}


def update_progress(uid: str, ncs_unit: str, value: int) -> None:
    with _connect() as con:
        con.execute(
            """
            INSERT INTO progress(uid, ncs_unit, value) VALUES(?,?,?)
            ON CONFLICT(uid, ncs_unit) DO UPDATE SET value=excluded.value
            """,
            (uid, ncs_unit, int(value)),
        )


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
    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO logs(uid, date, ncs_unit, bsr, image_note, image_b64, audio_note, ncs_term_ratio)
            VALUES(?,?,?,?,?,?,?,?)
            """,
            (uid, date, ncs_unit, bsr, image_note, image_b64, audio_note, ncs_term_ratio),
        )
        return int(cur.lastrowid)


def list_logs(uid: str) -> list[dict[str, Any]]:
    with _connect() as con:
        rows = con.execute(
            """
            SELECT id, date, ncs_unit, bsr, image_note, image_b64, audio_note, ncs_term_ratio
            FROM logs WHERE uid=? ORDER BY id DESC
            """,
            (uid,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_log(uid: str, log_id: int) -> None:
    with _connect() as con:
        con.execute("DELETE FROM logs WHERE uid=? AND id=?", (uid, int(log_id)))


def clear_logs(uid: str) -> None:
    with _connect() as con:
        con.execute("DELETE FROM logs WHERE uid=?", (uid,))


def add_researcher_log(*, log_date: str, note: str) -> int:
    """연구자 성찰 로그 저장 (질적 연구 데이터용)."""
    with _connect() as con:
        cur = con.execute(
            "INSERT INTO researcher_logs(log_date, note) VALUES(?,?)",
            (log_date, note),
        )
        return int(cur.lastrowid)


def list_researcher_logs() -> list[dict[str, Any]]:
    """연구자 성찰 로그 목록 (최신순)."""
    with _connect() as con:
        rows = con.execute(
            "SELECT id, log_date, note, created_at FROM researcher_logs ORDER BY log_date DESC, id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def save_portfolio_comment(
    uid: str, comment_text: str, reflection_level: str = "", *, confirmed: bool = True
) -> None:
    """학생 포트폴리오용 [지도교사 종합의견] 저장. confirmed=True일 때만 학생 화면에 확정 반영."""
    is_confirmed = 1 if confirmed else 0
    with _connect() as con:
        con.execute(
            """
            INSERT INTO portfolio_comments(uid, comment_text, reflection_level, updated_at, is_confirmed)
            VALUES(?,?,?,datetime('now'),?)
            ON CONFLICT(uid) DO UPDATE SET
              comment_text=excluded.comment_text,
              reflection_level=excluded.reflection_level,
              updated_at=datetime('now'),
              is_confirmed=excluded.is_confirmed
            """,
            (uid, comment_text, reflection_level, is_confirmed),
        )


def get_portfolio_comment(uid: str) -> dict[str, Any] | None:
    """학생 포트폴리오용 종합 의견 조회 (draft·확정 모두)."""
    with _connect() as con:
        row = con.execute(
            "SELECT uid, comment_text, reflection_level, updated_at, is_confirmed FROM portfolio_comments WHERE uid=?",
            (uid,),
        ).fetchone()
        return dict(row) if row else None


def get_confirmed_portfolio_comment(uid: str) -> dict[str, Any] | None:
    """학생에게 노출: 교사가 [최종 승인]으로 확정한 의견만."""
    row = get_portfolio_comment(uid)
    if not row:
        return None
    if int(row.get("is_confirmed") or 0):
        return row
    return None


# ───────────────────────────────────────────────────────────────────
# 학생 프로필 (이력서/포트폴리오 1페이지 데이터)
# ───────────────────────────────────────────────────────────────────
EMPTY_PROFILE: dict[str, Any] = {
    "full_name": "",
    "birth_date": "",
    "email": "",
    "phone": "",
    "motto": "",
    "photo_b64": "",
    "educations": [],     # [{period, school, dept, status}, ...]
    "careers": [],        # [{period, company, role, description}, ...]
    "certificates": [],   # [{date, name, issuer}, ...]
    "awards": [],         # [{date, title, organizer}, ...]
    "tech_stack": [],     # [{skill, score}, ...]
}


def _safe_json_loads(s: Any) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (TypeError, ValueError):
        return None


def get_student_profile(uid: str) -> dict[str, Any]:
    """학생 프로필 조회. 미설정 시 EMPTY_PROFILE을 반환 (구조 안정성 확보)."""
    init_db()
    with _connect() as con:
        row = con.execute(
            """
            SELECT uid, full_name, birth_date, email, phone, motto, photo_b64,
                   educations_json, careers_json, certificates_json, awards_json,
                   tech_stack_json, updated_at
            FROM student_profiles WHERE uid=?
            """,
            (uid,),
        ).fetchone()
    if not row:
        return {**EMPTY_PROFILE, "uid": uid, "updated_at": ""}

    data: dict[str, Any] = {
        "uid": row["uid"],
        "full_name": row["full_name"] or "",
        "birth_date": row["birth_date"] or "",
        "email": row["email"] or "",
        "phone": row["phone"] or "",
        "motto": row["motto"] or "",
        "photo_b64": row["photo_b64"] or "",
        "educations": _safe_json_loads(row["educations_json"]) or [],
        "careers": _safe_json_loads(row["careers_json"]) or [],
        "certificates": _safe_json_loads(row["certificates_json"]) or [],
        "awards": _safe_json_loads(row["awards_json"]) or [],
        "tech_stack": _safe_json_loads(row["tech_stack_json"]) or [],
        "updated_at": row["updated_at"] or "",
    }
    return data


def save_student_profile(uid: str, profile: dict[str, Any]) -> None:
    """학생 프로필 저장(upsert). list/dict 항목은 JSON 직렬화."""
    init_db()
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
    with _connect() as con:
        con.execute(
            """
            INSERT INTO student_profiles
              (uid, full_name, birth_date, email, phone, motto, photo_b64,
               educations_json, careers_json, certificates_json, awards_json,
               tech_stack_json, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(uid) DO UPDATE SET
              full_name=excluded.full_name,
              birth_date=excluded.birth_date,
              email=excluded.email,
              phone=excluded.phone,
              motto=excluded.motto,
              photo_b64=excluded.photo_b64,
              educations_json=excluded.educations_json,
              careers_json=excluded.careers_json,
              certificates_json=excluded.certificates_json,
              awards_json=excluded.awards_json,
              tech_stack_json=excluded.tech_stack_json,
              updated_at=datetime('now')
            """,
            (
                uid,
                payload["full_name"],
                payload["birth_date"],
                payload["email"],
                payload["phone"],
                payload["motto"],
                payload["photo_b64"],
                payload["educations_json"],
                payload["careers_json"],
                payload["certificates_json"],
                payload["awards_json"],
                payload["tech_stack_json"],
            ),
        )


# ───────────────────────────────────────────────────────────────────
# 데모 시드 로그
#   - 첫 실행 또는 학생별 로그가 비어 있을 때 한 번만 호출
#   - 모든 일자(date)는 TEST_PERIOD_START(2026-05-11) ~ app_today() 범위 평일에 분포
# ───────────────────────────────────────────────────────────────────
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
    """테스트 기간(2026-05-11 ~ 2026-05-29) 밖의 일지를 삭제한다.

    개발 중 다른 날짜로 누적된 더미 데이터·실수 입력을 정리하는 용도.
    반환: 삭제된 행 수.
    """
    init_db()
    with _connect() as con:
        cur = con.execute(
            "DELETE FROM logs WHERE date < ? OR date > ?",
            (TEST_PERIOD_START.isoformat(), TEST_PERIOD_END.isoformat()),
        )
        return cur.rowcount or 0


def seed_demo_logs_if_empty(*, force_refresh: bool = False) -> int:
    """
    학생별로 테스트 기간 내 일지가 0건이면 데모 일지를 자동 생성한다.
    - 일자 분포: TEST_PERIOD_START(2026-05-11) ~ app_today() 사이 평일에 골고루
    - 학생별 1~3건 (학생마다 살짝씩 다름)
    - force_refresh=True 면 학생의 전체 일지를 지우고 다시 생성

    반환: 새로 만든 일지 건수.
    """
    init_db()
    # 테스트 기간 외 더미 데이터(예: 2026-04 등)는 항상 정리
    _purge_logs_outside_test_period()

    today = app_today()
    weekdays = [d for d in test_period_weekdays() if d <= today]
    if not weekdays:
        return 0

    rnd = random.Random(20260511)
    created = 0

    with _connect() as con:
        for idx, uid in enumerate(STUDENT_UIDS):
            if force_refresh:
                con.execute("DELETE FROM logs WHERE uid=?", (uid,))
                cur_n = 0
            else:
                row = con.execute(
                    "SELECT COUNT(*) AS n FROM logs WHERE uid=? AND date BETWEEN ? AND ?",
                    (uid, TEST_PERIOD_START.isoformat(), TEST_PERIOD_END.isoformat()),
                ).fetchone()
                cur_n = int(row["n"] if row else 0)

            if cur_n > 0:
                continue

            # 학생별 1~3건 중에서 가용 평일 수만큼만 생성
            n_logs = max(1, min(3, len(weekdays), 1 + (idx % 3)))
            chosen_dates = sorted(rnd.sample(weekdays, n_logs))
            chosen_units = rnd.sample(_DEMO_LOG_TEMPLATES, n_logs)

            for d, (unit, bsr) in zip(chosen_dates, chosen_units):
                con.execute(
                    """
                    INSERT INTO logs(uid, date, ncs_unit, bsr, image_note, audio_note, ncs_term_ratio)
                    VALUES(?,?,?,?,?,?,?)
                    """,
                    (
                        uid,
                        d.isoformat(),
                        unit,
                        bsr,
                        "사진 업로드됨" if rnd.random() < 0.6 else None,
                        "음성 녹음됨" if rnd.random() < 0.25 else None,
                        round(rnd.uniform(55.0, 92.0), 1),
                    ),
                )
                # 진도도 살짝 증가
                con.execute(
                    """
                    INSERT INTO progress(uid, ncs_unit, value) VALUES(?,?,?)
                    ON CONFLICT(uid, ncs_unit) DO UPDATE SET
                      value = MIN(100, progress.value + excluded.value)
                    """,
                    (uid, unit, rnd.randint(8, 18)),
                )
                created += 1

    return created
