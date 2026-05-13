"""삭제 전 백업용 CSV/JSON 바이트 생성."""
from __future__ import annotations

import csv
import io
import json
from typing import Any

_LOG_KEYS: tuple[str, ...] = (
    "id",
    "date",
    "ncs_unit",
    "bsr",
    "image_note",
    "image_b64",
    "audio_note",
    "ncs_term_ratio",
)


def copy_log_row(row: dict[str, Any]) -> dict[str, Any]:
    """다이얼로그·백업용으로 일지 행을 얕은 복사한다."""
    return {k: row.get(k) for k in _LOG_KEYS}


def logs_to_csv_bytes(rows: list[dict[str, Any]], *, owner_uid: str = "") -> bytes:
    """실습 일지 목록을 UTF-8 BOM CSV로 직렬화한다."""
    buf = io.StringIO()
    fieldnames = ("student_uid", *_LOG_KEYS)
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore", quoting=csv.QUOTE_MINIMAL)
    w.writeheader()
    for r in rows:
        line: dict[str, Any] = {k: r.get(k, "") for k in _LOG_KEYS}
        for k in line:
            if line[k] is None:
                line[k] = ""
        line["student_uid"] = owner_uid or str(r.get("uid") or "")
        w.writerow(line)
    return buf.getvalue().encode("utf-8-sig")


def profile_to_json_bytes(profile: dict[str, Any]) -> bytes:
    """이력서 프로필을 JSON으로 직렬화한다. 사진 base64가 매우 크면 요약만 남긴다."""
    safe: dict[str, Any] = dict(profile)
    ph = safe.get("photo_b64")
    if isinstance(ph, str) and len(ph) > 300_000:
        safe["photo_b64"] = f"[BASE64_OMITTED_len={len(ph)}]"
    return json.dumps(safe, ensure_ascii=False, indent=2).encode("utf-8-sig")
