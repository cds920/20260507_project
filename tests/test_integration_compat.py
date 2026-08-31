"""4차 통합: 성찰메타 저장, BSR 호환, Sheets 컬럼, 기술 스택, 보안 정적 검사."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsr_utils import (  # noqa: E402
    GEMINI_PRIMARY_MODEL,
    REFLECTION_META_TAG,
    TURN_SUPPORT_MAX,
    build_reflection_string,
    generate_portfolio_entry,
    get_reflection_body,
    get_reflection_meta,
    parse_reflection_record,
    reflection_display_sections,
)

_FROZEN_LOGS_HEADERS = [
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

LEGACY_BSR = (
    "[배경] 전자회로 실습에서 전원 회로를 구성했다.\n"
    "[해결] 멀티미터로 전압을 측정하고 배선을 수정했다.\n"
    "[성과] 정상 전압이 나오도록 연결을 점검하는 경험을 했다.\n"
    + REFLECTION_META_TAG
    + json.dumps(
        {
            "raw_input": "전원 회로 구성",
            "task_type": "assembly",
            "turn1_question": "Q1",
            "turn1_answer": "A1",
            "turn2_question": "Q2",
            "turn2_answer": "A2",
            "problem_occurred": False,
            "evidence": ["전압 측정"],
            "image_analysis": "전원 회로 사진",
        },
        ensure_ascii=False,
    )
)

NEW_WSWNW_META = {
    "raw_input": "7세그먼트 회로 조립 후 오실로스코프로 신호를 측정했다.",
    "turn1_question": "어떤 기준으로 확인했나요?",
    "turn1_answer": "클럭과 출력을 비교했다.",
    "turn2_question": "다음에 어떻게 적용할까요?",
    "turn2_answer": "예상값을 먼저 정리하고 측정하겠다.",
    "task_type": "troubleshooting",
    "problem_occurred": True,
    "evidence": ["7세그먼트", "오실로스코프"],
    "image_analysis": "오실로스코프 화면",
    "input_validation": {
        "is_valid": True,
        "reason": "수행 작업과 실습 맥락이 식별됨",
        "source": "rule",
        "has_photo": False,
    },
    "analysis_mode": "text",
    "turn1_answer_initial": "그냥 봤다.",
    "turn1_sufficiency": {"is_sufficient": True, "reason": "비교 기준 포함"},
    "turn1_feedback": "무엇을 기준으로 비교했는지 적어 주세요.",
    "turn1_example_hint": "전원 전압을 확인한 뒤 …",
    "turn1_followup_question": "어떤 값을 비교했나요?",
    "turn1_retry_answer": "클럭과 출력을 비교했다.",
    "turn1_support_count": 1,
    "turn2_answer_initial": "잘하겠습니다.",
    "turn2_sufficiency": {"is_sufficient": True, "reason": "적용 계획 포함"},
    "turn2_feedback": "다음 실습에서 할 일을 구체적으로 적어 주세요.",
    "turn2_example_hint": "예상값을 먼저 정리하고 …",
    "turn2_followup_question": "어떤 순서로 측정할까요?",
    "turn2_retry_answer": "예상값을 먼저 정리하고 측정하겠다.",
    "turn2_support_count": 1,
}


class SheetsAndStackCompatTests(unittest.TestCase):
    def test_logs_headers_not_expanded(self):
        src = (ROOT / "db.py").read_text(encoding="utf-8")
        start = src.index("LOGS_HEADERS")
        chunk = src[start : start + 400]
        for col in _FROZEN_LOGS_HEADERS:
            self.assertIn(f'"{col}"', chunk)
        self.assertNotIn("input_validation", chunk)
        self.assertNotIn("turn1_sufficiency", chunk)
        self.assertNotIn("analysis_mode", chunk)
        listed = [
            line.strip().strip(",").strip('"')
            for line in chunk.splitlines()
            if line.strip().startswith('"')
        ]
        self.assertEqual(listed[:10], _FROZEN_LOGS_HEADERS)

    def test_primary_model_and_requirements_stack(self):
        self.assertEqual(GEMINI_PRIMARY_MODEL, "gemini-2.5-flash")
        req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
        for pkg in ("streamlit", "gspread", "pillow", "google-generativeai"):
            self.assertIn(pkg, req)
        for banned in ("langchain", "chromadb", "faiss", "firebase", "pinecone"):
            self.assertNotIn(banned, req)

    def test_no_rag_langchain_imports_in_runtime_py(self):
        banned = ("langchain", "chromadb", "firebase_admin", "faiss")
        for path in ROOT.glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for name in banned:
                self.assertNotRegex(
                    text,
                    rf"(?m)^\s*(import|from)\s+{name}\b",
                    msg=f"{path.name} imports {name}",
                )


class ReflectionMetaCompatTests(unittest.TestCase):
    def test_legacy_fields_roundtrip_inside_meta_json(self):
        rec = parse_reflection_record(LEGACY_BSR)
        self.assertEqual(rec["format"], "legacy_bsr")
        meta = rec["meta"]
        for key in (
            "raw_input",
            "turn1_question",
            "turn1_answer",
            "turn2_question",
            "turn2_answer",
            "task_type",
            "problem_occurred",
            "evidence",
            "image_analysis",
        ):
            self.assertIn(key, meta)

    def test_new_fields_roundtrip_without_new_sheet_columns(self):
        saved = build_reflection_string(
            "7세그먼트 회로를 조립하고 신호를 측정했다.",
            "클럭과 출력을 비교해 이상 여부를 판단했다.",
            "다음에는 예상값을 먼저 정리하고 측정하겠다.",
            meta=NEW_WSWNW_META,
        )
        self.assertIn("[What]", saved)
        self.assertIn(REFLECTION_META_TAG, saved)
        meta = get_reflection_meta(saved)
        self.assertEqual(meta["raw_input"], NEW_WSWNW_META["raw_input"])
        self.assertEqual(meta["task_type"], "troubleshooting")
        self.assertEqual(meta["analysis_mode"], "text")
        self.assertTrue(meta["input_validation"]["is_valid"])
        self.assertEqual(meta["turn1_answer"], NEW_WSWNW_META["turn1_answer"])
        self.assertEqual(meta["turn1_retry_answer"], NEW_WSWNW_META["turn1_retry_answer"])
        self.assertEqual(meta["turn1_support_count"], 1)
        self.assertEqual(meta["turn1_feedback"], NEW_WSWNW_META["turn1_feedback"])
        rec = parse_reflection_record(saved)
        self.assertEqual(rec["format"], "wswnw")

    def test_support_limit_is_two(self):
        self.assertEqual(TURN_SUPPORT_MAX, 2)


class LegacyBsrAndPortfolioCompatTests(unittest.TestCase):
    def test_legacy_bsr_is_not_converted_to_wswnw(self):
        rec = parse_reflection_record(LEGACY_BSR)
        self.assertEqual(rec["format"], "legacy_bsr")
        self.assertFalse(rec["what"])
        self.assertTrue(rec["legacy_background"])
        body = get_reflection_body(LEGACY_BSR)
        self.assertIn("[배경]", body)
        self.assertIn("[해결]", body)
        self.assertIn("[성과]", body)
        self.assertNotIn("[What]", body)
        labels = [t for t, _h, _b in reflection_display_sections(LEGACY_BSR)]
        self.assertTrue(any("배경" in t for t in labels))

    def test_portfolio_keeps_legacy_labels_and_does_not_invent_wswnw(self):
        entry = generate_portfolio_entry(LEGACY_BSR)
        self.assertEqual(entry["format"], "legacy_bsr")
        titles = [t for t, _c in entry["sections"]]
        self.assertTrue(all(t.startswith("이전 형식") for t in titles))
        joined = " ".join(c for _t, c in entry["sections"])
        self.assertIn("전원 회로", joined)
        self.assertNotIn("[What]", joined)

    def test_portfolio_wswnw_uses_rule_based_section_titles(self):
        saved = build_reflection_string(
            "만능기판에 7세그먼트 회로를 조립한 뒤 오실로스코프로 신호를 확인했다.",
            "클럭과 출력 파형을 비교하여 정상 여부를 판단했다.",
            "다음에는 예상값을 먼저 정리하고 전원, 입력, 출력 순서로 측정하겠다.",
            meta={"raw_input": "7세그먼트 조립", "task_type": "assembly"},
        )
        entry = generate_portfolio_entry(saved)
        self.assertEqual(entry["format"], "wswnw")
        titles = [t for t, _c in entry["sections"]]
        self.assertEqual(
            titles,
            ["실습 배경 및 목표", "수행 과정 및 핵심 판단", "성찰 및 향후 적용"],
        )
        src = (ROOT / "bsr_utils.py").read_text(encoding="utf-8")
        start = src.index("def generate_portfolio_entry")
        chunk = src[start : start + 1800]
        self.assertNotIn("genai", chunk)
        self.assertNotIn("_gemini_text", chunk)


class SecurityStaticTests(unittest.TestCase):
    def test_gitignore_keeps_secrets_local(self):
        gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".streamlit/*", gi)
        self.assertIn("secrets.toml.example", gi)

    def test_login_and_password_widgets_are_masked(self):
        app = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn('type="password"', app)
        ui = (ROOT / "ui_style.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(ui.count('type="password"'), 2)

    def test_no_credential_print_or_st_write_in_runtime(self):
        needles = (
            "print(api_key",
            "print(password",
            "st.write(api_key",
            "st.code(api_key",
            "st.json(st.secrets",
            "st.write(st.secrets",
        )
        for path in (ROOT / "app.py", ROOT / "student_view.py", ROOT / "teacher_view.py", ROOT / "bsr_utils.py", ROOT / "db.py"):
            text = path.read_text(encoding="utf-8")
            for needle in needles:
                self.assertNotIn(needle, text, msg=f"{path.name} contains {needle}")

    def test_authenticate_stores_uid_only_in_session(self):
        src = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertIn("authenticate(uid_norm", src)
        self.assertIn('st.session_state.user = user["uid"]', src)
        self.assertNotIn("st.session_state.user = user", src.replace('st.session_state.user = user["uid"]', ""))


if __name__ == "__main__":
    unittest.main()
