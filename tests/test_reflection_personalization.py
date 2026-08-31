"""지도교수 피드백: 질문 초점, 개별화 피드백, 초안 비평가화, 자동 진행 금지."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsr_utils import (  # noqa: E402
    compose_reflection_draft_from_answers,
    evaluate_now_what_answer,
    evaluate_so_what_answer,
    extract_answer_elements,
    fallback_now_what_question,
    fallback_so_what_question,
    record_turn_answer_history,
    resolve_so_what_focus,
)

MEMO = "오늘 회사에서 재료 정리했습니다."
VAGUE_SO = "몰라. 그냥 네이밍 되어있는대로 정리만 한건데."
GOOD_SO = "부품 이름과 규격 번호를 보고 같은 종류끼리 모았습니다."
VAGUE_NW = "그냥 순서대로 하면 되지 않아?"
GOOD_NW = "다음에는 종류별로 먼저 나누고 같은 종류는 규격 번호 순으로 정리하겠습니다."


class ReflectionPersonalizationTests(unittest.TestCase):
    def test_so_what_focus_follows_task_type(self):
        focus = resolve_so_what_focus({"task_type": "assembly", "raw_input": MEMO})
        self.assertEqual(focus["task_type"], "assembly")
        self.assertIn("확인 기준", focus["foci"])
        gen = resolve_so_what_focus({"task_type": "general", "raw_input": MEMO})
        self.assertIn("판단 또는 선택 이유", gen["foci"])

    def test_fallback_so_what_uses_memo_not_invented_ic(self):
        q = fallback_so_what_question({"task_type": "general", "raw_input": MEMO, "task": ""})
        self.assertIn("정리", q)
        self.assertNotIn("IC", q)
        self.assertNotIn("오실로스코프", q)

    def test_vague_naming_is_insufficient_and_personalized(self):
        result = evaluate_so_what_answer(
            VAGUE_SO, {"raw_input": MEMO}, "", use_llm=False
        )
        self.assertFalse(result["is_sufficient"], result)
        blob = result["feedback"] + result["example_hint"] + result["followup_question"]
        self.assertIn("네이밍", blob)

    def test_good_so_what_is_sufficient(self):
        result = evaluate_so_what_answer(
            GOOD_SO, {"raw_input": MEMO}, "", use_llm=False
        )
        self.assertTrue(result["is_sufficient"], result)

    def test_exhausted_so_what_is_not_marked_ready(self):
        meta: dict = {}
        ev = evaluate_so_what_answer(VAGUE_SO, {"raw_input": MEMO}, "", use_llm=False)
        record_turn_answer_history(meta, turn=1, answer=VAGUE_SO, evaluation=ev)
        record_turn_answer_history(meta, turn=1, answer=VAGUE_SO, evaluation=ev)
        self.assertEqual(meta["turn1_proceed_status"], "proceed_with_warning")
        self.assertFalse(meta["turn1_ready"])

    def test_now_what_fallback_uses_so_what_words(self):
        q = fallback_now_what_question(
            {"task_type": "general", "raw_input": MEMO},
            GOOD_SO,
        )
        self.assertTrue("이름" in q or "규격" in q, q)
        elems = extract_answer_elements(GOOD_SO)
        self.assertIn("이름", elems.get("criterion") or "")

    def test_vague_order_now_what_is_personalized(self):
        result = evaluate_now_what_answer(
            VAGUE_NW,
            {"raw_input": MEMO},
            "",
            turn1_answer=GOOD_SO,
            use_llm=False,
        )
        self.assertFalse(result["is_sufficient"], result)
        blob = result["feedback"] + result["example_hint"] + result["followup_question"]
        self.assertIn("순서", blob)

    def test_good_now_what_is_sufficient(self):
        result = evaluate_now_what_answer(
            GOOD_NW,
            {"raw_input": MEMO},
            "",
            turn1_answer=GOOD_SO,
            use_llm=False,
        )
        self.assertTrue(result["is_sufficient"], result)

    def test_draft_uses_student_words_without_evaluation(self):
        draft = compose_reflection_draft_from_answers(MEMO, GOOD_SO, GOOD_NW)
        blob = draft["what"] + draft["so_what"] + draft["now_what"]
        self.assertIn("재료", draft["what"])
        self.assertIn("이름", draft["so_what"])
        self.assertIn("규격", draft["so_what"])
        self.assertIn("종류", draft["now_what"])
        self.assertNotIn("부족했", blob)
        self.assertNotIn("질문이 있었", blob)


if __name__ == "__main__":
    unittest.main()
