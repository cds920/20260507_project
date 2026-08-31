"""So What / Now What 적합성 게이트 A–D 회귀."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsr_utils import (  # noqa: E402
    can_generate_now_what_question,
    can_open_final_reflection,
    evaluate_now_what_answer,
    evaluate_so_what_answer,
    generate_now_what_question,
    record_turn_answer_history,
)

Q1 = "어떤 부품을 어떤 기준으로 배치했나요?"
A1_SHORT = "전원선부터"
A1_GOOD = "전원부를 먼저 배치하고 신호 흐름에 따라 연결되는 부품을 순서대로 배치했습니다."
A2_VAGUE = "모르겠어.."
A2_GOOD = "다음에는 전원부를 먼저 배치한 뒤 신호 흐름과 연결관계를 확인하면서 부품 위치를 정하겠습니다."
MEMO_ANA = {"raw_input": "회로도를 그렸습니다", "task": "회로도"}


def _llm_says_sufficient() -> dict:
    return {
        "is_sufficient": True,
        "reason": "llm-upgrade",
        "feedback": "",
        "example_hint": "",
        "followup_question": "",
        "source": "gemini",
    }


class ReflectionSufficiencyGateTests(unittest.TestCase):
    def test_case_a_short_so_what_blocks_now_what(self):
        ev = evaluate_so_what_answer(A1_SHORT, MEMO_ANA, Q1, use_llm=False)
        self.assertFalse(ev["is_sufficient"], ev)
        self.assertIn("전원선부터", ev["feedback"])
        self.assertTrue(ev["example_hint"])
        self.assertTrue(ev["followup_question"])
        meta: dict = {}
        record_turn_answer_history(meta, turn=1, answer=A1_SHORT, evaluation=ev)
        self.assertFalse(can_generate_now_what_question(meta))
        q2 = generate_now_what_question(
            MEMO_ANA, Q1, A1_SHORT, allowed=can_generate_now_what_question(meta)
        )
        self.assertEqual(q2, "")
        self.assertFalse(can_open_final_reflection(meta))

    def test_case_b_revised_so_what_allows_now_what(self):
        ev = evaluate_so_what_answer(A1_GOOD, MEMO_ANA, Q1, use_llm=False)
        self.assertTrue(ev["is_sufficient"], ev)
        meta: dict = {}
        record_turn_answer_history(meta, turn=1, answer=A1_GOOD, evaluation=ev)
        self.assertTrue(can_generate_now_what_question(meta))
        q2 = generate_now_what_question(
            MEMO_ANA, Q1, A1_GOOD, allowed=can_generate_now_what_question(meta)
        )
        self.assertTrue(q2.strip())
        self.assertFalse(can_open_final_reflection(meta))

    def test_case_c_vague_now_what_blocks_final(self):
        meta: dict = {}
        so = evaluate_so_what_answer(A1_GOOD, MEMO_ANA, Q1, use_llm=False)
        record_turn_answer_history(meta, turn=1, answer=A1_GOOD, evaluation=so)
        ev = evaluate_now_what_answer(
            A2_VAGUE, MEMO_ANA, "다음에 어떻게 배치할까요?", turn1_answer=A1_GOOD, use_llm=False
        )
        self.assertFalse(ev["is_sufficient"], ev)
        self.assertIn("적용할 방법", ev["feedback"])
        record_turn_answer_history(meta, turn=2, answer=A2_VAGUE, evaluation=ev)
        self.assertFalse(can_open_final_reflection(meta))

    def test_case_d_revised_now_what_opens_final(self):
        meta: dict = {}
        so = evaluate_so_what_answer(A1_GOOD, MEMO_ANA, Q1, use_llm=False)
        record_turn_answer_history(meta, turn=1, answer=A1_GOOD, evaluation=so)
        ev = evaluate_now_what_answer(
            A2_GOOD, MEMO_ANA, "다음에 어떻게 배치할까요?", turn1_answer=A1_GOOD, use_llm=False
        )
        self.assertTrue(ev["is_sufficient"], ev)
        record_turn_answer_history(meta, turn=2, answer=A2_GOOD, evaluation=ev)
        self.assertTrue(can_open_final_reflection(meta))

    def test_llm_cannot_upgrade_short_so_what(self):
        with patch("bsr_utils._gemini_turn_answer_eval", return_value=_llm_says_sufficient()):
            ev = evaluate_so_what_answer(
                A1_SHORT, MEMO_ANA, Q1, use_llm=True, api_key="dummy"
            )
        self.assertFalse(ev["is_sufficient"], ev)
        self.assertTrue(ev["feedback"])

    def test_llm_cannot_upgrade_unknown_now_what(self):
        with patch("bsr_utils._gemini_turn_answer_eval", return_value=_llm_says_sufficient()):
            ev = evaluate_now_what_answer(
                A2_VAGUE,
                MEMO_ANA,
                "다음에 어떻게 배치할까요?",
                turn1_answer=A1_GOOD,
                use_llm=True,
                api_key="dummy",
            )
        self.assertFalse(ev["is_sufficient"], ev)
        self.assertTrue(ev["feedback"])

    def test_now_what_stock_phrases_are_insufficient(self):
        for ans in ("잘할게", "그냥 하면 돼", "모르겠어"):
            ev = evaluate_now_what_answer(
                ans, MEMO_ANA, "", turn1_answer=A1_GOOD, use_llm=False
            )
            self.assertFalse(ev["is_sufficient"], (ans, ev))

    def test_allow_proceed_is_the_only_bypass(self):
        ev = evaluate_so_what_answer(A1_SHORT, MEMO_ANA, Q1, use_llm=False)
        meta: dict = {}
        record_turn_answer_history(meta, turn=1, answer=A1_SHORT, evaluation=ev)
        self.assertFalse(can_generate_now_what_question(meta))
        meta["turn1_allow_proceed"] = True
        self.assertTrue(can_generate_now_what_question(meta))


if __name__ == "__main__":
    unittest.main()
