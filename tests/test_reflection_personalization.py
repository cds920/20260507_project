"""지도교수 피드백: 질문 초점, 개별화 피드백, 초안 비평가화, 자동 진행 금지."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

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
    generate_now_what_question,
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

    def test_circuit_assembly_now_what_does_not_become_sorting(self):
        memo = (
            "전자회로 도면을 보고 만능기판에 IC, 저항, 7세그먼트, 모터 등의 부품을 "
            "배치하고 납땜하여 회로를 조립했습니다. 조립 후 전원을 연결해 "
            "7세그먼트와 LED가 동작하는지 확인하고, 배선과 납땜 상태를 점검했습니다."
        )
        answer = (
            "회로도에서 부품의 위치와 연결 방향을 먼저 확인하고, "
            "실제 기판에서 부품 다리가 표시된 위치의 구멍에 들어갔는지 비교했습니다. "
            "또 부품이 기판에 너무 뜨지 않고 밀착되어 있는지, "
            "다리가 휘거나 옆 부품과 닿지 않는지도 확인했습니다."
        )
        ana = {
            "task_type": "assembly",
            "task": "전자회로 조립",
            "raw_input": memo,
            "ncs_unit": "전자 제품 생산",
            "image_analysis": ["만능기판", "IC", "7세그먼트"],
            "equipment": ["만능기판"],
        }
        q = fallback_now_what_question(ana, answer)
        self.assertNotIn("재료", q)
        self.assertNotIn("정리했", q)
        self.assertIn("조립", q)
        self.assertIn("위치", q)
        self.assertIn("방향", q)
        self.assertIn("밀착", q)
        self.assertIn("적용", q)

    def test_off_context_now_what_is_regenerated(self):
        memo = (
            "전자회로 도면을 보고 만능기판에 IC, 저항, 7세그먼트, 모터 등의 부품을 "
            "배치하고 납땜하여 회로를 조립했습니다."
        )
        answer = (
            "회로도에서 부품의 위치와 연결 방향을 먼저 확인하고, "
            "실제 기판에서 부품 다리가 표시된 위치의 구멍에 들어갔는지 비교했습니다. "
            "또 부품이 기판에 너무 뜨지 않고 밀착되어 있는지 확인했습니다."
        )
        ana = {
            "task_type": "assembly",
            "raw_input": memo,
            "ncs_unit": "전자 제품 생산",
            "image_analysis": ["만능기판", "IC"],
            "equipment": ["만능기판"],
        }
        bad = (
            "앞에서 표시를 기준으로 정리했다고 했어요. "
            "다음에 비슷한 재료를 정리한다면, "
            "필요한 부품을 더 쉽게 찾을 수 있도록 어떤 순서나 방식으로 배치하면 좋을까요?"
        )
        good = (
            "이번에 확인했던 부품 위치, 연결 방향, 기판 밀착 상태를 "
            "다음 회로 조립에서는 어떻게 적용하거나 개선하겠나요?"
        )
        with patch("bsr_utils.resolve_google_api_key", return_value="test-key"), patch(
            "bsr_utils.gemini_generate_text", side_effect=[bad, good]
        ):
            q = generate_now_what_question(
                ana,
                "부품 다리가 기판 구멍에 제대로 들어갔는지 어떤 부분을 확인했나요?",
                answer,
                allowed=True,
            )
        self.assertNotIn("재료", q)
        self.assertIn("조립", q)
        self.assertIn("밀착", q)

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
