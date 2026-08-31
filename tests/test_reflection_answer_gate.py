"""2차 스캐폴딩: So What / Now What 응답 적합성 게이트 테스트."""
from __future__ import annotations

import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsr_utils import (  # noqa: E402
    TURN_SUPPORT_MAX,
    build_reflection_string,
    evaluate_now_what_answer,
    evaluate_so_what_answer,
    generate_now_what_question,
    generate_so_what_question,
    get_reflection_meta,
    record_turn_answer_history,
)


class ReflectionAnswerGateTests(unittest.TestCase):
    def test_so_what_just_looked_is_insufficient_with_support(self):
        result = evaluate_so_what_answer("그냥 봤어요.", {}, "", use_llm=False)
        self.assertFalse(result["is_sufficient"], result)
        self.assertTrue(result["feedback"])
        self.assertTrue(result["example_hint"])
        self.assertTrue(result["followup_question"])
        self.assertNotIn("오실로스코프", result["feedback"] + result["example_hint"])

    def test_so_what_concrete_comparison_is_sufficient(self):
        result = evaluate_so_what_answer(
            "전원 전압을 확인한 뒤 클럭과 출력 신호를 측정하여 예상값과 비교했다.",
            {"raw_input": "7세그먼트 회로 조립 후 신호를 측정했다."},
            "어떤 값을 기준으로 확인했나요?",
            use_llm=False,
        )
        self.assertTrue(result["is_sufficient"], result)
        self.assertEqual(result["feedback"], "")
        self.assertEqual(result["followup_question"], "")

    def test_now_what_will_do_better_is_insufficient_with_support(self):
        result = evaluate_now_what_answer(
            "다음에는 잘하겠습니다.",
            {},
            "",
            turn1_answer="전원 전압을 확인했다.",
            use_llm=False,
        )
        self.assertFalse(result["is_sufficient"], result)
        self.assertTrue(result["feedback"])
        self.assertTrue(result["example_hint"])
        self.assertTrue(result["followup_question"])

    def test_now_what_ordered_plan_is_sufficient(self):
        result = evaluate_now_what_answer(
            "다음에는 예상값을 먼저 정리하고 전원, 입력, 출력 순서로 측정하여 비교하겠다.",
            {"raw_input": "전원과 출력 신호를 측정했다."},
            "다음 실습에서 어떻게 적용할까요?",
            turn1_answer="전원 전압을 확인한 뒤 출력 신호를 비교했다.",
            use_llm=False,
        )
        self.assertTrue(result["is_sufficient"], result)

    def test_support_max_is_two(self):
        self.assertEqual(TURN_SUPPORT_MAX, 2)

    def test_question_generators_are_unchanged_call_signatures(self):
        so_sig = inspect.signature(generate_so_what_question)
        nw_sig = inspect.signature(generate_now_what_question)
        self.assertIn("analysis", so_sig.parameters)
        self.assertIn("turn1_question", nw_sig.parameters)
        self.assertIn("turn1_answer", nw_sig.parameters)

    def _three_attempt_meta(self, turn: int) -> dict:
        meta: dict = {}
        first = evaluate_so_what_answer("그냥 봤어요.", {}, "", use_llm=False)
        second = evaluate_so_what_answer("신호를 봤어요.", {}, "", use_llm=False)
        if turn == 2:
            first = evaluate_now_what_answer(
                "다음에는 잘하겠습니다.", {}, "", turn1_answer="확인했다.", use_llm=False
            )
            second = evaluate_now_what_answer(
                "조심하겠습니다.", {}, "", turn1_answer="확인했다.", use_llm=False
            )
            third = evaluate_now_what_answer(
                "다음에는 예상값을 먼저 정리하고 전원, 입력, 출력 순서로 측정하여 비교하겠다.",
                {"raw_input": "전원과 출력 신호를 측정했다."},
                "",
                turn1_answer="전원 전압을 확인한 뒤 출력 신호를 비교했다.",
                use_llm=False,
            )
            answers = ("다음에는 잘하겠습니다.", "조심하겠습니다.", third and "다음에는 예상값을 먼저 정리하고 전원, 입력, 출력 순서로 측정하여 비교하겠다.")
            evals = (first, second, third)
        else:
            third = evaluate_so_what_answer(
                "전원 전압을 확인한 뒤 클럭과 출력 신호를 측정하여 예상값과 비교했다.",
                {"raw_input": "7세그먼트 회로 조립 후 신호를 측정했다."},
                "",
                use_llm=False,
            )
            answers = (
                "그냥 봤어요.",
                "신호를 봤어요.",
                "전원 전압을 확인한 뒤 클럭과 출력 신호를 측정하여 예상값과 비교했다.",
            )
            evals = (first, second, third)
        for ans, ev in zip(answers, evals):
            record_turn_answer_history(meta, turn=turn, answer=ans, evaluation=ev)
        return meta, answers, evals

    def test_so_what_three_attempts_all_kept_in_revisions(self):
        meta, answers, evals = self._three_attempt_meta(1)
        self.assertEqual(meta["turn1_answer_initial"], answers[0])
        self.assertIn(answers[1], meta["turn1_retry_answer"])
        self.assertIn(answers[2], meta["turn1_retry_answer"])
        self.assertEqual(meta["a1"], answers[2])
        revs = meta["turn1_revisions"]
        self.assertEqual(len(revs), 3)
        self.assertEqual([r["answer"] for r in revs], list(answers))
        self.assertFalse(revs[0]["validation"]["is_sufficient"])
        self.assertFalse(revs[1]["validation"]["is_sufficient"])
        self.assertTrue(revs[2]["validation"]["is_sufficient"])
        self.assertTrue(revs[0]["feedback"])
        self.assertTrue(revs[1]["feedback"])
        saved = build_reflection_string("w", "s", "n", meta={**meta, "turn1_answer": answers[2]})
        loaded = get_reflection_meta(saved)
        self.assertEqual(len(loaded["turn1_revisions"]), 3)
        self.assertEqual(loaded["turn1_answer_initial"], answers[0])
        self.assertEqual([r["answer"] for r in loaded["turn1_revisions"]], list(answers))

    def test_now_what_three_attempts_all_kept_in_revisions(self):
        meta, answers, evals = self._three_attempt_meta(2)
        self.assertEqual(meta["turn2_answer_initial"], answers[0])
        self.assertEqual(len(meta["turn2_revisions"]), 3)
        self.assertEqual([r["answer"] for r in meta["turn2_revisions"]], list(answers))
        self.assertFalse(meta["turn2_revisions"][0]["validation"]["is_sufficient"])
        self.assertTrue(meta["turn2_revisions"][2]["validation"]["is_sufficient"])
        saved = build_reflection_string("w", "s", "n", meta={**meta, "turn2_answer": answers[2]})
        loaded = get_reflection_meta(saved)
        self.assertEqual([r["answer"] for r in loaded["turn2_revisions"]], list(answers))
        self.assertIn("turn2_answer_initial", loaded)
        self.assertIn("turn2_retry_answer", loaded)


if __name__ == "__main__":
    unittest.main()
