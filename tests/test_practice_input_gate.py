"""1차 스캐폴딩: 실습내용 적합성·사진 분기·task_type 유지 테스트."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bsr_utils import (  # noqa: E402
    PHOTO_TEXT_RELEVANCE_NOTICE,
    analyze_practice_experience,
    assess_photo_text_relevance,
    check_practice_input_validity,
    heuristic_practice_analysis,
    resolve_practice_analysis_mode,
)

SUFFICIENT_MEMO = (
    "전자기능사 회로 조립 후 7세그먼트가 정상적으로 동작하지 않아 "
    "배선을 확인하고 오실로스코프로 신호를 측정했다."
)


class PracticeInputGateTests(unittest.TestCase):
    def test_1_today_was_hard_requests_revision(self):
        result = check_practice_input_validity(
            "오늘 힘들었다.", has_photo=False, use_llm=False
        )
        self.assertFalse(result["is_valid"])
        self.assertIn("작업", result["reason"] + "".join(result.get("missing") or []))
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=False, has_photo=False),
            "need_text",
        )

    def test_2_laughter_requests_revision(self):
        result = check_practice_input_validity("ㅋㅋㅋㅋ", has_photo=False, use_llm=False)
        self.assertFalse(result["is_valid"])
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=False, has_photo=False),
            "need_text",
        )

    def test_3_sufficient_text_without_photo_uses_text_analysis(self):
        result = check_practice_input_validity(
            SUFFICIENT_MEMO, has_photo=False, use_llm=False
        )
        self.assertTrue(result["is_valid"], result)
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=True, has_photo=False),
            "text",
        )
        analysis = analyze_practice_experience(SUFFICIENT_MEMO, [], "", api_key="")
        self.assertEqual(analysis["raw_input"], SUFFICIENT_MEMO)
        self.assertIn("task_type", analysis)
        self.assertIn("problem_occurred", analysis)
        self.assertIn("equipment", analysis)
        self.assertIn("evidence", analysis)

    def test_4_sufficient_text_with_photo_uses_multimodal_route(self):
        result = check_practice_input_validity(
            SUFFICIENT_MEMO, has_photo=True, use_llm=False
        )
        self.assertTrue(result["is_valid"], result)
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=True, has_photo=True),
            "multimodal",
        )
        detected = [{"객체": "오실로스코프", "신뢰도": "90%"}]
        analysis = heuristic_practice_analysis(
            SUFFICIENT_MEMO, detected, "전자회로조립"
        )
        self.assertIn("오실로스코프", analysis.get("equipment") or [])
        self.assertEqual(analysis["task_type"], "troubleshooting")

    def test_5_photo_with_insufficient_text_requests_revision(self):
        result = check_practice_input_validity(
            "잘했다.", has_photo=True, use_llm=False
        )
        self.assertFalse(result["is_valid"])
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=False, has_photo=True),
            "need_text",
        )
        # 사진이 있다는 이유만으로 오류/적합 판정하지 않는다.
        self.assertNotIn("사진", result.get("reason") or "")

    def test_6_problem_phrase_prefers_troubleshooting(self):
        analysis = heuristic_practice_analysis(SUFFICIENT_MEMO, None, "")
        self.assertTrue(analysis["problem_occurred"])
        self.assertEqual(analysis["task_type"], "troubleshooting")
        merged = analyze_practice_experience(SUFFICIENT_MEMO, None, "", api_key="")
        self.assertTrue(merged["problem_occurred"])
        self.assertEqual(merged["task_type"], "troubleshooting")

    def test_measurement_without_problem_is_not_forced_troubleshooting(self):
        memo = "오실로스코프로 7세그먼트 구동 파형을 측정하고 주파수를 기록했다."
        analysis = heuristic_practice_analysis(memo, None, "")
        self.assertFalse(analysis["problem_occurred"])
        self.assertEqual(analysis["task_type"], "measurement")

    def test_missing_photo_is_not_an_error_when_text_is_enough(self):
        result = check_practice_input_validity(
            SUFFICIENT_MEMO, has_photo=False, use_llm=False
        )
        self.assertTrue(result["is_valid"])
        self.assertNotIn("사진", result.get("reason") or "")

    def test_examples_love_and_unknown_are_invalid(self):
        for text in ("몰라요.", "선생님 사랑해요."):
            with self.subTest(text=text):
                result = check_practice_input_validity(
                    text, has_photo=False, use_llm=False
                )
                self.assertFalse(result["is_valid"], result)

    def test_unrelated_photo_warns_but_text_analysis_continues(self):
        text_ok = check_practice_input_validity(
            SUFFICIENT_MEMO, has_photo=True, use_llm=False
        )
        self.assertTrue(text_ok["is_valid"], text_ok)
        self.assertEqual(
            resolve_practice_analysis_mode(is_valid=True, has_photo=True),
            "multimodal",
        )
        rel = assess_photo_text_relevance(
            SUFFICIENT_MEMO,
            [{"객체": "음식"}, {"객체": "강아지"}],
            "",
        )
        self.assertFalse(rel["is_related"])
        self.assertFalse(rel["used_as_evidence"])
        analysis = analyze_practice_experience(SUFFICIENT_MEMO, [], "", api_key="")
        self.assertEqual(analysis["raw_input"], SUFFICIENT_MEMO)
        self.assertIn("task_type", analysis)
        self.assertTrue(PHOTO_TEXT_RELEVANCE_NOTICE)

    def test_related_photo_is_used_as_evidence(self):
        rel = assess_photo_text_relevance(
            SUFFICIENT_MEMO,
            [{"객체": "오실로스코프", "신뢰도": "90%"}],
            "전자회로조립",
        )
        self.assertTrue(rel["is_related"], rel)
        self.assertTrue(rel["used_as_evidence"])


if __name__ == "__main__":
    unittest.main()
