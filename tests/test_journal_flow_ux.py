"""3차 UX: 실습일지 작성 화면의 단계·버튼 문구가 유지되는지 확인."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class JournalFlowUxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.src = (ROOT / "student_view.py").read_text(encoding="utf-8")

    def test_five_stage_labels(self):
        for label in ("실습 기록", "실습 기록 확인", "So What", "Now What", "최종 확인·저장"):
            self.assertIn(f'"{label}"', self.src)

    def test_primary_buttons_exist(self):
        for label in (
            "실습 기록 완료 · AI 분석하기",
            "실습 기록 완료 · 성찰 시작하기",
            "답변 확인하기",
            "So What 완료 · 다음 단계",
            "성찰 완료 · 최종 기록 확인",
            "내용 확인 완료 · 최종 저장",
        ):
            self.assertIn(label, self.src)

    def test_photo_is_optional_copy(self):
        self.assertIn("실습 사진 (선택)", self.src)
        self.assertIn("생략할 수 있습니다", self.src)
        self.assertIn("PHOTO_TEXT_RELEVANCE_NOTICE", self.src)

    def test_friendly_notices(self):
        self.assertIn("PHOTO_TEXT_RELEVANCE_NOTICE", self.src)
        self.assertIn("현재 입력만으로 실습내용을 충분히 확인하기 어렵습니다", self.src)
        self.assertIn("AI 분석 중 일시적인 문제가 발생했습니다", self.src)
        self.assertIn("답변을 조금 더 구체화하면 실습 과정의 판단이 더 잘 드러납니다", self.src)
        self.assertIn("다음 실습에서 실제로 어떻게 적용할지 조금 더 구체적으로 작성해보세요", self.src)

    def test_final_draft_notice(self):
        self.assertIn("AI가 학생의 입력과 응답을 바탕으로 정리한 초안입니다", self.src)
        self.assertIn("성찰 기록을 마지막으로 확인해주세요", self.src)
        self.assertIn("So What 보완하기", self.src)
        writer = self.src.split("def _render_practice_log_chat_writer")[1].split("def _render_scaffolding_chat")[0]
        self.assertIn("_start_now_what", writer)
        self.assertIn("can_generate_now_what_question", writer)
        self.assertIn("can_open_final_reflection", writer)
        self.assertIn("실습 기록 완료 · 성찰 시작하기", writer)

    def test_ai_support_labels(self):
        self.assertIn("AI 피드백", self.src)
        self.assertIn("생각해 볼 예시", self.src)
        self.assertIn("추가 질문", self.src)
        self.assertNotIn("모범답안", self.src)
        self.assertNotIn("AI 정답", self.src)

    def test_no_new_score_copy_on_journal_writer(self):
        writer = self.src.split("def _render_practice_log_chat_writer")[1].split("def _render_scaffolding_chat")[0]
        for banned in ("NCS 역량 점수", "달성률", "AI 성장 점수", "성찰 능력 점수", "메타인지 점수"):
            self.assertNotIn(banned, writer)


if __name__ == "__main__":
    unittest.main()
