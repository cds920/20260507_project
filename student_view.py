import datetime
import hashlib
import html
import io
import json
import re
import time
from typing import Any

from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bsr_utils import (
    GEMINI_EMPTY_RESPONSE_MESSAGE,
    GEMINI_PRIMARY_MODEL,
    mark_primary_unavailable,
    analyze_practice_experience,
    build_reflection_string,
    check_evidence_validity,
    extract_background_section,
    extract_bsr_section,
    extract_generate_content_text,
    gemini_generate_text,
    gemini_safety_settings_block_none,
    generate_bsr_draft_from_keywords,
    generate_now_what_question,
    generate_portfolio_entry,
    generate_reflection_draft,
    generate_so_what_question,
    get_ai_scaffolding,
    get_gemini_model,
    get_reflection_example_sentence,
    parse_reflection_record,
    get_reflection_body,
    get_reflection_meta,
    reflection_display_sections,
    render_portfolio_entry_html,
    resolved_gemini_model_candidates,
    radar_scores_from_logs,
    render_bsr_highlighted,
    resolve_google_api_key,
    uploaded_files_to_gemini_pil_images,
)
from constants import (
    CHECKLIST,
    COLLOQUIAL_TO_NCS,
    DEFAULT_NCS_PROGRESS,
    ELECTRONICS_NCS_UNITS,
    GLOSSARY,
    NCS_DB,
    format_ncs_unit,
    ncs_unit_names_for_prompt,
)
from pathlib import Path

from backup_utils import copy_log_row, logs_to_csv_bytes, profile_to_json_bytes
from db import (
    DuplicateLogError,
    TEST_PERIOD_END,
    add_log,
    app_today,
    seoul_today,
    log_display_date,
    parse_calendar_date,
    clear_logs,
    clear_student_profile,
    delete_log,
    get_confirmed_portfolio_comment,
    get_student_profile,
    list_logs,
    save_student_profile,
    seed_progress_if_missing,
    student_label,
    update_progress,
)
from ui_style import P, render_password_change_expander, render_portfolio_print_button

# 차트용 메인 컬러
_CHART_PRIMARY = P["primary"]
_CHART_ACCENT = P["accent"]


def _evidence_file_size(img) -> int:
    """UploadedFile 크기(바이트). size 속성 없으면 read 길이로 대체."""
    s = getattr(img, "size", None)
    if isinstance(s, int) and s >= 0:
        return s
    try:
        img.seek(0)
        n = len(img.read())
        img.seek(0)
        return n
    except Exception:
        return 0


def _normalize_img_input(img_or_imgs) -> list:
    """단일 UploadedFile / 리스트 / None을 항상 List[UploadedFile]로 정규화."""
    if img_or_imgs is None:
        return []
    if isinstance(img_or_imgs, (list, tuple)):
        return [f for f in img_or_imgs if f is not None]
    return [img_or_imgs]


def _upload_files_meta_tuple(files: list) -> tuple[tuple[str, int], ...]:
    """업로드 집합이 바뀌었는지 빠르게 판별하기 위한 (이름, 크기) 튜플."""
    out: list[tuple[str, int]] = []
    for f in files:
        name = getattr(f, "name", "") or ""
        sz = getattr(f, "size", None)
        if not isinstance(sz, int) or sz < 0:
            sz = _evidence_file_size(f)
        out.append((name, int(sz)))
    return tuple(out)


def _vision_fingerprint_and_pils_from_files(uid: str, files: list) -> tuple[str, list]:
    """동일 업로드(이름·크기 메타 동일)면 디스크 재읽기·재압축 없이 세션의 PIL·핑거프린트를 재사용."""
    meta_key = f"img_upload_meta_{uid}"
    fp_key = f"img_content_fp_{uid}"
    pils_key = f"img_compressed_pils_{uid}"
    meta = _upload_files_meta_tuple(files)
    if (
        st.session_state.get(meta_key) == meta
        and fp_key in st.session_state
        and pils_key in st.session_state
    ):
        cached = list(st.session_state[pils_key])
        if cached:
            return str(st.session_state[fp_key]), cached
    pils, fp = uploaded_files_to_gemini_pil_images(files)
    if pils:
        st.session_state[meta_key] = meta
        st.session_state[fp_key] = fp
        st.session_state[pils_key] = pils
    else:
        st.session_state.pop(meta_key, None)
        st.session_state.pop(fp_key, None)
        st.session_state.pop(pils_key, None)
    return fp, pils


def _img_analysis_cache_sig_from_fp(fp: str, *, use_real_ai: bool, content: str) -> str:
    """압축 JPEG 기반 핑거프린트 + 모드. 실제 Vision 호출 경로에서는 메모(content)가 프롬프트에 없으므로 시그니처에서 제외."""
    force_sim = st.session_state.get("analyze_force_sim_mode", False)
    parts: list[str] = [fp or "empty", str(bool(use_real_ai)), str(bool(force_sim))]
    if (not use_real_ai) or force_sim:
        parts.append(hashlib.md5((content or "").encode("utf-8")).hexdigest()[:16])
    return "|".join(parts)


def _maybe_run_analyze_image(
    uid: str,
    img,
    *,
    use_real_ai: bool,
    content: str,
) -> tuple[list[dict], str, str]:
    """
    세션에 캐시된 시그니처와 같으면 Gemini Vision을 재호출하지 않는다.
    업로드 메타(이름·크기)가 같으면 압축 PIL·핑거프린트도 세션에서 재사용해 text_area 재실행 시 디스크 I/O를 줄인다.
    반환: (detected, suggested_unit, safety_advice)
    """
    files = _normalize_img_input(img)
    sig_key = f"img_analysis_sig_{uid}"
    result_key = f"img_result_{uid}"
    meta_key = f"img_upload_meta_{uid}"
    fp_key = f"img_content_fp_{uid}"
    pils_key = f"img_compressed_pils_{uid}"

    if not files:
        st.session_state.pop(meta_key, None)
        st.session_state.pop(fp_key, None)
        st.session_state.pop(pils_key, None)
        return (
            [{"객체": "사진 없음", "신뢰도": "—"}],
            "전자부품장착",
            "분석할 사진이 업로드되지 않았습니다.",
        )

    fp, pils = _vision_fingerprint_and_pils_from_files(uid, files)
    if not pils:
        return (
            [{"객체": "이미지 로드 실패", "신뢰도": "—"}],
            "전자부품장착",
            "이미지 파일을 열 수 없습니다. 다른 파일로 시도해 주세요.",
        )
    force_sim = st.session_state.get("analyze_force_sim_mode", False)
    sig = _img_analysis_cache_sig_from_fp(fp, use_real_ai=use_real_ai, content=content)

    if st.session_state.get(sig_key) == sig and result_key in st.session_state:
        t = st.session_state[result_key]
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            return list(t[0]), str(t[1]), str(t[2])
        if isinstance(t, (list, tuple)) and len(t) == 2:
            return list(t[0]), str(t[1]), ""

    primary_name = getattr(files[0], "name", "") if files else ""
    result = analyze_image(
        precompressed_pils=pils,
        use_real_api=use_real_ai and not force_sim,
        content=content or "",
        file_name=primary_name,
    )
    st.session_state[sig_key] = sig
    st.session_state[result_key] = result
    return result[0], result[1], result[2] if len(result) > 2 else ""


def _img_analysis_cache_hit(uid: str, img, *, use_real_ai: bool, content: str) -> bool:
    sig_key = f"img_analysis_sig_{uid}"
    result_key = f"img_result_{uid}"
    files = _normalize_img_input(img)
    if not files:
        return False
    fp, _ = _vision_fingerprint_and_pils_from_files(uid, files)
    sig = _img_analysis_cache_sig_from_fp(fp, use_real_ai=use_real_ai, content=content)
    return st.session_state.get(sig_key) == sig and result_key in st.session_state


def _evidence_validity_sig(uid: str, img, *, use_real_ai: bool, content: str) -> str:
    """본문·이미지(들) 조합이 같을 때만 증거 연관성 점수를 캐시."""
    files = _normalize_img_input(img)
    if not files:
        return ""
    fp, _ = _vision_fingerprint_and_pils_from_files(uid, files)
    base = _img_analysis_cache_sig_from_fp(fp, use_real_ai=use_real_ai, content=content)
    h = hashlib.md5((content or "").encode("utf-8")).hexdigest()[:16]
    return f"{base}|{h}"


def _get_google_api_key() -> str | None:
    """Gemini API 키: st.secrets 우선, 없으면 환경 변수 (bsr_utils.resolve_google_api_key)."""
    return resolve_google_api_key()


SYSTEM_PROMPT = """본 시스템은 공업고등학교 전기·전자과 실습 지도용 AI임. 학생 제출 실습 사진을 분석하여 아래 3항목을 반드시 답변함.

**1. 사진 속 주요 장비·기기**
멀티미터, 납땜기, 오실로스코프, 브레드보드, PCB, 전원공급기, 부품·IC, PLC 등 사진에 보이는 장비를 식별하고, 각각 신뢰도(추정%)를 부여함. 예: 멀티미터 (90%), 납땜기 (85%)

**2. NCS 단위 매칭**
해당 실습 활동이 NCS 국가직무능력표준의 어떤 단위와 가장 관련 있는지 판단함.
**전기·전자과 특성상 회로·부품·PCB·계측·임베디드·통신에 해당하면 전자 분야 능력단위를 우선 선택한다.**
반드시 아래 단위명 중 **하나만** 정확히 기재함(앞쪽이 전자·회로 중심):
""" + ncs_unit_names_for_prompt() + """

**3. 안전 수칙 조언**
학생의 안전 보호구(고글, 장갑, 안전화 등) 착용 여부 및 작업 환경의 안전성을 점검하여 조언함. 개선점이 있으면 구체적으로 기재하고, 양호한 경우 해당 사항을 명시함.

반드시 다음 형식으로만 답변함. 다른 내용은 작성하지 않음.

[장비]
- 장비1 (신뢰도%)
- 장비2 (신뢰도%)

[NCS단위]
단위명

[안전조언]
조언 내용"""

def _parse_ai_response(text: str) -> tuple[list[dict], str, str]:
    """AI 응답에서 장비 목록, NCS 단위, 안전 조언을 파싱."""
    detected: list[dict] = []
    suggested_unit = "전자부품장착"  # 기본값
    safety_advice = ""

    # [장비] 섹션 파싱: "- xxx (yy%)" 패턴
    equip_match = re.search(r"\[장비\](.*?)(?=\[NCS단위\]|\Z)", text, re.DOTALL)
    if equip_match:
        for line in equip_match.group(1).strip().split("\n"):
            m = re.search(r"[-•*]\s*(.+?)\s*\((\d+%?)\)", line.strip())
            if m:
                detected.append({"객체": m.group(1).strip(), "신뢰도": m.group(2) if "%" in m.group(2) else m.group(2) + "%"})
            elif line.strip() and not line.strip().startswith("["):
                detected.append({"객체": line.strip().lstrip("-•* "), "신뢰도": "—"})

    # [NCS단위] 섹션 파싱 (공백 제거 후 매칭)
    ncs_match = re.search(r"\[NCS단위\]\s*\n?\s*([^\n\[]+)", text)
    if ncs_match:
        raw = ncs_match.group(1).strip().replace(" ", "")
        if raw in NCS_DB:
            suggested_unit = raw
        else:
            for key in NCS_DB:
                if key.replace(" ", "") == raw or key in raw or raw in key.replace(" ", ""):
                    suggested_unit = key
                    break

    # [안전조언] 섹션 파싱
    safety_match = re.search(r"\[안전조언\](.*)", text, re.DOTALL)
    if safety_match:
        safety_advice = safety_match.group(1).strip()

    if not detected:
        detected = [{"객체": "이미지 분석 완료", "신뢰도": "—"}]
    return detected, suggested_unit, safety_advice


# 시뮬레이션 모드: 키워드 기반 맥락 부여 샘플 (파일명·텍스트 기반)
_SIM_SAMPLES: dict[str, str] = {
    "PLC": """[장비]
- PLC (88%)
- 래더 프로그래머 (82%)
- 입출력 모듈 (78%)
- 시퀀스 릴레이 (75%)

[NCS단위]
PLC제어

[안전조언]
전원 차단 후 결선 작업함. E-STOP 및 인터록 동작을 사전 점검할 것.""",
    "납땜": """[장비]
- 인두기 (90%)
- PCB (85%)
- 멀티미터 (80%)
- 플럭스 (75%)

[NCS단위]
전자부품장착

[안전조언]
고글·환기 유지. 인두기 정리 및 열선 안전 확인할 것.""",
    "계측": """[장비]
- 멀티미터 (90%)
- 오실로스코프 (85%)
- 메거 (80%)
- 테스터 (75%)

[NCS단위]
전자회로조립

[안전조언]
계측 전 무전압 확인. 프로브 절연 상태 점검할 것.""",
    "인버터": """[장비]
- 인버터 (88%)
- 모터 (85%)
- 파라미터 설정기 (80%)

[NCS단위]
인버터제어

[안전조언]
모터 접촉 시 회전 위험. 파라미터 변경 전 백업 확인할 것.""",
    "통신": """[장비]
- RS-485 모듈 (85%)
- Ethernet 스위치 (82%)
- Modbus 어댑터 (78%)

[NCS단위]
산업통신

[안전조언]
통신 케이블 차폐·접지 확인. 노드 주소 충돌 방지할 것.""",
}

_DEFAULT_SIM = """[장비]
- 멀티미터 (85%)
- PCB (80%)
- 납땜기 (90%)
- 브레드보드 (75%)

[NCS단위]
전자부품장착

[안전조언]
작업 시 고글 및 보호구 착용을 권장함. 인두기 사용 후 정리 상태를 확인할 것."""


def _get_simulation_response(file_name: str, content: str) -> str:
    """파일명·텍스트 키워드 기반 시뮬레이션 응답 선정."""
    combined = f"{file_name or ''} {content or ''}".lower()
    scores: dict[str, int] = {}
    for kw, sample in _SIM_SAMPLES.items():
        scores[kw] = combined.count(kw.lower()) + (2 if kw in (file_name or "") else 0)
    best = max(scores, key=scores.get)
    return _SIM_SAMPLES[best] if scores.get(best, 0) > 0 else _DEFAULT_SIM


# 장비명 → 본문 매칭용 유사 표현 (한글 축약, 영문 약어 등)
_EQUIP_ALIASES: dict[str, list[str]] = {
    "멀티미터": ["멀티", "테스터", "전압", "측정"],
    "인두기": ["인두", "납땜", "솔더"],
    "납땜기": ["인두", "납땜", "솔더"],
    "PLC": ["플씨", "래더", "시퀀스"],
    "PCB": ["기판", "회로기판", "피씨비"],
    "오실로스코프": ["오실로", "파형", "주파수"],
    "브레드보드": ["브레드", "점퍼"],
    "전선": ["전선", "배선", "결선"],
    "릴레이": ["릴레이", "계전기"],
    "모터": ["모터", "전동기"],
}


def _strip_polish_markdown(text: str) -> str:
    """다듬기 결과에서 마크다운 강조(별표·밑줄)를 제거해 순수 텍스트로 만든다."""
    if not text:
        return text
    out = text
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"\*([^*]+)\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    return out


def _build_polish_prompt(bsr_text: str, ncs_unit: str = "", ncs_element: str = "") -> str:
    """NCS 단위·요소를 반영한 다듬기 프롬프트 생성."""
    ncs_context = ""
    if ncs_unit and ncs_unit in NCS_DB:
        meta = NCS_DB[ncs_unit]
        kw = meta.get("keywords", [])[:12]
        elem = meta.get("elements", [])
        ncs_context = f"""
[참고] 이 실습은 NCS 능력단위 '{ncs_unit}'의 '{ncs_element or elem[0] if elem else ""}' 수행요소와 연관된다.
- 활용할 키워드·직무용어: {", ".join(kw)}
- 수행요소 예: {", ".join(elem)}
위 키워드·수행요소를 참고하여 단순한 동작을 전문적 기술 행위로 묘사하세요.
"""
    return f"""당신은 공업고등학교 NCS(국가직무능력표준) 수행준거 작성 전문가이자 교육공학 전문가입니다.
학생이 작성한 일상적 말투의 실습 성찰(What–So What–Now What)을 NCS 수행준거 양식의 격식 있는 문장으로 변환해 주세요.

【절대 규칙 — 위반 시 잘못된 응답으로 간주】
1. 절대로 별표 두 개, 밑줄 두 개 등 마크다운 강조 기호를 사용하지 말고 순수 텍스트만 출력할 것.
2. 반드시 [What], [So What], [Now What] 세 가지 태그를 모두 포함하여 단락을 나눌 것. 어느 하나라도 누락하면 안 됨.
3. 출력 본문에 코드 블록, 글머리표 기호만 있는 줄, 해시 제목(#)을 넣지 말 것.

【말투 변환 규칙】
- "~했어요", "~했음", "~했습니다" → "~할 수 있게 됨", "~를 확인하고 해결함", "~의 중요성을 인지함"
- "~해서", "~했더니" → "~를 수행한 결과", "~을 적용하여"
- 구어체·약어 → 공식적 NCS 직무표준 용어로 치환

【내용 보강 규칙】
- 단순한 동작 예시를 전문적 기술 행위로 확장하되, 원문에 없는 사실은 만들지 말 것
- 아래 NCS 단위 키워드·수행요소를 참고하여 원문 맥락에 맞게 구체화
{ncs_context}

【구조 유지 규칙】
- [What], [So What], [Now What] 순서로 각 태그 뒤에 한 칸 띄운 뒤 본문을 쓸 것
- [체크리스트: …]가 입력에 있으면 그대로 유지
- [성찰메타] 줄이 있으면 출력에 넣지 말 것
- 전문 용어(NCS·직무용어)는 정확히 보존
- 지나치게 길게 늘리지 말고, 핵심만 담음

【입력 텍스트】
---
{bsr_text}
---

위 내용을 NCS 수행준거 양식으로 다듬은 결과만 출력하세요. 설명·주석·머리말은 넣지 마세요."""


def _polish_bsr_with_gemini(bsr_text: str, ncs_unit: str = "", ncs_element: str = "") -> str | None:
    """Gemini API로 BSR 전체를 NCS 수행준거 양식으로 다듬기. ncs_unit/element로 내용 보강 참조. 실패 시 None."""
    api_key = _get_google_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        body = get_reflection_body(bsr_text)
        prompt = _build_polish_prompt(body, ncs_unit, ncs_element)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 2048},
        )
        if out:
            polished = _strip_polish_markdown(out.strip())
            meta = get_reflection_meta(bsr_text)
            if meta:
                polished = (
                    get_reflection_body(polished) or polished
                ).rstrip() + "\n[성찰메타]" + json.dumps(meta, ensure_ascii=False)
            return polished
    except Exception:
        pass
    return None


def _check_evidence_content_match(equip_names: list[str], content: str) -> bool:
    """
    탐지된 장비와 학생 본문의 연관성을 검사. 연관성이 너무 낮으면 False.
    (증거-텍스트 교차 검증용)
    """
    if not equip_names or not (content or "").strip():
        return True  # 판단 불가 시 통과
    text_raw = content.strip()
    text_lower = text_raw.lower()
    for eq in equip_names:
        eq_clean = (eq or "").strip()
        if len(eq_clean) < 2:
            continue
        if eq_clean in text_raw or eq_clean.lower() in text_lower:
            return True
        for alias in _EQUIP_ALIASES.get(eq_clean, []):
            if alias in text_raw or alias in text_lower:
                return True
        if len(eq_clean) >= 3 and (eq_clean[:3] in text_raw or eq_clean[:3] in text_lower):
            return True
        for token in eq_clean.replace("-", " ").split():
            if len(token) >= 2 and (token in text_raw or token in text_lower):
                return True
    return False


def _semantic_evidence_mismatch(equip_names: list[str], content: str, suggested_unit: str) -> bool:
    """
    사진·NCS 단위가 시사하는 직무 영역과 본문 키워드가 명백히 엇갈릴 때 True.
    (예: 사진·단위는 PLC인데 본문은 인버터만 서술)
    """
    text = (content or "").strip()
    if len(text) < 12:
        return False
    blob = " ".join(equip_names or []) + " " + (suggested_unit or "")
    # PLC 계열 신호
    plc_photo = any(
        k in blob
        for k in ("PLC", "래더", "시퀀스", "프로그래머", "입출력", "PLC제어")
    )
    plc_text = sum(1 for k in ("PLC", "래더", "시퀀스", "입출력", "프로그램") if k in text)
    inv_text = sum(1 for k in ("인버터", "VFD", "VF", "주파수 변환") if k in text)
    if plc_photo and inv_text >= 1 and plc_text == 0:
        return True
    # 인버터 계열 사진인데 본문만 PLC
    inv_photo = "인버터" in blob or "인버터제어" in (suggested_unit or "")
    if inv_photo and plc_text >= 1 and inv_text == 0:
        return True
    return False


def _gemini_vision_generate(genai, pil_imgs, prompt: str) -> tuple[str, str]:
    """이미지(여러 장 가능)+프롬프트로 텍스트 응답. (text, 사용한 모델명).

    ``[prompt, *PIL.Image]`` 형태로 ``generate_content``에 전달한다.
    안전 필터는 실습 묘사를 위해 BLOCK_NONE. 응답은 ``extract_generate_content_text``로 수집한다.
    """
    if not isinstance(pil_imgs, (list, tuple)):
        pil_imgs = [pil_imgs]
    pil_imgs = [p for p in pil_imgs if p is not None]
    if not pil_imgs:
        raise ValueError("분석할 이미지가 없습니다.")

    pil_rgb_copies = [p.convert("RGB").copy() for p in pil_imgs]
    for p in pil_rgb_copies:
        try:
            p.load()
        except Exception:
            pass
    payload: list = [prompt, *pil_rgb_copies]
    safety = gemini_safety_settings_block_none()
    gen_kwargs: dict = {"generation_config": {"temperature": 0.2, "max_output_tokens": 1024}}
    if safety:
        gen_kwargs["safety_settings"] = safety

    attempt_logs: list[str] = []
    last_err: Exception | None = None
    for model_name in resolved_gemini_model_candidates(genai):
        try:
            model = get_gemini_model(genai, model_name)
            if model is None:
                msg = f"{model_name}: GenerativeModel init failed"
                attempt_logs.append(msg)
                if model_name == GEMINI_PRIMARY_MODEL:
                    import logging
                    logging.getLogger("ai_final.gemini").warning(
                        "Gemini vision primary unavailable: %s", msg
                    )
                continue
            response = model.generate_content(payload, **gen_kwargs)
            text = extract_generate_content_text(response)
            if text:
                if model_name != GEMINI_PRIMARY_MODEL:
                    import logging
                    logging.getLogger("ai_final.gemini").warning(
                        "Gemini vision fallback model used: primary=%s used=%s",
                        GEMINI_PRIMARY_MODEL,
                        model_name,
                    )
                return text, model_name
            fr = ""
            try:
                c0 = (getattr(response, "candidates", None) or [None])[0]
                fr = str(getattr(c0, "finish_reason", "") or "")
            except Exception:
                pass
            msg = f"{model_name}: 빈 응답 (finish_reason={fr!r})"
            attempt_logs.append(msg)
            last_err = RuntimeError(GEMINI_EMPTY_RESPONSE_MESSAGE)
            if model_name == GEMINI_PRIMARY_MODEL:
                import logging
                logging.getLogger("ai_final.gemini").warning(
                    "Gemini vision primary empty: %s", msg
                )
        except Exception as e:
            attempt_logs.append(f"{model_name}: {e}")
            last_err = e
            if model_name == GEMINI_PRIMARY_MODEL:
                import logging
                logging.getLogger("ai_final.gemini").warning(
                    "Gemini vision primary failed: %s", e
                )
                if "429" in str(e) or "quota" in str(e).lower():
                    mark_primary_unavailable(str(e)[:180])
            continue
    detail = "\n".join(attempt_logs) if attempt_logs else "(시도 로그 없음)"
    raise RuntimeError(
        "모든 Gemini 이미지 모델에서 실패했습니다.\n\n" + detail
    ) from last_err


def analyze_image(
    image_file=None,
    *,
    precompressed_pils: list | None = None,
    use_real_api: bool = True,
    content: str = "",
    file_name: str = "",
) -> tuple[list[dict], str, str]:
    """
    실습 사진 분석. use_real_api=False이거나 Quota 초과 시 시뮬레이션 모드로 자동 전환.
    image_file은 단일 UploadedFile 또는 List[UploadedFile] 모두 허용.
    precompressed_pils가 주어지면 업로드 파일을 다시 읽지 않고 해당 PIL(JPEG 압축 완료본)만 Gemini에 보낸다.
    반환: (탐지된 장비 목록, 추천 NCS 단위, 안전 조언)
    """
    if precompressed_pils is not None:
        files: list = []
        pil_images: list = [p for p in precompressed_pils if p is not None]
    else:
        files = _normalize_img_input(image_file)
        pil_images = []
    primary_name = file_name or (getattr(files[0], "name", "") if files else "")

    use_sim_key = "analyze_force_sim_mode"
    if st.session_state.get(use_sim_key, False):
        use_real_api = False
    if not use_real_api:
        sim_text = _get_simulation_response(primary_name, content)
        return _parse_ai_response(sim_text)

    if precompressed_pils is None and not files:
        return (
            [{"객체": "사진 없음", "신뢰도": "—"}],
            "전자부품장착",
            "분석할 사진이 업로드되지 않았습니다.",
        )
    if precompressed_pils is not None and not pil_images:
        return (
            [{"객체": "사진 없음", "신뢰도": "—"}],
            "전자부품장착",
            "분석할 사진이 업로드되지 않았습니다.",
        )

    api_key = _get_google_api_key()
    if not api_key:
        st.error(
            "**Google AI API 키가 설정되지 않았습니다.**\n\n"
            "`.streamlit/secrets.toml`에 `GOOGLE_API_KEY = \"your-key\"` 를 추가하거나, "
            "환경 변수 `GOOGLE_API_KEY`를 설정해 주세요. "
            "**관리자에게 API 설정을 확인하세요.**\n\n"
            "[Google AI Studio](https://aistudio.google.com/apikey)에서 API 키를 발급받을 수 있습니다."
        )
        return (
            [{"객체": "API 키 미설정", "신뢰도": "—"}],
            "전자부품장착",
            "API 키를 설정한 후 다시 시도해 주세요.",
        )

    try:
        import google.generativeai as genai

        if not pil_images:
            pil_images, _fp = uploaded_files_to_gemini_pil_images(files)
            if not pil_images:
                st.warning(
                    "업로드한 모든 이미지를 불러오지 못했습니다. "
                    "파일이 손상되었거나 지원하지 않는 형식일 수 있어요. "
                    "JPG·PNG 이미지로 다시 업로드해 주세요."
                )
                return (
                    [{"객체": "이미지 로드 실패", "신뢰도": "—"}],
                    "전자부품장착",
                    "이미지 파일을 열 수 없습니다. 다른 파일로 시도해 주세요.",
                )
            if len(pil_images) < len(files):
                st.caption(
                    f"※ 업로드한 {len(files)}장 중 {len(files) - len(pil_images)}장은 불러오지 못해 제외했습니다."
                )

        # 다중 이미지일 때 시스템 프롬프트에 안내문 추가
        prompt = SYSTEM_PROMPT
        if len(pil_images) > 1:
            multi_note = (
                f"\n\n[중요] 아래 {len(pil_images)}장의 사진은 **동일한 학생이 같은 실습 시간에 찍은 사진들**이다. "
                "한 장씩 따로 보지 말고 **모든 사진을 종합해 하나의 실습 상황**으로 해석하라. "
                "여러 사진에 걸쳐 보이는 장비를 누락 없이 통합해서 보고하고, "
                "사진들 간 절차의 흐름(준비 → 측정 → 결과 등)이 보이면 안전 조언에 반영하라."
            )
            prompt = SYSTEM_PROMPT + multi_note

        genai.configure(api_key=api_key)
        response_text, _used_model = _gemini_vision_generate(genai, pil_images, prompt)

        if not response_text:
            st.warning(GEMINI_EMPTY_RESPONSE_MESSAGE)
            return (
                [{"객체": "분석 결과 없음", "신뢰도": "—"}],
                "전자부품장착",
                GEMINI_EMPTY_RESPONSE_MESSAGE,
            )

        detected, suggested_unit, safety_advice = _parse_ai_response(response_text)
        st.session_state.pop("analyze_force_sim_mode", None)
        return detected, suggested_unit, safety_advice

    except Exception as e:
        with st.expander("오류 상세 (개발자·관리자용)", expanded=False):
            st.code(str(e)[:4000])
            st.caption(
                "[Google AI Studio](https://aistudio.google.com/apikey)에서 발급한 키인지, "
                "Cloud에서 **Generative Language API** 사용·청구·할당량을 확인하세요. "
                "키에 **HTTP 리퍼러/앱 제한**이 있으면 로컬 Streamlit에서 404·403이 날 수 있습니다."
            )
            if st.session_state.get("analyze_force_sim_mode", False):
                st.warning(
                    "**시뮬 강제 모드**가 켜져 있으면 실제 API를 호출하지 않습니다. 아래를 눌러 끈 뒤 사진을 다시 올려 보세요."
                )
                _sim_btn_uid = str(st.session_state.get("user") or "guest")
                if st.button(
                    "시뮬 강제 모드 끄고 페이지 새로고침",
                    key=f"clear_force_sim_{_sim_btn_uid}",
                    width="stretch",
                ):
                    st.session_state.pop("analyze_force_sim_mode", None)
                    st.rerun()
        st.session_state["analyze_force_sim_mode"] = True
        st.warning(
            "**실시간 Gemini 이미지 분석에 연결하지 못했습니다.** "
            "아래 펼침 메시지를 참고해 키·API 설정을 점검해 주세요. "
            "그동안 **로컬 분석 모드**로 결과를 보여 드립니다."
        )
        st.info(
            "로컬 분석은 파일명·픽셀 기반 추정이라 실제 사진과 다를 수 있습니다. "
            "API가 정상이 되면 자동으로 고품질 분석으로 전환됩니다."
        )
        sim_text = _get_simulation_response(primary_name, content)
        return _parse_ai_response(sim_text)


def _detect_ncs_unit(content: str, image_hint: str | None = None) -> str:
    """텍스트 키워드로 NCS 매칭. 텍스트가 비거나 매칭 없으면 image_hint(사진 분석) 사용. 동점 시 전자 능력단위 우선."""
    text = (content or "").strip()
    scores: dict[str, int] = {}
    for unit, meta in NCS_DB.items():
        scores[unit] = 0
        for kw in meta.get("keywords", []):
            if kw and kw in text:
                scores[unit] += 1

    best_score = max(scores.values()) if scores else 0

    # 텍스트가 비었거나, 키워드 매칭이 없을 때 → 사진이 있으면 사진 힌트 사용
    if (not text or best_score == 0) and image_hint and image_hint in NCS_DB:
        return image_hint
    if best_score == 0:
        return "전자회로조립"  # 힌트도 없으면 전자 실습 기본값

    candidates = [u for u, s in scores.items() if s == best_score]
    if len(candidates) == 1:
        return candidates[0]
    for u in ELECTRONICS_NCS_UNITS:
        if u in candidates:
            return u
    return sorted(candidates)[0]


def _detect_element(unit: str, content: str) -> str:
    """NCS 능력단위별 세부 요소(Element) 매칭. 용산철도고 교과 범위 반영."""
    text = content or ""
    if unit == "PLC제어":
        if any(k in text for k in ["결선", "배선", "I/O", "입출력", "입출력결선"]):
            return "입출력 결선하기"
        if any(k in text for k in ["시운전", "테스트", "동작", "디버깅", "트러블"]):
            return "시운전하기"
        return "프로그램 작성하기"

    if unit == "전자부품장착":
        if any(k in text for k in ["납땜", "솔더", "솔더링"]):
            return "납땜하기"
        if any(k in text for k in ["검사", "불량", "테스터", "멀티미터", "측정", "수리", "고쳤", "핸드폰", "폰", "도통", "연속성"]):
            return "부품 검사하기"
        return "장착 상태 점검하기"

    if unit == "인버터제어":
        if any(k in text for k in ["파라미터", "설정", "주파수", "가감속", "VFD"]):
            return "파라미터 설정하기"
        if any(k in text for k in ["배선", "통신", "RS485", "Modbus", "연결"]):
            return "배선/통신 연결하기"
        return "운전 튜닝하기"

    if unit == "산업통신":
        if any(k in text for k in ["네트워크", "노드", "주소", "IP", "토폴로지"]):
            return "네트워크 구성하기"
        if any(k in text for k in ["장애", "타임아웃", "프레임", "오류"]):
            return "통신 장애 분석하기"
        return "장비 통신 설정하기"

    if unit == "모터제어":
        if any(k in text for k in ["회로", "결선", "MC", "OLR", "Y-Δ", "스타델타"]):
            return "회로 구성하기"
        if any(k in text for k in ["시퀀스", "운전", "정역", "역전"]):
            return "시퀀스 운전하기"
        return "보호장치 적용하기"

    if unit == "센서응용":
        if any(k in text for k in ["선정", "근접", "포토", "엔코더", "NPN", "PNP"]):
            return "센서 선정하기"
        if any(k in text for k in ["배선", "설치", "0-10V", "4-20mA"]):
            return "배선/설치하기"
        return "신호 점검하기"

    if unit == "마이크로컨트롤러":
        if any(k in text for k in ["GPIO", "PWM", "입출력", "LED"]):
            return "입출력 제어하기"
        if any(k in text for k in ["UART", "I2C", "SPI", "통신", "시리얼"]):
            return "통신 구현하기"
        return "디버깅하기"

    if unit == "전기안전":
        if any(k in text for k in ["위험", "파악", "LOTO", "차단"]):
            return "위험요인 파악하기"
        if any(k in text for k in ["PPE", "보호구", "고글", "장갑"]):
            return "안전조치 수행하기"
        return "점검 기록하기"

    if unit == "전기설비시공":
        if any(k in text for k in ["배관", "배선", "전선", "덕트", "트레이"]):
            return "배관·배선하기"
        if any(k in text for k in ["절연", "접지", "메거", "절연저항"]):
            return "절연·접지 점검하기"
        return "기기 설치하기"

    if unit == "전기설비유지보수":
        if any(k in text for k in ["정기", "점검", "열화상", "일정"]):
            return "정기 점검하기"
        if any(k in text for k in ["고장", "진단", "트러블", "이상"]):
            return "고장 진단하기"
        return "부품 교체하기"

    if unit == "전자회로조립":
        if any(k in text for k in ["준비", "부품", "극성", "데이터시트"]):
            return "부품 준비하기"
        if any(k in text for k in ["기능", "점검", "측정", "전압"]):
            return "기능 점검하기"
        return "회로 조립하기"

    if unit == "전자회로설계":
        if any(k in text for k in ["해석", "회로도", "이득", "주파수"]):
            return "회로 해석하기"
        if any(k in text for k in ["시뮬레이션", "SPICE", "검증"]):
            return "시뮬레이션/검증하기"
        return "회로 설계하기"

    if unit == "PCB설계":
        if any(k in text for k in ["배치", "부품 배치", "레이아웃"]):
            return "부품 배치하기"
        if any(k in text for k in ["라우팅", "패턴", "GND", "비아"]):
            return "패턴 라우팅하기"
        return "DRC/제조데이터 출력하기"

    if unit == "임베디드하드웨어설계":
        if any(k in text for k in ["사양", "정의", "요구사항"]):
            return "시스템 사양 정의하기"
        if any(k in text for k in ["레이아웃", "검증"]):
            return "레이아웃 검증하기"
        if any(k in text for k in ["회로", "스키매틱", "센서", "MCU"]):
            return "회로 설계하기"
        return "회로 설계하기"

    if unit == "임베디드소프트웨어개발":
        if any(k in text for k in ["디버깅", "검증"]):
            return "디버깅·검증하기"
        if any(k in text for k in ["코드", "구현", "C", "드라이버"]):
            return "코드 구현하기"
        return "펌웨어 설계하기"

    if unit == "반도체제조":
        if any(k in text for k in ["웨이퍼", "준비"]):
            return "웨이퍼 준비하기"
        if any(k in text for k in ["검사", "측정", "품질"]):
            return "품질 검사하기"
        if any(k in text for k in ["공정", "포토", "에칭", "박막", "리소그래피"]):
            return "공정 수행하기"
        return "공정 수행하기"

    if unit == "통신기기하드웨어개발":
        if any(k in text for k in ["RF", "안테나", "기저대역"]):
            return "RF/기저대역 구현하기"
        if any(k in text for k in ["회로", "통신"]):
            return "통신 회로 설계하기"
        return "통신 회로 설계하기"

    if unit == "디지털방송기기개발":
        if any(k in text for k in ["인코딩", "디코딩", "부호화"]):
            return "인코딩/디코딩 구현하기"
        if any(k in text for k in ["신호", "방송"]):
            return "방송 신호 처리 설계하기"
        return "방송 신호 처리 설계하기"

    if unit == "스마트가전기기개발":
        if any(k in text for k in ["인터페이스", "설계"]):
            return "IoT 인터페이스 설계하기"
        if any(k in text for k in ["연동", "검증", "센서", "IoT"]):
            return "연동 검증하기"
        return "IoT 인터페이스 설계하기"

    return NCS_DB.get(unit, {}).get("elements", ["해당 요소"])[0]


def _build_bsr_string(background: str, haegyul: str, seungwa: str, checked_items: list[str], meta: dict | None = None) -> str:
    """What–So What–Now What 문자열. 인자명은 기존 호출부 호환용."""
    return build_reflection_string(
        background,
        haegyul,
        seungwa,
        meta=meta,
        checked_items=checked_items,
    )


def _render_bsr_reflection_card_html(
    background: str,
    haegyul: str,
    seungwa: str,
    checked_items: list[str],
    polished: str | None,
) -> str:
    """원문 vs AI 다듬기 2열 + 단계별 화살표 (교육용 BSR 미리보기)."""
    pol = (polished or "").strip()
    pb = extract_bsr_section(pol, "What") or extract_bsr_section(pol, "배경") if pol else ""
    ph = extract_bsr_section(pol, "So What") or extract_bsr_section(pol, "해결") if pol else ""
    ps = extract_bsr_section(pol, "Now What") or extract_bsr_section(pol, "성과") if pol else ""
    pchk_m = re.search(r"\[체크리스트:[^\]]*\]", pol) if pol else None
    pchk = pchk_m.group(0) if pchk_m else ""
    ochk = f"[체크리스트: {'; '.join(checked_items)}]" if checked_items else ""

    def _orig_body(mini: str) -> str:
        if not (mini or "").strip():
            return "<div class='bsr-col-body'><span class='bsr-placeholder'>(내용 없음)</span></div>"
        return f"<div class='bsr-col-body'>{render_bsr_highlighted(mini.strip())}</div>"

    def _ref_body(mini: str) -> str:
        if not pol:
            return (
                "<div class='bsr-col-body bsr-col-body--empty'><span class='bsr-placeholder'>"
                "AI 전문 문장으로 다듬기 후 표시됩니다</span></div>"
            )
        if not (mini or "").strip():
            return (
                "<div class='bsr-col-body'><span class='bsr-placeholder'>(내용 없음)</span></div>"
            )
        return f"<div class='bsr-col-body'>{render_bsr_highlighted(mini.strip())}</div>"

    def _pair(title: str, orig_mini: str, ref_mini: str) -> str:
        return (
            f"<h4 class='bsr-reflection-h4'>{html.escape(title)}</h4>"
            "<div class='bsr-pair-grid'>"
            "<div class='bsr-col bsr-col--original'>"
            "<span class='bsr-col-label'>작성 원문</span>"
            f"{_orig_body(orig_mini)}</div>"
            "<div class='bsr-col bsr-col--refined'>"
            "<span class='bsr-col-label'>AI 다듬기</span>"
            f"{_ref_body(ref_mini)}</div>"
            "</div>"
        )

    chunks: list[str] = ["<div class='bsr-reflection-card'>"]

    chunks.append(_pair("What — 실무 경험", f"[What] {background or ''}", f"[What] {pb}" if pol else ""))
    chunks.append("<div class='bsr-flow-divider' aria-hidden='true'></div>")
    chunks.append(_pair("So What — 판단 및 성찰", f"[So What] {haegyul or ''}", f"[So What] {ph}" if pol else ""))
    chunks.append("<div class='bsr-flow-divider' aria-hidden='true'></div>")
    chunks.append(_pair("Now What — 향후 적용", f"[Now What] {seungwa or ''}", f"[Now What] {ps}" if pol else ""))

    if checked_items or pchk:
        chunks.append("<div class='bsr-flow-divider' aria-hidden='true'></div>")
        o_chk = ochk if ochk else "[체크리스트: ]"
        o_html = _orig_body(o_chk)
        if pol and pchk:
            r_html = f"<div class='bsr-col-body'>{render_bsr_highlighted(pchk)}</div>"
        elif pol:
            r_html = (
                "<div class='bsr-col-body bsr-col-body--empty'><span class='bsr-placeholder'>"
                "AI 다듬기 결과에 체크리스트가 없습니다</span></div>"
            )
        else:
            r_html = (
                "<div class='bsr-col-body bsr-col-body--empty'><span class='bsr-placeholder'>"
                "AI 전문 문장으로 다듬기 후 표시됩니다</span></div>"
            )
        chunks.append(
            "<h4 class='bsr-reflection-h4'>수행준거 체크리스트</h4>"
            "<div class='bsr-pair-grid'>"
            "<div class='bsr-col bsr-col--original'>"
            "<span class='bsr-col-label'>작성 원문</span>"
            f"{o_html}</div>"
            "<div class='bsr-col bsr-col--refined'>"
            "<span class='bsr-col-label'>AI 다듬기</span>"
            f"{r_html}</div>"
            "</div>"
        )

    chunks.append("</div>")
    return "".join(chunks)


def _convert_to_ncs_terms(text: str) -> list[tuple[str, str, str]]:
    """학생이 쉽게 말한 구어를 NCS 직무표준 용어로 변환하여 (구어, NCS용어, 설명) 반환."""
    if not (t := (text or "").strip()):
        return []
    t_lower = t.lower()
    found: list[tuple[str, str, str]] = []
    for phrases, ncs_term, desc in COLLOQUIAL_TO_NCS:
        for p in phrases:
            if p in t or p.lower() in t_lower:
                found.append((p, ncs_term, desc))
                break
    return found


def _rewrite_to_ncs_terms_fallback(text: str) -> str:
    """사전 치환 기반 폴백 (API 실패 시)."""
    if not (t := (text or "").strip()):
        return ""
    t_lower = t.lower()
    replacements: list[tuple[str, str]] = []
    for phrases, ncs_term, _ in COLLOQUIAL_TO_NCS:
        for p in phrases:
            if not p:
                continue
            if p in t:
                replacements.append((p, ncs_term))
                break
            if p.lower() in t_lower:
                idx = t_lower.find(p.lower())
                actual = t[idx : idx + len(p)]
                replacements.append((actual, ncs_term))
                break
    replacements.sort(key=lambda x: -len(x[0]))
    result = t
    for phrase, ncs_term in replacements:
        result = result.replace(phrase, ncs_term)
    return result


REWRITE_NCS_PROMPT = """당신은 공업고등학교 NCS(국가직무능력표준) 수행준거 작성 전문가입니다.
학생의 구어체 실습 기록을 **NCS 수행준거 양식(~할 수 있다, ~함)**과 전문 기술 용어를 사용하여 격식 있는 전문가 톤으로 다듬어라.

규칙:
1. 없는 사실을 지어내지 말고, 학생이 쓴 핵심 동작(예: 납땜, 패턴도 확인, PLC 프로그래밍 등)은 반드시 포함한다.
2. 출력 형식: [능력단위명(NCS코드)] ... 내용 ... 형태로 시작. 예: [전자부품장착(1902020101_16v3)] 설계된 패턴도를 분석하여 회로의 연결성을 확인하고, 규격에 맞는 납땜 작업을 통해 부품 장착을 완료함.
3. 능력단위·코드는 입력 내용에서 추론(납땜·PCB·전자→전자부품장착 1902020101_16v3, PLC·래더→PLC제어 1902050106_14v1 등). 확실하지 않으면 전자부품장착 등 보수적으로 선택.
4. ~함, ~완료함, ~확인함 등 수행준거 표현 사용. 짧은 입력이면 1문장으로 완결.
5. 입력에 [What][So What][Now What] 또는 [배경][해결][성과] 구조가 있으면 각 구간을 유지하며 다듬되, 원문에 없는 내용은 추가하지 말 것.

입력 (학생 구어체):
---
{text}
---

다듬은 결과만 출력. 설명·주석 없음."""


def _rewrite_to_ncs_terms_with_gemini(text: str) -> str | None:
    """Gemini API로 구어체를 NCS 전문가 톤 1문장으로 변환. 실패 시 None."""
    api_key = _get_google_api_key()
    if not api_key or not (t := (text or "").strip()) or len(t) < 5:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        prompt = REWRITE_NCS_PROMPT.format(text=t)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 512},
        )
        if out:
            return out
    except Exception:
        pass
    return None


def _rewrite_to_ncs_terms(text: str, use_gemini: bool = True) -> str:
    """실습 기록을 NCS 표준용어로 풀어써서 반환. use_gemini=True이면 Gemini AI 활용, 실패 시 사전 치환 폴백."""
    if not (t := (text or "").strip()):
        return ""
    if use_gemini:
        result = _rewrite_to_ncs_terms_with_gemini(t)
        if result:
            return result
    return _rewrite_to_ncs_terms_fallback(t)


# ─────────────────────────────────────────────────────────────────
# 2-Turn 스캐폴딩(채팅형) — AI가 메모를 보고 2번의 심화 질문을 던진다
# ─────────────────────────────────────────────────────────────────
_SCAFFOLD_GREETING_RE = re.compile(
    r"안녕하세요|안녕하십니까|무엇을 도와드릴|어떻게 도와드릴|"
    r"무엇을 하고 싶으|궁금한 점.? 있으|도움이 필요"
)
# thinking이 없는/약한 모델을 먼저 써서 질문이 중간에 끊기지 않게 한다.
_SCAFFOLD_PREFERRED_MODELS = (
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)


def _memo_snippet(text: str, n: int = 28) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    if not s:
        return ""
    return s if len(s) <= n else (s[:n].rstrip() + "…")


def _is_generic_scaffold_greeting(text: str) -> bool:
    t = (text or "").strip()
    return (not t) or bool(_SCAFFOLD_GREETING_RE.search(t))


def _looks_like_complete_scaffold_utterance(text: str) -> bool:
    """피드백 등 비질문 응답이 중간에 잘렸는지 판별.

    문장 중간의 마침표가 아니라, 응답 전체가 완전한 종결로 끝나는지 본다.
    """
    t = (text or "").strip()
    if len(t) < 12 or _is_generic_scaffold_greeting(t):
        return False
    return bool(re.search(r"(?:[.!?。！]|요|다|까|죠)\s*$", t))


def _looks_like_complete_scaffold_question(text: str) -> bool:
    """2-Turn 질문은 반드시 물음표로 끝나야 하며, 잘린 문장·인사는 탈락."""
    t = (text or "").strip()
    if len(t) < 12 or _is_generic_scaffold_greeting(t):
        return False
    if "?" not in t and "？" not in t:
        return False
    # 조사·명사에서 끊긴 문장("…측정 작업을")은 탈락
    if re.search(r"(을|를|이|가|은|는|와|과|로|으로|의|작업|과정|부분)\s*$", t):
        return False
    return True


def _clean_scaffold_question(text: str) -> str:
    cleaned = (text or "").strip().strip('"“”').strip()
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    kept = [ln for ln in lines if not _SCAFFOLD_GREETING_RE.search(ln)]
    cleaned = " ".join(kept) if kept else cleaned
    if "?" in cleaned or "？" in cleaned:
        q_idx = max(cleaned.rfind("?"), cleaned.rfind("？"))
        if q_idx >= 0:
            cleaned = cleaned[: q_idx + 1].strip()
    return cleaned


def _fallback_turn1_question(memo: str, ncs_unit: str = "") -> str:
    unit = (ncs_unit or "").strip()
    unit_bit = f"{unit} 기준으로 " if unit else ""
    return (
        f"오늘 수행한 작업을 {unit_bit}떠올려 볼 때, "
        "가장 까다로웠던 부분과 그 문제를 어떤 순서로 확인하고 해결했나요?"
    )


def _fallback_turn2_question(memo: str, answer1: str) -> str:
    from bsr_utils import fallback_now_what_question

    return fallback_now_what_question({"task_type": "general", "raw_input": memo or ""}, answer1)


def _gemini_followup_question(prompt: str, *, require_question: bool = True) -> str | None:
    """Gemini로 스캐폴딩 한 턴을 생성. 잘리거나 인사만 오면 None.

    Gemini 2.x thinking 모델은 ``max_output_tokens``를 내부 토큰이 잠식해
    질문이 중간에 끊긴다. thinking을 끄고, 가벼운 모델을 먼저 쓰며,
    불완전한 문장은 화면에 내보내지 않는다.
    """
    api_key = _get_google_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        seen: set[str] = set()
        names: list[str] = []
        for n in list(_SCAFFOLD_PREFERRED_MODELS) + resolved_gemini_model_candidates(genai):
            low = (n or "").lower()
            if not n or n in seen:
                continue
            if "pro" in low or "imagen" in low or "embed" in low:
                continue
            seen.add(n)
            names.append(n)
        names = names[:6]
        safety = gemini_safety_settings_block_none()
        configs = (
            {
                "temperature": 0.4,
                "max_output_tokens": 1024,
                "thinking_config": {"thinking_budget": 0},
            },
            {"temperature": 0.4, "max_output_tokens": 1024},
        )
        for name in names:
            model = get_gemini_model(genai, name)
            if model is None:
                continue
            for gc in configs:
                try:
                    kwargs: dict = {"generation_config": dict(gc)}
                    if safety:
                        kwargs["safety_settings"] = safety
                    response = model.generate_content(prompt, **kwargs)
                    text = extract_generate_content_text(response)
                    if not text:
                        continue
                    cleaned = (
                        _clean_scaffold_question(text)
                        if require_question
                        else (text or "").strip().strip('"').strip()
                    )
                    if require_question:
                        if _looks_like_complete_scaffold_question(cleaned):
                            return cleaned
                    elif _looks_like_complete_scaffold_utterance(cleaned):
                        return cleaned
                except Exception:
                    continue
    except Exception:
        pass
    return None


def _scaffold_turn1_question(memo: str, ncs_unit: str = "", analysis: dict | None = None) -> str:
    """[Turn 1] So What? — 판단·기준·방법을 하나만 묻는다."""
    ana = analysis or analyze_practice_experience(memo, [], ncs_unit)
    return generate_so_what_question(ana)


def _scaffold_turn2_question(
    memo: str,
    answer1: str,
    analysis: dict | None = None,
    turn1_question: str = "",
) -> str:
    """[Turn 2] Now What? — Turn 1 답변을 다음 실습으로 전이한다."""
    ana = analysis or analyze_practice_experience(memo, [], "")
    return generate_now_what_question(ana, turn1_question, answer1)


def _scaffold_build_final_bsr(
    memo: str, answer1: str, answer2: str, detected_list: list[dict], analysis: dict | None = None
) -> dict[str, str]:
    """메모 + 1·2차 답변을 What–So What–Now What 초안으로 종합."""
    ana = analysis or analyze_practice_experience(memo, detected_list or [], "")
    try:
        draft = generate_reflection_draft(ana, answer1, answer2)
        return {
            "what": draft.get("what", ""),
            "so_what": draft.get("so_what", ""),
            "now_what": draft.get("now_what", ""),
            "background": draft.get("what", ""),
            "solution": draft.get("so_what", ""),
            "reflection": draft.get("now_what", ""),
        }
    except Exception:
        return {
            "what": "",
            "so_what": "",
            "now_what": "",
            "background": "",
            "solution": "",
            "reflection": "",
        }


def _scaffold_final_feedback(memo: str, answer1: str, answer2: str) -> str:
    """학생이 적은 내용만 바탕으로 한 짧은 피드백. 없는 전문성을 보태지 않는다."""
    prompt = f"""당신은 공업고등학교 전기·전자과 실습 지도교사입니다.
아래 학생이 실제로 적은 내용만 근거로 2~3문장 피드백을 쓰세요.
학생이 말하지 않은 장비·기술·성과를 만들지 마세요. 과장된 칭찬은 하지 마세요.
머리말·번호 없이 존댓말 문단만 출력하세요.

[학생 메모]
{(memo or '').strip()[:1200]}

[So What 답변]
{(answer1 or '').strip()[:1200]}

[Now What 답변]
{(answer2 or '').strip()[:1200]}
"""
    out = _gemini_followup_question(prompt, require_question=False)
    if out:
        return out
    return (
        "확인한 기준과 다음에 적용하고 싶은 점을 구체적으로 적어 주셔서, "
        "같은 작업을 반복할 때 점검 순서를 잡기 좋습니다. "
        "다음 실습에서도 오늘 말한 확인 방법을 한 단계씩 적용해 보시면 됩니다."
    )


AI_GROWTH_PROMPT = """당신은 공업고등학교 NCS 직무 역량 코치입니다. 학생의 실습 성찰(What–So What–Now What) 이력을 분석하여 맞춤형 성장 조언을 작성해 주세요.

다음 4가지를 **전문가 톤**으로 작성하세요. 각 항목은 반드시 `[1]` ~ `[4]` 레이블로 시작하고, 항목당 2~4문장.

[1] 현재 가장 뛰어난 직무 강점: 구체적 사례를 들어 학생의 강점을 칭찬하세요.

[2] 보완이 필요한 성찰 포인트: 메타인지·과정 서술 강화 방향을 제시하세요.

[3] 🏆 나의 베스트 실습 순간: 학생의 전체 일지 중 What–So What–Now What 연결이 가장 잘 드러나고 판단·전이가 돋보이는 최고의 일지 1건을 선정해. 선정된 일지의 날짜/제목을 명시하고, 어떤 점이 훌륭했는지 2~3문장으로 구체적으로 칭찬해 줘.

[4] 🚀 레벨업 미션: 학생의 최근 실습 패턴과 강점을 분석하여, 다음번 실습 현장에서 바로 시도해 볼 수 있는 구체적인 행동 미션 1가지를 제안해. 미션 내용과 그것을 달성했을 때의 기대 효과를 실무자의 관점에서 작성해 줘.

성찰 일지 이력:
---
{bsr_history}
---

[1]부터 [4]까지 레이블을 유지한 채 본문만 출력하세요."""


def _parse_ai_growth_report_sections(text: str) -> dict[str, str]:
    """AI 성장 총평 응답을 [1]~[4] 구간별로 분리."""
    raw = (text or "").strip()
    if not raw:
        return {}
    marker_re = re.compile(r"(?:^|\n)\s*\[([1-4])\]\s*", re.MULTILINE)
    matches = list(marker_re.finditer(raw))
    key_map = {
        "1": "strength",
        "2": "reflection",
        "3": "best_moment",
        "4": "mission",
    }
    out: dict[str, str] = {}
    for i, match in enumerate(matches):
        num = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()
        key = key_map.get(num)
        if key and body:
            out[key] = body
    return out


def _growth_summary_for_card(text: str) -> str:
    """카드 1(성장 총평)에 표시할 [1]+[2] 본문. 파싱 실패 시 원문."""
    sections = _parse_ai_growth_report_sections(text)
    parts: list[str] = []
    if sections.get("strength"):
        parts.append(sections["strength"])
    if sections.get("reflection"):
        parts.append(sections["reflection"])
    if parts:
        return "\n\n".join(parts)
    return text


def _get_ai_growth_report(bsr_logs: list[dict]) -> str | None:
    """Gemini API로 BSR 이력 기반 AI 맞춤형 성장 총평 생성. 실패 시 None."""
    if not bsr_logs:
        return None
    api_key = _get_google_api_key()
    if not api_key:
        return None
    history = "\n\n".join(
        f"[{r.get('date','')}] {r.get('ncs_unit','')}\n{get_reflection_body(str(r.get('bsr','')))[:800]}"
        for r in bsr_logs[:15]
    )
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        prompt = AI_GROWTH_PROMPT.format(bsr_history=history)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 2048},
        )
        if out:
            return out
    except Exception:
        pass
    return None


def _seungwa_char_count(bsr: str) -> int:
    """So What / Now What 또는 레거시 [성과] 글자 수 (성찰 깊이 근사)."""
    rec = parse_reflection_record(bsr or "")
    body = rec.get("so_what") or rec.get("now_what") or rec.get("legacy_reflection") or ""
    return len(str(body).strip())


def _professional_term_hits(bsr: str) -> int:
    """BSR에 등장하는 서로 다른 GLOSSARY·NCS 키워드 개수(빈도 근사)."""
    t = get_reflection_body(bsr or "")
    seen: set[str] = set()
    for term in GLOSSARY:
        if term in t:
            seen.add(term)
    for meta in NCS_DB.values():
        for kw in meta.get("keywords", []):
            if kw and len(kw) >= 2 and kw in t:
                seen.add(kw)
    return min(len(seen), 50)


def _build_last3_meta_stats_block(logs: list[dict]) -> str:
    """최근 3개 일지의 성찰·용어 지표를 Gemini용 텍스트로 정리."""
    recent = logs[:3]
    lines: list[str] = []
    for i, row in enumerate(recent, start=1):
        bsr = get_reflection_body(str(row.get("bsr") or ""))
        date = row.get("date", "")
        unit = row.get("ncs_unit", "")
        sc = _seungwa_char_count(str(row.get("bsr") or ""))
        th = _professional_term_hits(str(row.get("bsr") or ""))
        lines.append(
            f"일지 {i} [{date}] 능력단위:{unit}\n"
            f"  - 성찰(So What/Now What) 글자 수: {sc}자\n"
            f"  - 전문 용어 매칭 빈도(근사): {th}\n"
            f"  - 일지 앞부분 요약: {(bsr[:200] + '…') if len(bsr) > 200 else bsr}"
        )
    return "\n\n".join(lines)


AI_META_COACH_PROMPT = """당신은 공업고등학교 NCS 직무 역량을 지도하는 교육·메타인지 코치입니다.

아래는 한 학생의 **최근 3개 실습 일지**에 대해 산출한 지표입니다. 각 일지마다 So What/Now What 구간 글자 수(성찰 깊이 근사)와 전문 용어 매칭 빈도를 비교할 수 있습니다.

---
{stats_block}
---

**작성 지침**
1. 세 일지를 비교하여 **성장한 점**을 구체적으로 칭찬하세요 (성찰의 깊이, 전문 용어 사용 측면).
2. **보완할 점**을 구체적으로 제시하세요. 특히 So What·Now What 구간이 짧거나 메타인지적 표현(이유, 판단, 다음에는 등)이 부족한 경우를 짚어 주세요.
3. 전문가 톤, 2~4문단. 번호·마크다운 제목 없이 본문만 서술하세요."""


def _get_ai_meta_coach_comment(logs: list[dict]) -> str | None:
    """최근 3개 일지 기반 메타인지·성장 코멘트 (Gemini). 실패 시 None."""
    if not logs:
        return None
    api_key = _get_google_api_key()
    if not api_key:
        return None
    stats_block = _build_last3_meta_stats_block(logs)
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        prompt = AI_META_COACH_PROMPT.format(stats_block=stats_block)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.45, "max_output_tokens": 1200},
        )
        if out:
            return out
    except Exception:
        pass
    return None


def _log_competency_scores(bsr_text: str) -> dict[str, float]:
    """BSR 텍스트에서 역량 차원 점수 (구체성, 전문용어, 안전, 성찰)."""
    text = get_reflection_body(bsr_text or "").strip()
    length = min(5, max(0, (len(text) // 30) + 1))
    all_kw = set(GLOSSARY.keys())
    for meta in NCS_DB.values():
        all_kw.update(meta.get("keywords", []))
    term = min(5, max(0, sum(1 for w in all_kw if w in text) + 1))
    safety = min(5, max(0, sum(text.count(k) for k in ["안전", "접지", "감전", "보호구", "LOTO", "ELB", "차단기"]) + 1))
    high_w = ["깨달음", "성찰", "과정", "이유", "개선", "다음에는", "배운", "스스로", "이해", "알게"]
    reflection = min(5, max(0, sum(2 for w in high_w if w in text) + 1))
    return {"구체성": length, "전문용어": term, "안전": safety, "성찰": min(reflection, 5)}


def _evaluate_seungwa_reflection(bsr_logs: list[dict]) -> tuple[str, str]:
    """BSR 로그에서 성찰 수준(높음/보통/낮음)과 코멘트 반환."""
    high_words = {"깨달음", "성찰", "과정", "이유", "개선", "다음에는", "배운", "어려웠던", "스스로", "이해", "알게", "생각", "판단", "고민"}
    medium_words = {"확인", "점검", "수행", "적용", "이해함", "배웠"}
    low_patterns = ["했다", "됐다", "완료", "끝냄", "했다."]
    scores: list[int] = []
    for row in bsr_logs:
        bsr = (row.get("bsr") or "").strip()
        rec = parse_reflection_record(bsr)
        seg = (
            rec.get("so_what")
            or rec.get("now_what")
            or rec.get("legacy_reflection")
            or ""
        ).strip()
        if not seg:
            continue
        score = 0
        if len(seg) >= 50:
            score += 2
        elif len(seg) >= 20:
            score += 1
        for w in high_words:
            if w in seg:
                score += 2
                break
        for w in medium_words:
            if w in seg:
                score += 1
                break
        for p in low_patterns:
            if p in seg and len(seg) < 30:
                score -= 1
                break
        scores.append(max(0, score))
    if not scores:
        return "—", "성찰 구간이 없어 수준을 평가할 수 없습니다."
    avg = sum(scores) / len(scores)
    if avg >= 3:
        return "높음", "과정·이유·개선점을 구체적으로 서술하여 메타인지적 성찰 수준이 높습니다."
    if avg >= 1.5:
        return "보통", "기본적인 수행 중심 기술에 일부 성찰 요소가 포함되어 있습니다."
    return "낮음", "결과 중심의 간단한 서술 위주이며, 성찰 키워드 보완을 권장합니다."


def _extract_used_professional_terms(logs: list[dict]) -> list[str]:
    """일지에서 사용된 NCS·GLOSSARY 매칭 전문 용어 목록 (중복 제거, 사용 빈도순)."""
    all_terms: set[str] = set(GLOSSARY.keys())
    for meta in NCS_DB.values():
        all_terms.update(k for k in meta.get("keywords", []) if k and len(k) >= 2)
    for _, ncs_term, _ in COLLOQUIAL_TO_NCS:
        if ncs_term and len(ncs_term) >= 2:
            all_terms.add(ncs_term)
    used: dict[str, int] = {}
    for row in logs:
        text = get_reflection_body(row.get("bsr") or "").strip()
        for term in sorted(all_terms, key=len, reverse=True):
            if term in text:
                used[term] = used.get(term, 0) + 1
    return [t for t, _ in sorted(used.items(), key=lambda x: -x[1])]


def _compute_ncs_term_ratio(bsr_text: str) -> float:
    """BSR 내 구어체 대비 NCS 표준 용어 사용 비율(0~100)."""
    if not (t := get_reflection_body(bsr_text or "").strip()):
        return 0.0
    all_ncs: set[str] = set(GLOSSARY.keys())
    for meta in NCS_DB.values():
        all_ncs.update(k for k in meta.get("keywords", []) if k and len(k) >= 2)
    for _, ncs_term, _ in COLLOQUIAL_TO_NCS:
        if ncs_term and len(ncs_term) >= 2:
            all_ncs.add(ncs_term)
    ncs_found = sum(1 for term in all_ncs if term in t)
    word_count = max(1, len([w for w in re.sub(r"[^\w\s]", " ", t).split() if len(w) >= 2]))
    return min(100.0, round(100.0 * ncs_found / min(25, word_count), 1))


def _ncs_experience_counts_from_logs(logs: list[dict]) -> dict[str, int]:
    """저장된 일지의 ncs_unit 매핑 건수. 기본 능력단위는 0건으로 포함한다."""
    counts: dict[str, int] = {str(k): 0 for k in DEFAULT_NCS_PROGRESS}
    known = list(counts.keys())
    for row in logs or []:
        raw = str(row.get("ncs_unit") or "").strip()
        if not raw:
            continue
        matched = ""
        for k in known:
            if k == raw or k in raw:
                matched = k
                break
        if not matched:
            cleaned = _clean_ncs_unit_name(raw)
            for k in known:
                if cleaned and (k == cleaned or k in cleaned):
                    matched = k
                    break
            if not matched:
                matched = cleaned or raw
                if matched not in counts:
                    counts[matched] = 0
        counts[matched] = counts.get(matched, 0) + 1
    return counts


def _render_ncs_progress_section(uid: str, *, compact: bool = True) -> None:
    """NCS 기반 실무 경험 현황 — 저장된 일지 건수 기준 표시(이수율·진도율 아님)."""
    st.markdown('<div class="ncs-block">', unsafe_allow_html=True)
    st.markdown("#### NCS 기반 실무 경험 현황")
    logs_for_chart = list_logs(uid)
    counts = _ncs_experience_counts_from_logs(logs_for_chart)

    if not logs_for_chart:
        st.info("저장된 실습일지가 없습니다. 일지를 저장하면 능력단위별 경험 축적 현황이 표시됩니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    linked = sum(1 for v in counts.values() if v > 0)
    mc1, mc2 = st.columns(2)
    mc1.metric("기록이 연결된 능력단위", f"{linked}개")
    mc2.metric("누적 실습 일지", f"{len(logs_for_chart)}건")

    with st.expander("NCS 능력단위별 실무 경험 축적 현황", expanded=True):
        for unit, n in counts.items():
            label = format_ncs_unit(unit)
            if n > 0:
                st.markdown(f"**{label}** · {n}건")
            else:
                st.caption(f"{label} · 0건")

    if compact:
        col_bar = st.container()
        col_radar = st.container()
    else:
        col_bar, col_radar = st.columns(2)

    with col_bar:
        if compact:
            st.caption("NCS 능력단위별 실무 경험 분포")
        else:
            st.markdown("**NCS 능력단위별 실무 경험 분포**")
        bar_items = [(format_ncs_unit(u), n) for u, n in counts.items() if n >= 1]
        if bar_items:
            bar_df = pd.DataFrame(
                {
                    "단위": [u for u, _ in bar_items],
                    "기록 건수": [n for _, n in bar_items],
                }
            )
            bar_fig = px.bar(
                bar_df,
                x="기록 건수",
                y="단위",
                orientation="h",
                color_discrete_sequence=[_CHART_PRIMARY],
            )
            if compact:
                bar_fig.update_layout(
                    margin=dict(l=90, r=20, t=20, b=35),
                    showlegend=False,
                    xaxis_title="기록 건수",
                    yaxis_title="",
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    height=max(220, 40 * len(bar_items) + 90),
                    font=dict(size=10),
                )
                bar_fig.update_xaxes(dtick=1, rangemode="tozero")
                bar_fig.update_yaxes(autorange="reversed")
            else:
                bar_fig.update_layout(
                    margin=dict(l=136, r=28, t=12, b=52),
                    showlegend=False,
                    xaxis_title="기록 건수",
                    yaxis_title="",
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    height=420,
                    font=dict(size=14),
                )
                bar_fig.update_xaxes(
                    dtick=1,
                    rangemode="tozero",
                    title_font=dict(size=14),
                    tickfont=dict(size=13),
                )
                bar_fig.update_yaxes(
                    autorange="reversed",
                    tickfont=dict(size=14),
                )
            bar_fig.update_traces(marker_line_width=0)
            st.plotly_chart(bar_fig, width="stretch")

    with col_radar:
        if compact:
            st.caption("직무 영역별 실무 경험 분포")
        else:
            st.markdown("**직무 영역별 실무 경험 분포**")
        text_all = " ".join(get_reflection_body(str(r.get("bsr", ""))) for r in logs_for_chart)
        axes = ["설계", "제작", "계측", "제어", "안전"]
        keywords = {
            "설계": ["설계", "회로도", "스키매틱", "시뮬레이션"],
            "제작": ["조립", "납땜", "배선", "배관", "장착"],
            "계측": ["측정", "멀티미터", "오실로스코프", "메거", "계측"],
            "제어": ["PLC", "인버터", "시퀀스", "프로그램", "모터제어"],
            "안전": ["안전", "접지", "감전", "보호구", "LOTO", "인터록"],
        }
        scores = [sum(text_all.count(k) for k in keywords[a]) for a in axes]
        total = float(sum(scores))
        if total <= 0:
            values = np.zeros(len(axes), dtype=float)
        else:
            values = np.array(scores, dtype=float) / total * 100.0
        r_vals = list(values) + [float(values[0])]
        theta_vals = axes + [axes[0]]

        _teal = "15, 118, 110"
        fig = go.Figure()
        for ring in [25, 50, 75]:
            fig.add_trace(
                go.Scatterpolar(
                    r=[ring] * (len(axes) + 1),
                    theta=theta_vals,
                    fill="toself",
                    fillcolor=f"rgba({_teal}, 0.04)",
                    line=dict(color=f"rgba({_teal}, 0.2)", width=1, dash="dot"),
                    name="",
                    showlegend=False,
                )
            )
        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=theta_vals,
                fill="toself",
                line=dict(color=_CHART_PRIMARY, width=2),
                fillcolor=f"rgba({_teal}, 0.15)",
                showlegend=False,
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[0, 25, 50, 75, 100],
                    ticksuffix="%",
                    tickfont=dict(size=10 if compact else 13, color="#64748b"),
                    gridcolor=f"rgba({_teal}, 0.12)",
                    linecolor=f"rgba({_teal}, 0.15)",
                ),
                angularaxis=dict(
                    tickfont=dict(size=12 if compact else 15, color=P["text"]),
                    gridcolor=f"rgba({_teal}, 0.12)",
                ),
                bgcolor="rgba(248, 250, 252, 0.6)",
            ),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            margin=dict(l=30, r=30, t=25, b=25) if compact else dict(l=72, r=72, t=36, b=56),
            showlegend=False,
            height=240 if compact else 420,
        )
        st.plotly_chart(fig, width="stretch")
        if compact:
            st.caption(
                "※ 누적된 실습기록을 기준으로 한 경험 분포이며, 공식 NCS 성취도 평가 결과가 아닙니다."
            )
        else:
            st.markdown(
                '<p style="margin:0.45rem 0 0.2rem 0;padding:0 0.1rem 0.25rem 0.1rem;'
                'font-size:0.86rem;line-height:1.55;color:#64748b;word-break:keep-all;">'
                "※ 누적된 실습기록을 기준으로 한 경험 분포이며, "
                "공식 NCS 성취도 평가 결과가 아닙니다.</p>",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)


def _clean_ncs_unit_name(name: str) -> str:
    """NCS 능력단위 표시용 — 영어/숫자 코드(`1902020101_16v3`)·대괄호 코드·'NCS 능력단위:' 접두어 제거."""
    if not name:
        return ""
    cleaned = str(name)
    cleaned = re.sub(r"\bNCS\s*능력단위\s*[:：]?", "", cleaned)
    cleaned = re.sub(r"[\[\(]\s*\d{5,}[_\-][\w\.]+\s*[\]\)]", "", cleaned)
    cleaned = re.sub(r"\b\d{5,}[_\-][\w\.]+\b", "", cleaned)
    cleaned = re.sub(r"[\[\(]\s*[\]\)]", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" |·-—_")
    return cleaned


def _bsr_preview_snippet(bsr_text: str, max_len: int = 30) -> str:
    """BSR 텍스트에서 [배경]/[해결]/[성과]/[체크리스트:…] 태그를 모두 제거한 뒤 앞 N자만 미리보기."""
    if not bsr_text:
        return ""
    text = re.sub(r"\s+", " ", get_reflection_body(str(bsr_text))).strip()
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        return ""
    return (text[:max_len] + "...") if len(text) > max_len else text


def _data_uri_to_bytes(data_uri: str | None) -> bytes | None:
    """data:image/...;base64,... 형태 또는 순수 base64를 바이트로 디코딩."""
    if not data_uri or not str(data_uri).strip():
        return None
    s = str(data_uri).strip()
    try:
        import base64

        payload = s.split(",", 1)[1] if "," in s else s
        raw = base64.b64decode(payload)
        return raw if raw else None
    except Exception:
        return None


def _journal_expander_title_from_row(row: dict[str, Any]) -> str:
    """실습 이력 expander 제목: [날짜 - 주요 성과(한 줄)]."""
    date_s = str(row.get("date") or "—").strip()
    bsr = str(row.get("bsr") or "")
    rec = parse_reflection_record(bsr)
    outcome = (rec.get("now_what") or rec.get("so_what") or rec.get("legacy_reflection") or "").strip()
    if not outcome:
        outcome = (_bsr_preview_snippet(bsr, max_len=56) or "—").strip()
    one_line = re.sub(r"\s+", " ", outcome.replace("\n", " ")).strip()
    if len(one_line) > 56:
        one_line = one_line[:56] + "…"
    return f"[{date_s} - {one_line}]"


def _render_student_practice_log_detail(uid: str, row: dict[str, Any]) -> None:
    """실습 이력 관리 — expander 내부: B/S/R 전문, 증거 사진, NCS 수행준거, AI 톤 변환."""
    date_val = row.get("date") or "—"
    ncs_name = _clean_ncs_unit_name(row.get("ncs_unit", "") or "") or "—"
    bsr_raw = str(row.get("bsr") or "")
    created = (row.get("created_at") or "").strip()
    if created:
        st.caption(f"저장 시각 · {created} · 일지 #{row.get('id', '')}")
    else:
        st.caption(f"일지 #{row.get('id', '')}")

    sections = reflection_display_sections(bsr_raw)
    accents = ("#1d4ed8", "#0f766e", "#b45309")
    chk_m = re.search(r"\[체크리스트:\s*([^\]]+)\]", bsr_raw)
    chk_items = (
        [s.strip() for s in chk_m.group(1).split(";") if s.strip()]
        if chk_m
        else []
    )

    def _detail_para(label: str, body: str, accent: str) -> str:
        body_clean = (body or "").strip()
        if not body_clean:
            return ""
        body_safe = html.escape(body_clean).replace("\n", "<br/>")
        return (
            "<div style=\""
            f"border-left:3px solid {accent};"
            "padding:0.35rem 0 0.35rem 0.85rem;"
            "margin:0.65rem 0;\">"
            f"<div style=\"font-size:0.8rem;font-weight:600;color:{accent};"
            f"letter-spacing:0.02em;margin-bottom:0.2rem;\">{html.escape(label)}</div>"
            f"<div style=\"color:#1f2937;line-height:1.75;font-size:0.95rem;\">"
            f"{body_safe}</div></div>"
        )

    sections_html = "".join(
        _detail_para(title, body, accents[i % 3])
        for i, (title, _hint, body) in enumerate(sections)
    )
    if not sections_html.strip():
        safe_full = html.escape(bsr_raw).replace("\n", "<br/>")
        sections_html = (
            "<div style=\"color:#64748b;font-size:0.88rem;line-height:1.7;\">"
            f"{safe_full}</div>"
        )

    checklist_html = ""
    if chk_items:
        items_html = "".join(
            f"<li style=\"margin:0.15rem 0;color:#334155;\">{html.escape(it)}</li>"
            for it in chk_items
        )
        checklist_html = (
            "<div style=\"margin-top:0.9rem;padding-top:0.75rem;"
            "border-top:1px dashed #cbd5e1;\">"
            "<div style=\"font-size:0.8rem;font-weight:600;color:#4b5563;"
            "margin-bottom:0.3rem;\">NCS 수행준거 점검</div>"
            f"<ul style=\"margin:0;padding-left:1.2rem;font-size:0.9rem;\">{items_html}</ul>"
            "</div>"
        )

    title_html = (
        "<div style=\"display:flex;flex-wrap:wrap;align-items:baseline;gap:0.5rem 0.75rem;"
        "padding-bottom:0.55rem;margin-bottom:0.35rem;border-bottom:1px solid #e2e8f0;\">"
        "<span style=\"font-size:0.85rem;color:#475569;background:#eff6ff;"
        "padding:0.2rem 0.6rem;border-radius:6px;font-weight:500;\">"
        f"{html.escape(str(date_val))}</span>"
        "<span style=\"font-size:1.05rem;font-weight:650;color:#0f172a;\">"
        f"{html.escape(ncs_name)}</span></div>"
    )

    st.markdown(
        "<div style=\"background:linear-gradient(145deg,#ffffff 0%,#f8fafc 100%);"
        "border:1px solid #e2e8f0;border-radius:12px;padding:1.1rem 1.25rem;"
        "margin-bottom:0.85rem;box-shadow:0 2px 8px rgba(15,23,42,0.06);\">"
        f"{title_html}{sections_html}{checklist_html}</div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown("**증거 사진**")
        img_bytes = _data_uri_to_bytes(row.get("image_b64"))
        note = (row.get("image_note") or "").strip()
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
            if note:
                st.caption(note)
        elif note:
            st.info(
                f"{note} — 저장된 대표 사진이 없을 수 있습니다. "
                "여러 장을 올린 경우 DB에는 첫 번째 장이 대표로 보관됩니다.",
                icon=":material/photo_library:",
            )
        else:
            st.caption("등록된 사진이 없습니다.")

    selected_id = int(row.get("id") or 0)
    convert_clicked = st.button(
        "AI 기반 NCS 전문가 톤 변환",
        key=f"ncs_bsr_btn_{uid}_{selected_id}",
        width="stretch",
        icon=":material/auto_awesome:",
        help="이 일지 내용을 NCS 표준 용어로 정제하여 표시합니다.",
    )
    st.caption("버튼을 누르면 위 실습 내용이 NCS 표준 용어 버전으로 변환됩니다.")

    bsr_cache_key = f"ncs_rewrite_bsr_{uid}"
    if bsr_cache_key not in st.session_state:
        st.session_state[bsr_cache_key] = {}
    bsr_cache = st.session_state[bsr_cache_key]
    bsr_cached = bsr_cache.get(bsr_raw)

    if convert_clicked:
        with st.spinner("AI 변환 중..."):
            bsr_ai = _rewrite_to_ncs_terms_with_gemini(bsr_raw)
        if bsr_ai:
            bsr_cache[bsr_raw] = bsr_ai
            bsr_cached = bsr_ai
        else:
            st.warning(
                "API를 사용할 수 없어 사전 기반 치환 결과를 표시합니다.",
                icon=":material/warning:",
            )
            bsr_cached = _rewrite_to_ncs_terms_fallback(bsr_raw)
            bsr_cache[bsr_raw] = bsr_cached

    if bsr_cached and bsr_cached.strip():
        safe_rew = html.escape(bsr_cached).replace("\n", "<br/>")
        st.caption("NCS 표준 용어 버전 (AI 전문가 톤)")
        st.markdown(
            "<div style=\"border-left:4px solid #0f766e;"
            "background:#f0fdfa;padding:0.85rem 1rem;margin-top:0.35rem;"
            "border-radius:8px;color:#334155;line-height:1.75;\">"
            f"{safe_rew}</div>",
            unsafe_allow_html=True,
        )


def _bsr_summary_line(section_text: str, *, max_len: int = 120) -> str:
    """BSR 한 구간을 카드 요약 한 줄로 축약."""
    if not section_text:
        return "—"
    t = re.sub(r"\s+", " ", str(section_text).replace("\n", " ")).strip()
    if not t:
        return "—"
    return (t[:max_len] + "…") if len(t) > max_len else t


def _timeline_saved_at_display(row: dict[str, Any]) -> str:
    """logs.created_at에서 표시용 시각 문자열."""
    ca = str(row.get("created_at") or "").strip()
    if not ca:
        return "—"
    if " " in ca:
        tail = ca.split(" ", 1)[1]
        return tail[:8] if len(tail) >= 5 else ca
    if "T" in ca:
        tail = ca.split("T", 1)[1]
        return tail[:8]
    return ca[:16]


def _render_today_practice_timeline(uid: str) -> None:
    """메인 일지 작성 화면 하단: 오늘(app_today) 저장분 요약 타임라인."""
    today_iso = seoul_today().isoformat()
    rows = [
        r
        for r in list_logs(uid)
        if str(r.get("date") or "").strip() == today_iso
    ]
    rows.sort(
        key=lambda r: (str(r.get("created_at") or ""), int(r.get("id") or 0)),
        reverse=True,
    )

    st.subheader("오늘의 실습 기록 현황", divider="gray")

    if not rows:
        st.info(
            "오늘 첫 실습 기록을 남겨보세요! 성찰 대화를 마친 뒤 **최종 일지 저장하기**로 저장하면 "
            "이곳에 오늘 작성한 일지가 최신순으로 표시됩니다.",
            icon=":material/post_add:",
        )
        return

    st.markdown("##### :material/history: 오늘의 타임라인")
    st.caption(f"{len(rows)}건 저장됨 · 날짜 {today_iso} · 최신 저장이 위에 표시됩니다.")

    for row in rows:
        ncs_short = _clean_ncs_unit_name(row.get("ncs_unit", "") or "") or "—"
        bsr = str(row.get("bsr") or "")
        rec = parse_reflection_record(bsr)
        bg = rec.get("what") or rec.get("legacy_background") or ""
        hg = rec.get("so_what") or rec.get("legacy_solution") or ""
        rw = rec.get("now_what") or rec.get("legacy_reflection") or ""
        img_bytes = _data_uri_to_bytes(row.get("image_b64"))

        with st.container(border=True):
            head_l, head_r = st.columns([2, 1])
            with head_l:
                st.caption(
                    f"저장 시각 · {_timeline_saved_at_display(row)}  ·  "
                    f"일지 #{row.get('id', '—')}  ·  {ncs_short}"
                )
            with head_r:
                if img_bytes:
                    st.image(img_bytes, width=100)
                else:
                    st.caption("증거 사진 없음")

            st.markdown(
                f"**What?** · {_bsr_summary_line(bg)}\n\n"
                f"**So What?** · {_bsr_summary_line(hg)}\n\n"
                f"**Now What?** · {_bsr_summary_line(rw)}"
            )


# ═══════════════════════════════════════════════════════════════════
# 학생 화면 가독성 개선용 작은 UI 헬퍼들 (ui_style.py의 CSS와 짝을 이룸)
# ═══════════════════════════════════════════════════════════════════
def _render_page_header(eyebrow: str, title: str, desc: str) -> None:
    """학생 페이지 상단에 친근한 안내 헤더를 렌더한다 (학생들이 무엇을 하는 화면인지 1초 내 파악).

    ``desc``에는 HTML 조각을 넣는다. 여러 문단·강조는 ``<p>``·``<strong>`` 등으로 구성한다.
    (Streamlit은 블록 HTML 안쪽에서 마크다운을 추가 파싱하지 않는다.)
    """
    st.markdown(
        f"<div class='student-page-header'>"
        f"<div class='student-page-header__eyebrow'>{html.escape(eyebrow)}</div>"
        f"<h2 class='student-page-header__title'>{html.escape(title)}</h2>"
        f"<div class='student-page-header__desc'>{desc}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_stepper(active: int, completed: list[bool]) -> None:
    """Step 1·2·3 진행 인디케이터.

    - active: 1~3 중 현재 작업 중인 단계.
    - completed: 각 단계의 완료 여부 [step1_done, step2_done, step3_done].
    """
    labels = ["메모·증거", "AI 초안", "다듬기·저장"]
    parts: list[str] = []
    for i, label in enumerate(labels):
        step_no = i + 1
        is_done = completed[i] if i < len(completed) else False
        is_active = (step_no == active) and not is_done
        cls = ""
        if is_done:
            cls = " stepper__item--done"
        elif is_active:
            cls = " stepper__item--active"
        symbol = str(step_no)
        parts.append(
            f"<div class='stepper__item{cls}'>"
            f"<span class='stepper__bullet'>{symbol}</span>"
            f"<span class='stepper__label'>{html.escape(label)}</span>"
            "</div>"
        )
        if i < len(labels) - 1:
            conn_done = completed[i] if i < len(completed) else False
            parts.append(
                f"<div class='stepper__connector{' stepper__connector--done' if conn_done else ''}'></div>"
            )
    st.markdown(
        "<div class='stepper'>" + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _render_step_head(num: int, title: str, sub: str = "", status: str = "", status_kind: str = "") -> None:
    """Step 카드 내부 상단에 번호/제목/상태 배지 헤더를 렌더한다."""
    status_html = ""
    if status:
        kind_cls = ""
        if status_kind == "ok":
            kind_cls = " step-card__meta--ok"
        elif status_kind == "warn":
            kind_cls = " step-card__meta--warn"
        status_html = (
            f"<span class='step-card__meta{kind_cls}'>{html.escape(status)}</span>"
        )
    sub_html = (
        f"<p class='step-card__sub' style='font-size:0.85em;color:#666666;margin:0;line-height:1.55;'>"
        f"{html.escape(sub)}</p>"
        if sub
        else ""
    )
    st.markdown(
        f"<div class='step-card__head'>"
        f"<span class='step-card__num'>{num}</span>"
        f"<div class='step-card__head-text'>"
        f"<p class='step-card__title' style='font-size:1.1em;font-weight:bold;margin:0;line-height:1.35;color:#0f172a;'>"
        f"{html.escape(title)}</p>"
        f"{sub_html}"
        f"</div>{status_html}</div>",
        unsafe_allow_html=True,
    )


def _render_dash_chips(items: list[dict]) -> None:
    """상단 대시보드 칩 그룹.

    items: [{"label": "...", "value": "...", "trend": "..."(선택), "trend_kind": "up"|"down"|""(선택)}, ...]
    """
    cards: list[str] = []
    for item in items:
        trend_html = ""
        if item.get("trend"):
            tk = item.get("trend_kind", "")
            tcls = {"up": " dash-chip__trend--up", "down": " dash-chip__trend--down"}.get(tk, "")
            trend_html = (
                f"<span class='dash-chip__trend{tcls}'>{html.escape(item['trend'])}</span>"
            )
        cards.append(
            "<div class='dash-chip'>"
            f"<span class='dash-chip__label'>{html.escape(item['label'])}</span>"
            f"<span class='dash-chip__value'>{html.escape(str(item['value']))}</span>"
            f"{trend_html}"
            "</div>"
        )
    st.markdown(
        "<div class='dash-chips'>" + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


@st.dialog("실습 일지 삭제 확인")
def _dlg_student_delete_one_log(owner_uid: str, row: dict[str, Any]) -> None:
    lid = row.get("id")
    st.markdown(f"**삭제하시겠습니까?**  \n일지 **#{lid}** · 날짜 **{row.get('date', '—')}**")
    st.caption("삭제 후에는 복구할 수 없습니다. 먼저 **백업 CSV**를 받은 뒤 삭제를 진행하세요.")
    st.download_button(
        "삭제 대상 일지 백업 (CSV)",
        data=logs_to_csv_bytes([row], owner_uid=owner_uid),
        file_name=f"backup_log_{owner_uid}_{lid}.csv",
        mime="text/csv",
        key=f"stu_dlg_dl_one_{owner_uid}_{lid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"stu_dlg_can_one_{owner_uid}_{lid}", width="stretch"):
            st.session_state.pop("_stu_dlg_del_one", None)
            st.rerun()
    with c2:
        if st.button("예, 삭제합니다", type="primary", key=f"stu_dlg_go_one_{owner_uid}_{lid}", width="stretch"):
            delete_log(owner_uid, int(lid))
            st.session_state.pop("_stu_dlg_del_one", None)
            st.rerun()


@st.dialog("모든 실습 일지 삭제 확인")
def _dlg_student_clear_all_logs(owner_uid: str, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    st.markdown(f"**정말 모두 삭제하시겠습니까?**  \n총 **{n}건**의 일지가 삭제됩니다.")
    st.caption("복구할 수 없습니다. **전체 백업 CSV**를 저장한 뒤 진행하세요.")
    st.download_button(
        "전체 일지 백업 (CSV)",
        data=logs_to_csv_bytes(rows, owner_uid=owner_uid),
        file_name=f"backup_all_logs_{owner_uid}.csv",
        mime="text/csv",
        key=f"stu_dlg_dl_all_{owner_uid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"stu_dlg_can_all_{owner_uid}", width="stretch"):
            st.session_state.pop("_stu_dlg_clear_all", None)
            st.rerun()
    with c2:
        if st.button("예, 모두 삭제합니다", type="primary", key=f"stu_dlg_go_all_{owner_uid}", width="stretch"):
            clear_logs(owner_uid)
            st.session_state.ncs_progress = seed_progress_if_missing(owner_uid, DEFAULT_NCS_PROGRESS)
            st.session_state.pop("_stu_dlg_clear_all", None)
            st.rerun()


@st.dialog("이력서 데이터 삭제 확인")
def _dlg_student_clear_profile(owner_uid: str) -> None:
    prof = get_student_profile(owner_uid)
    st.markdown("**저장된 이력서·표지 정보를 모두 삭제하시겠습니까?**")
    st.caption("실습 일지는 삭제되지 않습니다. 복구할 수 없습니다. **JSON 백업**을 받은 뒤 진행하세요.")
    st.download_button(
        "현재 이력서 백업 (JSON)",
        data=profile_to_json_bytes(prof),
        file_name=f"backup_profile_{owner_uid}.json",
        mime="application/json",
        key=f"stu_dlg_dl_prof_{owner_uid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"stu_dlg_can_prof_{owner_uid}", width="stretch"):
            st.session_state.pop("_stu_dlg_profile", None)
            st.rerun()
    with c2:
        if st.button("예, 이력서를 삭제합니다", type="primary", key=f"stu_dlg_go_prof_{owner_uid}", width="stretch"):
            clear_student_profile(owner_uid)
            st.session_state.pop("_stu_dlg_profile", None)
            st.rerun()


def _run_student_log_delete_dialogs(uid: str, logs: list[dict[str, Any]]) -> None:
    p1 = st.session_state.get("_stu_dlg_del_one")
    if isinstance(p1, dict) and p1.get("uid") == uid and p1.get("row"):
        _dlg_student_delete_one_log(uid, p1["row"])
    p2 = st.session_state.get("_stu_dlg_clear_all")
    if isinstance(p2, dict) and p2.get("uid") == uid and isinstance(p2.get("rows"), list):
        _dlg_student_clear_all_logs(uid, p2["rows"])


def _reset_scaffolding_chat(uid: str) -> None:
    for suffix in ("sc_step", "sc_q1", "sc_a1", "sc_q2", "sc_a2"):
        st.session_state.pop(f"{suffix}_{uid}", None)


def _scaffold_dialogue_card_html(kind: str, label: str, body: str) -> str:
    """성찰 대화 Q/A 카드. 본문은 자르지 않고 줄바꿈하여 전부 표시한다."""
    styles = {
        "memo": ("#ffffff", "#94a3b8", "#334155", "500"),
        "question": ("#f0fdfa", "#0f766e", "#0f766e", "600"),
        "answer": ("#f8fafc", "#475569", "#1e293b", "500"),
        "feedback": ("#fffbeb", "#b45309", "#78350f", "500"),
    }
    bg, border, label_c, weight = styles.get(kind, styles["memo"])
    body_html = html.escape(body or "").replace("\n", "<br/>")
    if not body_html.strip():
        return ""
    return (
        f"<div class=\"wswnw-card wswnw-card--{html.escape(kind)}\" style=\""
        f"background:{bg};border:1px solid {border};border-left:4px solid {border};"
        "border-radius:10px;padding:0.85rem 1rem;margin:0.65rem 0;"
        "overflow:visible;max-height:none;text-overflow:clip;\">"
        f"<div style=\"font-size:0.82rem;font-weight:700;color:{label_c};"
        "letter-spacing:-0.01em;margin:0 0 0.4rem 0;\">"
        f"{html.escape(label)}</div>"
        f"<div style=\"font-size:0.98rem;line-height:1.7;font-weight:{weight};"
        "color:#0f172a;white-space:pre-wrap;overflow-wrap:anywhere;"
        "word-break:break-word;overflow:visible;text-overflow:clip;\">"
        f"{body_html}</div></div>"
    )


def _render_scaffold_dialogue(meta: dict) -> None:
    """이모지 없이 Q1/A1/Q2/A2 라벨로 성찰 대화를 표시한다."""
    memo = str(meta.get("memo") or "").strip()
    q1 = str(meta.get("q1") or "").strip()
    a1 = str(meta.get("a1") or "").strip()
    q2 = str(meta.get("q2") or "").strip()
    a2 = str(meta.get("a2") or "").strip()
    feedback = str(meta.get("feedback") or "").strip()
    parts: list[str] = []
    if memo:
        parts.append(_scaffold_dialogue_card_html("memo", "실습 메모", memo))
    if q1:
        parts.append(
            _scaffold_dialogue_card_html("question", "Q1. AI 성찰 질문 · So What?", q1)
        )
    if a1:
        parts.append(_scaffold_dialogue_card_html("answer", "A1. 나의 답변", a1))
    if q2:
        parts.append(
            _scaffold_dialogue_card_html("question", "Q2. AI 성찰 질문 · Now What?", q2)
        )
    if a2:
        parts.append(_scaffold_dialogue_card_html("answer", "A2. 나의 답변", a2))
    if feedback:
        parts.append(_scaffold_dialogue_card_html("feedback", "AI 성찰 피드백", feedback))
    if parts:
        st.markdown(
            "<div class='wswnw-dialogue' style='overflow:visible;'>"
            + "".join(parts)
            + "</div>",
            unsafe_allow_html=True,
        )


def _reset_practice_chat(uid: str) -> None:
    """일지 저장/초기화 후 챗봇 작성 상태를 모두 비운다."""
    for key in (
        f"scaffold_step_{uid}",
        f"scaffold_messages_{uid}",
        f"scaffold_meta_{uid}",
        f"chat_bg_{uid}",
        f"chat_hg_{uid}",
        f"chat_sw_{uid}",
        f"chat_what_{uid}",
        f"chat_so_{uid}",
        f"chat_now_{uid}",
        f"draft_memo_{uid}",
        f"evidence_img_{uid}",
        f"img_result_{uid}",
        f"practice_date_{uid}",
    ):
        st.session_state.pop(key, None)


def _render_practice_log_chat_writer(uid: str) -> None:
    """[실습 일지 작성] What(입력) → So What(질문1) → Now What(질문2) → 성찰 초안 저장."""
    use_real_ai = True
    step_key = f"scaffold_step_{uid}"
    msgs_key = f"scaffold_messages_{uid}"
    meta_key = f"scaffold_meta_{uid}"
    date_key = f"practice_date_{uid}"

    if step_key not in st.session_state:
        st.session_state[step_key] = 0
    if msgs_key not in st.session_state:
        st.session_state[msgs_key] = []
    if meta_key not in st.session_state:
        st.session_state[meta_key] = {}
    if date_key not in st.session_state:
        st.session_state[date_key] = seoul_today()
    elif (
        int(st.session_state.get(step_key) or 0) == 0
        and st.session_state.get(date_key) == TEST_PERIOD_END
        and seoul_today() > TEST_PERIOD_END
    ):
        # 테스트 기간 클램프(5/29)가 기본값으로 남은 세션은 실제 오늘로 되돌린다.
        st.session_state[date_key] = seoul_today()

    step = int(st.session_state[step_key])

    _render_page_header(
        eyebrow="AI SCAFFOLDING",
        title="실습 일지 작성",
        desc=(
            '<p style="margin:0 0 0.35rem 0;">1. 날짜·사진·메모로 <strong>What?</strong>(경험)을 남깁니다.</p>'
            '<p style="margin:0 0 0.35rem 0;">2. AI의 <strong>So What?</strong> 질문에 답합니다.</p>'
            '<p style="margin:0 0 0.2rem 0;">3. <strong>Now What?</strong>에 답한 뒤 성찰 일지를 확인하고 저장합니다.</p>'
        ),
    )

    # ─────────────────────────────────────────────────────────
    # 입력 영역 — Step 0에서는 펼치고, 대화가 시작되면 접어 둔다.
    # (file_uploader/date_input 위젯이 계속 마운트되어 값이 유지됨)
    # ─────────────────────────────────────────────────────────
    with st.container(border=True):
        with st.expander("실습 정보 입력 (날짜 · 사진 · 메모)", expanded=(step == 0)):
            practice_date = st.date_input(
                "실습 날짜 선택",
                key=date_key,
                min_value=datetime.date(2024, 1, 1),
                max_value=seoul_today() + datetime.timedelta(days=30),
                help="포트폴리오에 표시되는 실제 실습일입니다. 저장일이 아닙니다.",
            )
            imgs_raw = st.file_uploader(
                "오늘 실습의 핵심 사진 업로드 (여러 장 가능)",
                type=["jpg", "png", "jpeg"],
                accept_multiple_files=True,
                key=f"evidence_img_{uid}",
                help="회로·장비·계측 사진을 올리면 AI가 메모와 함께 분석합니다.",
            )
            imgs = _normalize_img_input(imgs_raw)
            memo = st.text_area(
                "실습 메모 (키워드 또는 단문)",
                height=160,
                placeholder=(
                    "예) 오늘 사용한 장비명, 관찰한 현상, 발생한 문제, "
                    "새롭게 학습한 내용 등을 자유롭게 기재하십시오."
                ),
                key=f"draft_memo_{uid}",
            )
            if imgs:
                if len(imgs) == 1:
                    st.image(imgs[0], width="stretch")
                else:
                    st.caption(f"업로드된 사진 {len(imgs)}장 — 모두 함께 분석됩니다.")

            if step == 0:
                if st.button(
                    "성찰 대화 시작하기 (So What?)",
                    key=f"scaffold_start_{uid}",
                    type="primary",
                    width="stretch",
                    icon=":material/forum:",
                ):
                    if not (memo or "").strip():
                        st.warning(
                            "실습 메모(키워드 또는 단문)를 먼저 입력하시기 바랍니다.",
                            icon=":material/warning:",
                        )
                    else:
                        detected_list: list[dict] = []
                        image_hint = None
                        if imgs:
                            with st.spinner("사진과 메모를 분석하는 중..."):
                                detected_list, image_hint, _ = _maybe_run_analyze_image(
                                    uid, imgs, use_real_ai=use_real_ai, content=memo,
                                )
                        matched_unit = _detect_ncs_unit(memo, image_hint=image_hint)
                        matched_element = _detect_element(matched_unit, memo)
                        detected_clean = [
                            d for d in (detected_list or [])
                            if isinstance(d, dict)
                            and d.get("객체") not in ("사진 없음", "이미지 로드 실패", None, "")
                        ]
                        with st.spinner("경험(What)을 분석하고 So What? 질문을 준비하는 중..."):
                            analysis = analyze_practice_experience(
                                memo, detected_clean, matched_unit
                            )
                            q1 = _scaffold_turn1_question(memo, matched_unit, analysis=analysis)
                        st.session_state[meta_key] = {
                            "memo": memo.strip(),
                            "unit": matched_unit,
                            "element": matched_element,
                            "analysis": analysis,
                            "q1": q1,
                        }
                        st.session_state[msgs_key] = [
                            {"role": "user", "content": f"**실습 메모**\n\n{memo.strip()}"},
                            {"role": "assistant", "content": q1},
                        ]
                        st.session_state[step_key] = 1
                        st.rerun()

    if step == 0:
        st.divider()
        _render_today_practice_timeline(uid)
        return

    meta = st.session_state.get(meta_key, {})
    memo_text = meta.get("memo", "")

    # ─────────────────────────────────────────────────────────
    # 대화 영역 — 채팅 히스토리 + (단계별) 입력/결과
    # ─────────────────────────────────────────────────────────
    with st.container(border=True):
        st.subheader("성찰 대화 (So What? → Now What?)", divider="gray")
        matched_unit_show = str(meta.get("unit") or "").strip()
        if matched_unit_show:
            st.success(
                f"**추천 NCS 능력단위**  \n{format_ncs_unit(matched_unit_show)}",
                icon=":material/track_changes:",
            )
        _render_scaffold_dialogue(meta)

        analysis = meta.get("analysis") if isinstance(meta.get("analysis"), dict) else None

        if step == 1:
            ans1 = st.chat_input("So What? 질문에 답변해 보세요.", key=f"scaffold_in1_{uid}")
            if ans1:
                st.session_state[msgs_key].append({"role": "user", "content": ans1.strip()})
                meta["a1"] = ans1.strip()
                with st.spinner("Now What? 질문을 준비하는 중..."):
                    q2 = _scaffold_turn2_question(
                        memo_text,
                        ans1.strip(),
                        analysis=analysis,
                        turn1_question=meta.get("q1", ""),
                    )
                meta["q2"] = q2
                st.session_state[msgs_key].append({"role": "assistant", "content": q2})
                st.session_state[meta_key] = meta
                st.session_state[step_key] = 2
                st.rerun()

        elif step == 2:
            ans2 = st.chat_input("Now What? 질문에 답변해 보세요.", key=f"scaffold_in2_{uid}")
            if ans2:
                st.session_state[msgs_key].append({"role": "user", "content": ans2.strip()})
                meta["a2"] = ans2.strip()
                st.session_state[meta_key] = meta
                st.session_state[step_key] = 3
                st.rerun()

        elif step >= 3:
            if not meta.get("generated"):
                cached = st.session_state.get(f"img_result_{uid}")
                detected = list(cached[0]) if cached and cached[0] else []
                with st.spinner("What–So What–Now What 성찰 일지를 작성하는 중..."):
                    bsr = _scaffold_build_final_bsr(
                        memo_text,
                        meta.get("a1", ""),
                        meta.get("a2", ""),
                        detected,
                        analysis=analysis,
                    )
                    feedback = _scaffold_final_feedback(
                        memo_text, meta.get("a1", ""), meta.get("a2", "")
                    )
                if bsr.get("what") or bsr.get("so_what") or bsr.get("now_what") or bsr.get("background"):
                    st.session_state[f"chat_bg_{uid}"] = bsr.get("what") or bsr.get("background", "")
                    st.session_state[f"chat_hg_{uid}"] = bsr.get("so_what") or bsr.get("solution", "")
                    st.session_state[f"chat_sw_{uid}"] = bsr.get("now_what") or bsr.get("reflection", "")
                    meta["feedback"] = feedback
                    meta["generated"] = True
                    st.session_state[meta_key] = meta
                    st.rerun()
                else:
                    st.warning(
                        f"{GEMINI_EMPTY_RESPONSE_MESSAGE} "
                        "(API 키·네트워크를 확인하거나 잠시 후 다시 시도해 주십시오.)",
                        icon=":material/warning:",
                    )

    # ── Step 3: 성찰 일지 확인·정제 + 저장 ──
    if step >= 3:
        with st.container(border=True):
            st.subheader("실무 성찰 일지 확인 및 저장", divider="gray")
            st.caption(
                "AI와의 성찰 대화를 바탕으로 작성한 초안입니다. "
                "내용을 확인하고 자신의 표현으로 수정한 뒤 저장하세요."
            )
            bg = st.text_area(
                "What — 실무 경험",
                key=f"chat_bg_{uid}",
                height=150,
            )
            hg = st.text_area(
                "So What — 판단 및 성찰",
                key=f"chat_hg_{uid}",
                height=150,
            )
            sw = st.text_area(
                "Now What — 향후 적용",
                key=f"chat_sw_{uid}",
                height=150,
            )

            col_save, col_reset = st.columns([2, 1])
            with col_save:
                save_clicked = st.button(
                    "💾 최종 일지 저장하기",
                    key=f"scaffold_save_{uid}",
                    type="primary",
                    width="stretch",
                    icon=":material/save:",
                )
            with col_reset:
                if st.button(
                    "처음부터 다시",
                    key=f"scaffold_reset_{uid}",
                    width="stretch",
                    icon=":material/restart_alt:",
                ):
                    _reset_practice_chat(uid)
                    st.rerun()

            if save_clicked:
                bg_v = (bg or "").strip()
                hg_v = (hg or "").strip()
                sw_v = (sw or "").strip() or hg_v
                if not (bg_v or hg_v or sw_v):
                    st.warning(
                        "저장할 내용이 비어 있습니다. 초안을 확인해 주세요.",
                        icon=":material/warning:",
                    )
                else:
                    unit = meta.get("unit", "") or _detect_ncs_unit(memo_text)
                    ana = meta.get("analysis") if isinstance(meta.get("analysis"), dict) else {}
                    save_meta = {
                        "task_type": ana.get("task_type"),
                        "problem_occurred": ana.get("problem_occurred"),
                        "task": ana.get("task"),
                        "equipment": ana.get("equipment"),
                        "ncs_unit": unit or ana.get("ncs_unit"),
                        "raw_input": ana.get("raw_input") or memo_text,
                        "reflection_focus": ana.get("reflection_focus"),
                        "turn1_question": meta.get("q1", ""),
                        "turn1_answer": meta.get("a1", ""),
                        "turn2_question": meta.get("q2", ""),
                        "turn2_answer": meta.get("a2", ""),
                        "evidence": ana.get("evidence"),
                        "image_analysis": ana.get("image_analysis"),
                    }
                    bsr_final = _build_bsr_string(bg_v, hg_v, sw_v, [], meta=save_meta)
                    base_text = f"{bg_v} {hg_v} {sw_v}"
                    length_score = min(5, max(1, (len(base_text) // 30) + 1))
                    all_kw = set(GLOSSARY.keys())
                    for _meta in NCS_DB.values():
                        all_kw.update(_meta.get("keywords", []))
                    for phrases, _, _ in COLLOQUIAL_TO_NCS:
                        all_kw.update(phrases)
                    term_score = min(5, max(1, sum(1 for w in all_kw if w in base_text) + 1))
                    safety_hits = sum(
                        base_text.count(k)
                        for k in ["안전", "접지", "감전", "보호구", "LOTO", "ELB", "차단기"]
                    )
                    safety_score = min(5, max(1, safety_hits + 1))

                    if isinstance(practice_date, datetime.date):
                        date_str = practice_date.isoformat()
                    else:
                        parsed = parse_calendar_date(practice_date)
                        date_str = parsed.isoformat() if parsed else seoul_today().isoformat()
                    ncs_ratio = _compute_ncs_term_ratio(bsr_final)

                    evidence_b64 = None
                    primary_img = imgs[0] if imgs else None
                    if primary_img is not None:
                        try:
                            primary_img.seek(0)
                        except Exception:
                            pass
                        evidence_b64 = _photo_to_base64(primary_img, max_side=720, for_sheet=True)
                    if imgs:
                        image_note_text = (
                            f"사진 {len(imgs)}장 업로드됨" if len(imgs) > 1 else "사진 업로드됨"
                        )
                    else:
                        image_note_text = None

                    try:
                        with st.spinner(
                            "데이터를 안전하게 저장하는 중입니다... 잠시만 기다려주세요 🚀"
                        ):
                            add_log(
                                uid=uid,
                                date=date_str,
                                ncs_unit=unit,
                                bsr=bsr_final,
                                image_note=image_note_text,
                                image_b64=evidence_b64,
                                ncs_term_ratio=ncs_ratio,
                            )
                            if unit:
                                progress_gain = min(
                                    8, max(2, (length_score + term_score + safety_score) // 2)
                                )
                                current = int(
                                    (st.session_state.ncs_progress or {}).get(unit, 0)
                                )
                                new_val = min(current + progress_gain, 100)
                                st.session_state.ncs_progress[unit] = new_val
                                update_progress(uid, unit, new_val)
                            _reset_practice_chat(uid)
                            st.session_state[step_key] = 0
                        st.success("성공적으로 저장되었습니다!")
                        time.sleep(1)
                        st.rerun()
                    except DuplicateLogError:
                        st.warning("⚠️ 이미 동일한 내용의 일지가 방금 저장되었습니다.")
                    except Exception:
                        st.error(
                            "일시적인 네트워크 지연이 발생했습니다. 5초 뒤에 다시 시도해 주세요."
                        )

    st.divider()
    _render_today_practice_timeline(uid)


def _render_scaffolding_chat(uid: str, imgs: list, use_real_ai: bool) -> None:
    """[Step 2] 2-Turn 스캐폴딩 채팅 — 메모 → Q1 → 답변 → Q2 → 답변 → BSR 초안 완성."""
    step_key = f"sc_step_{uid}"
    q1_key, a1_key = f"sc_q1_{uid}", f"sc_a1_{uid}"
    q2_key, a2_key = f"sc_q2_{uid}", f"sc_a2_{uid}"
    step = int(st.session_state.get(step_key, 0))

    memo_raw = (st.session_state.get(f"draft_memo_{uid}") or "").strip()
    draft_meta = st.session_state.get(f"draft_{uid}") or {}
    matched_unit = draft_meta.get("unit", "") if isinstance(draft_meta, dict) else ""

    def _detected_list() -> list[dict]:
        cached = st.session_state.get(f"img_result_{uid}")
        if cached and cached[0]:
            return list(cached[0])
        return []

    # ── 시작 전: 안내 + 시작 버튼 ──
    if step == 0:
        st.caption(
            "Step 1에서 입력한 메모를 바탕으로 AI 튜터가 2번의 질문을 드립니다. "
            "질문에 답하면 답변이 모두 종합되어 성찰 일지 초안이 자동 완성됩니다."
        )
        if st.button(
            "AI 피드백 받기 (2-Turn 스캐폴딩 시작)",
            key=f"sc_start_{uid}",
            type="primary",
            width="stretch",
            icon=":material/forum:",
        ):
            if not memo_raw:
                st.warning(
                    "Step 1의 실습 메모(키워드 또는 단문)를 먼저 입력하시기 바랍니다.",
                    icon=":material/warning:",
                )
            else:
                with st.spinner("AI 튜터가 첫 번째 심화 질문을 준비하는 중..."):
                    st.session_state[q1_key] = _scaffold_turn1_question(memo_raw, matched_unit)
                st.session_state[step_key] = 1
                st.rerun()
        return

    # ── 대화 히스토리 표시 ──
    with st.chat_message("user", avatar="🧑‍🔧"):
        st.markdown(f"**나의 실습 메모**\n\n{memo_raw or '_메모 없음_'}")
    if st.session_state.get(q1_key):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(st.session_state[q1_key])
    if st.session_state.get(a1_key):
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(st.session_state[a1_key])
    if st.session_state.get(q2_key):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(st.session_state[q2_key])
    if st.session_state.get(a2_key):
        with st.chat_message("user", avatar="🧑‍🔧"):
            st.markdown(st.session_state[a2_key])

    # ── Turn 1 답변 입력 ──
    if step == 1:
        ans1 = st.chat_input("AI의 첫 번째 질문에 답해 주세요...", key=f"sc_in1_{uid}")
        if ans1:
            st.session_state[a1_key] = ans1.strip()
            with st.spinner("AI 튜터가 두 번째 꼬리 질문을 준비하는 중..."):
                st.session_state[q2_key] = _scaffold_turn2_question(memo_raw, ans1.strip())
            st.session_state[step_key] = 2
            st.rerun()

    # ── Turn 2 답변 입력 → 최종 BSR 초안 생성 ──
    elif step == 2:
        ans2 = st.chat_input("AI의 두 번째 질문에 답해 주세요...", key=f"sc_in2_{uid}")
        if ans2:
            st.session_state[a2_key] = ans2.strip()
            with st.spinner("메모와 두 답변을 종합하여 성찰 일지 초안을 완성하는 중..."):
                draft_d = _scaffold_build_final_bsr(
                    memo_raw,
                    st.session_state.get(a1_key, ""),
                    ans2.strip(),
                    _detected_list(),
                )
            if draft_d.get("background") or draft_d.get("solution") or draft_d.get("reflection"):
                st.session_state[f"content_{uid}"] = draft_d.get("background", "")
                st.session_state[f"ans_haegyul_{uid}"] = draft_d.get("solution", "")
                st.session_state[f"ans_seungwa_{uid}"] = draft_d.get("reflection", "")
                st.session_state[f"bsr_editor_open_{uid}"] = True
                st.session_state[f"ai_draft_just_generated_{uid}"] = True
                st.session_state[step_key] = 3
                st.rerun()
            else:
                st.warning(
                    f"{GEMINI_EMPTY_RESPONSE_MESSAGE} "
                    "(API 키·네트워크를 확인하거나 잠시 후 다시 시도해 주십시오.)",
                    icon=":material/warning:",
                )

    # ── 완료 ──
    elif step >= 3:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "좋습니다! 답변을 모두 종합하여 아래 **성찰 일지 초안**을 작성해 두었습니다. "
                "자신의 표현으로 다듬은 뒤 저장해 주세요."
            )
        if st.button(
            "대화 다시 시작",
            key=f"sc_restart_{uid}",
            width="stretch",
            icon=":material/restart_alt:",
        ):
            _reset_scaffolding_chat(uid)
            st.rerun()


def show_student(uid: str) -> None:
    NAV_OPTIONS = [
        "내 프로필 관리",
        "실습 일지 작성",
        "실습 이력 관리",
        "AI 성장 진단",
        "NCS 종합 직무 포트폴리오",
    ]

    # --- 사이드바: 학생 프로필 + 세로형 메뉴 ---
    with st.sidebar:
        st.markdown(
            f"""
<div style="padding:0.35rem 0 0.1rem 0;">
  <div style="font-size:0.72rem;color:#64748b;letter-spacing:0.08em;text-transform:uppercase;font-weight:600;">Student Profile</div>
  <div style="font-size:1.15rem;font-weight:700;color:#0f172a;margin-top:0.15rem;line-height:1.25;">{student_label(uid)}</div>
  <div style="font-size:0.82rem;color:#475569;margin-top:0.1rem;">직무 역량 관리 대시보드</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<div style='font-size:0.72rem;color:#64748b;font-weight:700;"
            "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.35rem;'>Menu</div>",
            unsafe_allow_html=True,
        )
        nav = st.radio(
            "메뉴",
            options=NAV_OPTIONS,
            key=f"student_nav_{uid}",
            label_visibility="collapsed",
        )

        # ─── 비밀번호 변경 (사이드바 최하단) ───
        st.divider()
        render_password_change_expander(uid, key_prefix=f"student_{uid}")

    # --- 메인 영역 헤더 ---
    # 모든 학생 화면이 자체적으로 _render_page_header(...)를 그리므로 이 공통 헤더는 더 이상 사용하지 않는다.
    # (혹시 향후 새 메뉴가 추가되었을 때를 대비해 fallback 형태로 남겨 둔다.)
    _custom_header_navs = {
        NAV_OPTIONS[0],
        NAV_OPTIONS[1],
        NAV_OPTIONS[2],
        NAV_OPTIONS[3],
        NAV_OPTIONS[4],
    }
    if nav not in _custom_header_navs:
        st.markdown(
            f"<div style='display:flex;align-items:baseline;gap:0.6rem;"
            f"margin:0 0 0.6rem 0;'>"
            f"<h2 style='margin:0;color:#0f172a;font-weight:800;'>{nav}</h2>"
            f"<span style='color:#64748b;font-size:0.88rem;'>· {student_label(uid)}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    if nav == NAV_OPTIONS[0]:
        _show_profile_management(uid)

    elif nav == NAV_OPTIONS[1]:
        _render_practice_log_chat_writer(uid)
    elif nav == "__legacy_practice_writer_disabled__":
        # 학생 화면은 항상 실제 AI 분석을 사용한다 (시뮬레이션 토글은 v3에서 제거됨).
        use_real_ai = True

        draft_key = f"draft_{uid}"
        if draft_key not in st.session_state:
            st.session_state[draft_key] = None

        # ── (전체 문장 다듬기) 결과 적용 ─────────────────────────
        # form 안의 textarea가 렌더되기 전에, 폴리싱된 결과(있다면)를 위젯 키에 미리 주입한다.
        # 이 단계를 위젯 렌더 이후에 시도하면 Streamlit이 예외를 던지므로 반드시 여기서 처리.
        _pending_polish = st.session_state.pop(f"polish_pending_{uid}", None)
        if _pending_polish:
            st.session_state[f"content_{uid}"] = _pending_polish.get("bg", "")
            st.session_state[f"ans_haegyul_{uid}"] = _pending_polish.get("hg", "")
            st.session_state[f"ans_seungwa_{uid}"] = _pending_polish.get("sw", "")

        # ─── 1. 페이지 상단: 안내 헤더 + Step 진행 인디케이터 ───
        # 안내 문구: 단계는 줄바꿈, 주의는 굵게·이탤릭 (블록 HTML 내부는 MD 미적용 → HTML로 구성)
        _render_page_header(
            eyebrow="STEP-BY-STEP",
            title="실습 일지 작성",
            desc=(
                '<p style="margin:0 0 0.35rem 0;">1. 사진과 메모 업로드</p>'
                '<p style="margin:0 0 0.35rem 0;">2. AI 초안 자동 생성</p>'
                '<p style="margin:0 0 0.75rem 0;">3. 내 표현으로 정제하여 저장</p>'
                '<p style="margin:0.85rem 0 0.4rem 0;line-height:1.6;">'
                "🚨 <strong>주의:</strong> 이력에 남기려면 맨 아래 폼의 "
                "<strong>「최종 확인 및 분석 요청」</strong>을 반드시 눌러야 합니다."
                "</p>"
                '<p style="margin:0;font-size:0.9em;line-height:1.55;color:inherit;opacity:0.92;">'
                "<em>(초안만 받은 상태나 다듬기만 한 상태에서는 저장되지 않습니다.)</em>"
                "</p>"
            ),
        )

        # 현재 진행 상태 계산 (세션 상태 기반)
        _memo_now = (st.session_state.get(f"draft_memo_{uid}") or "").strip()
        _has_evidence = bool(st.session_state.get(f"evidence_img_{uid}"))
        _bg_now = (st.session_state.get(f"content_{uid}") or "").strip()
        _hg_now = (st.session_state.get(f"ans_haegyul_{uid}") or "").strip()
        _sw_now = (st.session_state.get(f"ans_seungwa_{uid}") or "").strip()
        _step1_done = bool(_memo_now or _has_evidence)
        _step2_done = bool(_bg_now or _hg_now or _sw_now)
        _step3_done = bool(_bg_now and _hg_now and _sw_now)
        if not _step1_done:
            _active_step = 1
        elif not _step2_done:
            _active_step = 2
        else:
            _active_step = 3
        _render_stepper(_active_step, [_step1_done, _step2_done, _step3_done])
        st.caption(
            "사진을 올리면 **Gemini 등 외부 API**로 이미지·증거 분석이 한 번에 여러 번 돌아가서, "
            "처음에는 **수십 초** 걸릴 수 있습니다. 같은 사진·메모 조합은 캐시되어 이후에는 더 빨라집니다."
        )

        # 2. 화면을 7:3으로 분할 (왼쪽: 작성 흐름, 오른쪽: NCS 진행 현황)
        col_main, col_side = st.columns([7, 3])
        checked_items: list[str] = []

        # 3. 왼쪽 넓은 영역 (col_main) — Step 1/2/3 일지 작성
        with col_main:
            # ═══════════════════════════════════════════════════════
            # Step 1 — 실습 메모 및 증거 제출
            # ═══════════════════════════════════════════════════════
            _step1_status = ("작성됨", "ok") if _step1_done else ("작성 전", "")
            with st.container(border=True):
                _render_step_head(
                    num=1,
                    title="실습 사진 및 메모 업로드",
                    sub="오늘 실습의 핵심 사진과 키워드 메모를 입력하십시오.",
                    status=_step1_status[0],
                    status_kind=_step1_status[1],
                )

                _practice_date_key = f"practice_date_{uid}"
                if _practice_date_key not in st.session_state:
                    st.session_state[_practice_date_key] = app_today()
                practice_date = st.date_input(
                    "실습 날짜 선택",
                    key=_practice_date_key,
                )

                # ── (1-A) 사진 업로드 — 화면 최상단에 강조 (여러 장 가능) ──
                imgs = st.file_uploader(
                    "오늘 실습의 핵심 사진 업로드 (여러 장 가능)",
                    type=["jpg", "png", "jpeg"],
                    accept_multiple_files=True,
                    key=f"evidence_img_{uid}",
                    help=(
                        "회로·장비·계측 사진을 업로드해 주십시오. "
                        "여러 장(회로, 측정 결과, 결과물 등)을 함께 올리면 "
                        "AI가 모든 사진을 종합하여 분석합니다."
                    ),
                )
                # 정규화: file_uploader는 accept_multiple_files=True일 때 list를 반환하지만,
                # 안전을 위해 None/단일/리스트 모두 처리.
                imgs = _normalize_img_input(imgs)
                # 호환용 별칭 — 기존 코드가 `img`로 부르는 자리들이 의미상 "대표 사진" 역할일 때 사용.
                img = imgs[0] if imgs else None

                if not imgs:
                    for _k in (
                        f"img_analysis_sig_{uid}",
                        f"img_result_{uid}",
                        f"evidence_low_{uid}",
                        f"evidence_low_sig_{uid}",
                    ):
                        st.session_state.pop(_k, None)

                # ── (1-B) 메모 입력 — 사진 아래에 큼직한 textarea ──
                memo = st.text_area(
                    "실습 메모 (키워드 또는 단문)",
                    height=200,
                    placeholder=(
                        "예) 오늘 사용한 장비명, 관찰한 현상, 발생한 문제, "
                        "새롭게 학습한 내용 등을 자유롭게 기재하십시오."
                    ),
                    key=f"draft_memo_{uid}",
                )

                # ── 증거 사진 업로드 시: 사진 분석 결과 요약 ──
                if imgs:
                    bg_ctx_photo = (
                        st.session_state.get(f"content_{uid}") or ""
                    ).strip() or (memo or "").strip()
                    force_sim_photo = st.session_state.get("analyze_force_sim_mode", False)
                    cache_hit_photo = _img_analysis_cache_hit(
                        uid, imgs, use_real_ai=use_real_ai, content=bg_ctx_photo or ""
                    )
                    spinner_label = (
                        f"실습 사진 {len(imgs)}장 분석 및 NCS 단위 매칭 진행 중..."
                        if (use_real_ai and not force_sim_photo)
                        else "시뮬레이션 모드로 표시 중..."
                    )
                    if not cache_hit_photo:
                        with st.spinner(spinner_label):
                            detected_p, suggested_unit_p, safety_advice_p = _maybe_run_analyze_image(
                                uid, imgs, use_real_ai=use_real_ai, content=bg_ctx_photo or "",
                            )
                    else:
                        detected_p, suggested_unit_p, safety_advice_p = _maybe_run_analyze_image(
                            uid, imgs, use_real_ai=use_real_ai, content=bg_ctx_photo or "",
                        )

                    semantic_low_p = False
                    ev_sig_key = f"evidence_low_sig_{uid}"
                    ev_low_key = f"evidence_low_{uid}"
                    if (
                        bg_ctx_photo
                        and bg_ctx_photo.strip()
                        and use_real_ai
                        and not force_sim_photo
                        and _get_google_api_key()
                        and extract_background_section(bg_ctx_photo).strip()
                    ):
                        ev_sig = _evidence_validity_sig(
                            uid, imgs, use_real_ai=use_real_ai, content=bg_ctx_photo or ""
                        )
                        if (
                            st.session_state.get(ev_sig_key) == ev_sig
                            and ev_low_key in st.session_state
                        ):
                            semantic_low_p = bool(st.session_state[ev_low_key])
                        else:
                            try:
                                for _f in imgs:
                                    try:
                                        _f.seek(0)
                                    except Exception:
                                        pass
                                ev = check_evidence_validity(
                                    imgs, bg_ctx_photo, api_key=_get_google_api_key()
                                )
                                semantic_low_p = ev < 40.0
                            except Exception:
                                semantic_low_p = False
                            st.session_state[ev_sig_key] = ev_sig
                            st.session_state[ev_low_key] = semantic_low_p
                    if semantic_low_p:
                        st.warning(
                            "증거 사진과 본문의 연관성이 낮은 것으로 분석되었습니다. "
                            "업로드한 사진과 본문 내용이 일치하는지 확인하시기 바랍니다.",
                            icon=":material/warning:",
                        )
                    elif bg_ctx_photo and bg_ctx_photo.strip():
                        equip_names_p = [d.get("객체", "") for d in detected_p if d.get("객체")]
                        _ev_match = _check_evidence_content_match(equip_names_p, bg_ctx_photo)
                        _domain_mm = _semantic_evidence_mismatch(
                            equip_names_p, bg_ctx_photo, suggested_unit_p
                        )
                        if not _ev_match or _domain_mm:
                            st.warning(
                                "**증거 사진과 본문 내용의 연관성이 낮은 것으로 분석되었습니다.** "
                                "사진에 식별된 장비 및 활동이 본문과 일치하는지 확인하시기 바랍니다.",
                                icon=":material/warning:",
                            )
                    safety_sc = min(
                        5,
                        1 + sum(
                            1
                            for k in ["접지", "보호구", "안전", "LOTO"]
                            if safety_advice_p and k in safety_advice_p
                        ),
                    )
                    # ── 모바일 친화 갤러리: 한 줄에 2장씩 그리드 배치, 폭 자동 맞춤 ──
                    if len(imgs) == 1:
                        st.image(imgs[0], width="stretch")
                    else:
                        st.caption(f"업로드된 사진 {len(imgs)}장 — 모두 함께 분석됩니다.")
                        cols_per_row = 2
                        for row_start in range(0, len(imgs), cols_per_row):
                            cols_grid = st.columns(cols_per_row)
                            for col_idx in range(cols_per_row):
                                file_idx = row_start + col_idx
                                if file_idx >= len(imgs):
                                    break
                                with cols_grid[col_idx]:
                                    st.image(
                                        imgs[file_idx],
                                        width="stretch",
                                        caption=f"#{file_idx + 1} {getattr(imgs[file_idx], 'name', '') or ''}",
                                    )
                    if detected_p:
                        _eq_lines = "\n".join(
                            f"• {d.get('객체', '')} ({d.get('신뢰도', '—')})"
                            for d in detected_p[:6]
                        )
                        st.info(
                            f"**인식된 장비**\n\n{_eq_lines}",
                            icon=":material/photo_camera:",
                        )
                    else:
                        st.caption("인식된 장비: —")
                    st.success(
                        f"**추천 NCS 능력단위**  \n{format_ncs_unit(suggested_unit_p)}",
                        icon=":material/track_changes:",
                    )
                    _safety_snip = (
                        (safety_advice_p[:120] + "…")
                        if safety_advice_p and len(safety_advice_p) > 120
                        else (safety_advice_p or "안전 코멘트 없음")
                    )
                    _safety_body = f"**안전 점검 ({safety_sc}/5)**  \n{_safety_snip}"
                    if safety_sc >= 4:
                        st.success(_safety_body, icon=":material/shield:")
                    elif safety_sc >= 2:
                        st.info(_safety_body, icon=":material/shield:")
                    else:
                        st.warning(_safety_body, icon=":material/shield:")

            # ═══════════════════════════════════════════════════════
            # Step 2 — AI BSR 초안 자동 완성
            # ═══════════════════════════════════════════════════════
            _step2_status = ("AI 초안 생성됨", "ok") if _step2_done else ("AI 초안 대기", "")
            with st.container(border=True):
                _render_step_head(
                    num=2,
                    title="AI 대화형 작성 (2-Turn 스캐폴딩)",
                    sub="AI가 메모를 보고 2번의 심화 질문을 던집니다. 답하면 성찰 일지 초안이 완성됩니다.",
                    status=_step2_status[0],
                    status_kind=_step2_status[1],
                )

                _render_scaffolding_chat(uid, imgs, use_real_ai)

                with st.expander(
                    "빠른 1-step 초안 (질문 없이 바로 생성)",
                    expanded=False,
                    icon=":material/bolt:",
                ):
                    st.caption(
                        "심화 질문 과정 없이 메모와 사진만으로 곧바로 성찰 일지 초안을 생성합니다."
                    )
                    do_ai_draft = st.button(
                        "AI 성찰 초안 자동 생성",
                        key=f"bsr_ai_draft_{uid}",
                        width="stretch",
                        icon=":material/auto_awesome:",
                    )
                if do_ai_draft:
                    memo_raw = (st.session_state.get(f"draft_memo_{uid}") or "").strip()
                    if not memo_raw:
                        st.warning(
                            "Step 1의 실습 메모(키워드 또는 단문)를 먼저 입력하시기 바랍니다.",
                            icon=":material/warning:",
                        )
                    else:
                        detected_list: list[dict] = []
                        cached_ir = st.session_state.get(f"img_result_{uid}")
                        if cached_ir and imgs:
                            detected_list = list(cached_ir[0]) if cached_ir[0] else []
                        elif imgs:
                            with st.spinner("사진 분석을 준비하는 중..."):
                                _maybe_run_analyze_image(
                                    uid,
                                    imgs,
                                    use_real_ai=use_real_ai,
                                    content=memo_raw,
                                )
                            cached_ir = st.session_state.get(f"img_result_{uid}")
                            if cached_ir and cached_ir[0]:
                                detected_list = list(cached_ir[0])
                        try:
                            with st.spinner("성찰 일지 초안을 생성하는 중..."):
                                draft_d = generate_bsr_draft_from_keywords(
                                    memo_raw,
                                    detected_list,
                                    _get_google_api_key() or "",
                                )
                        except Exception as e:
                            st.error(f"상세 에러 내용: {str(e)}")
                            draft_d = {}
                        if draft_d.get("background") or draft_d.get("solution") or draft_d.get("reflection"):
                            st.session_state[f"content_{uid}"] = draft_d.get("background", "")
                            st.session_state[f"ans_haegyul_{uid}"] = draft_d.get("solution", "")
                            st.session_state[f"ans_seungwa_{uid}"] = draft_d.get("reflection", "")
                            st.session_state[f"bsr_editor_open_{uid}"] = True
                            st.session_state[f"ai_draft_just_generated_{uid}"] = True
                            st.toast(
                                "AI 초안이 Step 3에 반영되었습니다.",
                                icon=":material/check_circle:",
                            )
                            st.rerun()
                        else:
                            st.warning(
                                f"{GEMINI_EMPTY_RESPONSE_MESSAGE} "
                                "(API 키·네트워크를 확인하거나 잠시 후 다시 시도해 주십시오.)",
                                icon=":material/warning:",
                            )

            # ═══════════════════════════════════════════════════════
            # Step 3 — 결과 확인 및 다듬기
            # ═══════════════════════════════════════════════════════
            _step3_status = ("저장 준비됨", "ok") if _step3_done else ("작성 중", "")
            with st.container(border=True):
                _render_step_head(
                    num=3,
                    title="내 표현으로 정제 및 저장",
                    sub="AI 튜터의 안내를 참고하여 NCS 기반 실습 일지를 완성합니다.",
                    status=_step3_status[0],
                    status_kind=_step3_status[1],
                )

                # ─────────────────────────────────────────────────
                # A. 작성 가이드 영역 — 가이드 요청 + 출력
                # ─────────────────────────────────────────────────
                st.subheader("AI 작성 가이드", divider="gray")
                st.caption(
                    "실습 메모를 기반으로 AI가 제안하는 작성 가이드를 먼저 확인하시기 바랍니다."
                )
                run_match = st.button(
                    "작성 가이드 받기 / 새로고침",
                    key=f"run_match_{uid}",
                    width="stretch",
                    icon=":material/edit_document:",
                )
                if run_match:
                    cached = st.session_state.get(f"img_result_{uid}")
                    if cached and imgs:
                        image_hint = cached[1]
                    elif imgs:
                        bg_try = (st.session_state.get(f"content_{uid}") or "").strip() or (memo or "").strip()
                        _, image_hint, _ = _maybe_run_analyze_image(
                            uid, imgs, use_real_ai=use_real_ai, content=bg_try,
                        )
                        cached = st.session_state.get(f"img_result_{uid}")
                    else:
                        image_hint = None
                    bg_ctx = (st.session_state.get(f"content_{uid}") or "").strip() or (memo or "").strip()
                    matched_unit = _detect_ncs_unit(bg_ctx, image_hint=image_hint)
                    matched_element = _detect_element(matched_unit, bg_ctx)
                    detected_list = list(cached[0]) if cached else []
                    api_k = _get_google_api_key()
                    recent_logs = list_logs(uid)[:10]
                    r_axes, r_vals = radar_scores_from_logs(recent_logs)
                    with st.spinner("실습 내용에 맞춘 역질문·성찰 예시를 생성하는 중..."):
                        questions = get_ai_scaffolding(
                            bg_ctx,
                            detected_list,
                            matched_unit,
                            prior_radar_axes=r_axes,
                            prior_radar_values=r_vals,
                            api_key=api_k,
                        )
                        reflection_ex = get_reflection_example_sentence(
                            bg_ctx,
                            detected_list,
                            matched_unit,
                            api_key=api_k,
                        )
                    st.session_state[draft_key] = {
                        "content": bg_ctx,
                        "unit": matched_unit,
                        "element": matched_element,
                        "questions": questions,
                        "reflection_example": reflection_ex,
                    }
                    st.rerun()

                # ── A-2. AI 튜터 가이드 출력 (expander, 결과 있으면 자동 펼침) ──
                draft_r = st.session_state.get(draft_key)
                with st.expander(
                    "AI 튜터 가이드 (역질문 및 성찰 예시)",
                    expanded=bool(draft_r),
                    icon=":material/lightbulb:",
                ):
                    if draft_r:
                        bg_for_ctx_head = (
                            (st.session_state.get(f"content_{uid}") or "").strip()
                            or (memo or "").strip()
                            or (draft_r.get("content") or "")
                        )
                        st.markdown(
                            f"<span class='ncs-tag'>매칭된 NCS 단위: {format_ncs_unit(draft_r['unit'])} &gt; {draft_r['element']}</span>",
                            unsafe_allow_html=True,
                        )
                        draft_ncs = _convert_to_ncs_terms(bg_for_ctx_head)
                        if draft_ncs:
                            ncs_summary = ", ".join(f"{n}" for (_, n, __) in draft_ncs)
                            st.caption(f"배경 요약(NCS 용어): {ncs_summary}")

                        qs_list = draft_r.get("questions") or []
                        if not qs_list and draft_r.get("question"):
                            qs_list = [str(draft_r["question"])]
                        if qs_list:
                            q_items = "".join(
                                f"<li class='meta-cognition-qitem'>{html.escape(q or '')}</li>"
                                for q in qs_list[:3]
                            )
                            st.markdown(
                                "<div class='meta-cognition-coach'>"
                                "<p class='meta-cognition-title'>메타인지를 깨우는 질문</p>"
                                f"<ol class='meta-cognition-qlist'>{q_items}</ol>"
                                "</div>",
                                unsafe_allow_html=True,
                            )

                        ref_ex = (draft_r.get("reflection_example") or "").strip()
                        if not ref_ex:
                            _img_c = st.session_state.get(f"img_result_{uid}")
                            _det = list(_img_c[0]) if _img_c and _img_c[0] else []
                            ref_ex = get_reflection_example_sentence(
                                bg_for_ctx_head,
                                _det,
                                draft_r.get("unit") or "",
                                api_key=None,
                            )
                        if ref_ex:
                            safe_ref = html.escape(ref_ex)
                            st.markdown(
                                "<p class='reflection-example-heading'>성찰 문장 예시</p>"
                                f"<div class='reflection-example-box'>{safe_ref}</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption(
                            "아직 생성된 가이드가 없습니다. 상단의 [작성 가이드 받기 / 새로고침] 버튼을 실행하면 "
                            "맞춤 역질문 3개와 성찰 문장 예시가 이 영역에 표시됩니다."
                        )

                # ── B. 생성된 BSR 초안 미리보기 (Step 2 결과가 있을 때만) ──
                bg_state = (st.session_state.get(f"content_{uid}") or "").strip()
                hg_state = (st.session_state.get(f"ans_haegyul_{uid}") or "").strip()
                sw_state = (st.session_state.get(f"ans_seungwa_{uid}") or "").strip()
                just_generated = st.session_state.pop(f"ai_draft_just_generated_{uid}", False)
                if bg_state or hg_state or sw_state:
                    if just_generated:
                        st.success(
                            "AI 초안이 생성되었습니다. 아래 입력창에서 자신의 표현으로 정제하여 작성하시기 바랍니다.",
                            icon=":material/check_circle:",
                        )
                    with st.expander(
                        "생성된 성찰 초안 미리보기",
                        expanded=False,
                        icon=":material/list_alt:",
                    ):
                        if bg_state:
                            st.info(f"**What — 실무 경험**\n\n{bg_state}")
                        else:
                            st.caption("What —")
                        if hg_state:
                            st.info(f"**So What — 판단 및 성찰**\n\n{hg_state}")
                        else:
                            st.caption("So What —")
                        if sw_state:
                            st.success(f"**Now What — 향후 적용**\n\n{sw_state}")
                        else:
                            st.caption("Now What —")

                # ─────────────────────────────────────────────────
                # C. 일지 최종 작성 폼 — 입력창 3개 + 하단 2열 버튼
                # ─────────────────────────────────────────────────
                st.subheader("일지 최종 작성", divider="gray")
                # 폴리싱 결과를 알리기 위한 토스트 메시지(있으면 한 번 보여주고 비움)
                _polish_toast = st.session_state.pop(f"polish_toast_{uid}", None)
                if _polish_toast:
                    st.success(_polish_toast, icon=":material/auto_awesome:")

                with st.form(key=f"log_form_{uid}", clear_on_submit=False):
                    st.info(
                        "**일지 내용 입력 안내**  \n"
                        "AI 가이드를 참고하여 자신의 표현으로 작성하시기 바랍니다. "
                        "[전체 문장 다듬기] 버튼은 AI가 NCS 수행준거 양식으로 정제해 주며, "
                        "[최종 확인 및 분석 요청] 버튼을 통해 일지가 저장됩니다.",
                        icon=":material/edit_note:",
                    )
                    content = st.text_area(
                        "What — 실무 경험",
                        height=200,
                        placeholder=(
                            "예) 오늘은 PLC 시퀀스 실습을 수행하였습니다. "
                            "조립한 회로, 사용한 장비, 실습 목적 등을 기재하시기 바랍니다."
                        ),
                        key=f"content_{uid}",
                    )
                    ans = st.text_area(
                        "So What — 판단 및 성찰",
                        height=200,
                        placeholder=(
                            "예) LED가 점등되지 않아 회로도를 재확인하고 극성을 점검하였습니다. "
                            "발생한 문제와 해결 절차를 단계별로 기재하시기 바랍니다."
                        ),
                        key=f"ans_haegyul_{uid}",
                    )
                    seungwa = st.text_area(
                        "Now What — 향후 적용",
                        height=200,
                        placeholder=(
                            "예) 회로 측정 전 전원 및 접지 상태를 먼저 확인해야 함을 학습하였습니다. "
                            "학습한 내용이나 다음 실습에 적용할 사항을 기재하시기 바랍니다."
                        ),
                        key=f"ans_seungwa_{uid}",
                    )

                    # NCS 수행준거 자가 점검표 — 폼 안의 체크박스는 '저장' 버튼을 눌러야 커밋됨
                    draft = st.session_state.get(draft_key)
                    cl_items = (
                        CHECKLIST.get((draft["unit"], draft["element"]), []) if draft else []
                    )
                    if cl_items:
                        with st.expander(
                            "NCS 수행준거 자가 점검표 (선택)",
                            expanded=False,
                            icon=":material/checklist:",
                        ):
                            st.caption("수행한 항목을 선택하면 저장 시 체크리스트가 성찰 일지에 포함되어 기록됩니다.")
                            for idx, item in enumerate(cl_items):
                                if st.checkbox(
                                    item,
                                    key=f"{uid}_cl_{draft['unit']}_{draft['element']}_{idx}",
                                ):
                                    checked_items.append(item)

                    bg_for_context = (content or "").strip() or (memo or "").strip()

                    # ── 폼 하단 2열 버튼: [전체 문장 다듬기] | [최종 확인 및 분석 요청] ──
                    col_polish, col_final = st.columns([1, 1])
                    with col_polish:
                        polish_all_clicked = st.form_submit_button(
                            "전체 문장 다듬기 (AI 제안)",
                            width="stretch",
                            icon=":material/auto_awesome:",
                            help="입력한 What·So What·Now What을 NCS 수행준거 양식의 정제된 문장으로 동시에 변환합니다.",
                        )
                    with col_final:
                        submitted = st.form_submit_button(
                            "최종 확인 및 분석 요청",
                            type="primary",
                            width="stretch",
                            icon=":material/check_circle:",
                        )

            # ─────────────────────────────────────────────────
            # 폼 결과 처리 — 폴리싱 / 최종 저장
            # ─────────────────────────────────────────────────
            if polish_all_clicked:
                _bg_now = (st.session_state.get(f"content_{uid}") or "").strip()
                _hg_now = (st.session_state.get(f"ans_haegyul_{uid}") or "").strip()
                _sw_now = (st.session_state.get(f"ans_seungwa_{uid}") or "").strip()
                if not (_bg_now or _hg_now or _sw_now):
                    st.warning(
                        "다듬을 내용이 입력되지 않았습니다. What·So What·Now What 항목 중 "
                        "하나 이상을 작성하시기 바랍니다.",
                        icon=":material/warning:",
                    )
                else:
                    _draft_meta = st.session_state.get(draft_key) or {}
                    _ncs_unit = _draft_meta.get("unit", "")
                    _ncs_elem = _draft_meta.get("element", "")
                    _combined = _build_bsr_string(_bg_now, _hg_now, _sw_now, [])
                    with st.spinner("AI가 NCS 수행준거 양식으로 문장을 정제하는 중..."):
                        _polished = _polish_bsr_with_gemini(
                            _combined, ncs_unit=_ncs_unit, ncs_element=_ncs_elem,
                        )
                    if _polished:
                        _new_bg = (
                            extract_bsr_section(_polished, "What")
                            or extract_bsr_section(_polished, "배경")
                            or _bg_now
                        )
                        _new_hg = (
                            extract_bsr_section(_polished, "So What")
                            or extract_bsr_section(_polished, "해결")
                            or _hg_now
                        )
                        _new_sw = (
                            extract_bsr_section(_polished, "Now What")
                            or extract_bsr_section(_polished, "성과")
                            or _sw_now
                        )
                        # 위젯 키를 같은 run 안에서 직접 수정하면 예외 → 다음 run 초입에 반영한다.
                        st.session_state[f"polish_pending_{uid}"] = {
                            "bg": _new_bg,
                            "hg": _new_hg,
                            "sw": _new_sw,
                        }
                        st.session_state[f"polish_toast_{uid}"] = (
                            "AI가 문장을 정제하였습니다. 입력창의 내용을 검토한 뒤 "
                            "[최종 확인 및 분석 요청] 버튼을 통해 저장하시기 바랍니다."
                        )
                        st.rerun()
                    else:
                        st.warning(
                            "AI 정제에 실패하였습니다. API 키 설정을 확인하거나 잠시 후 다시 시도해 주십시오.",
                            icon=":material/warning:",
                        )

            if submitted:
                draft_save = st.session_state.get(draft_key)
                if not draft_save:
                    st.warning(
                        "[작성 가이드 받기 / 새로고침]을 먼저 실행하시기 바랍니다. "
                        "AI가 NCS 능력단위를 매칭한 후 저장이 가능합니다.",
                        icon=":material/warning:",
                    )
                else:
                    haegyul = (ans or "").strip()
                    seungwa_val = (seungwa or "").strip() or haegyul
                    bg_save = (content or "").strip() or (draft_save.get("content", "") or "")
                    bsr_final = _build_bsr_string(
                        bg_save,
                        haegyul,
                        seungwa_val,
                        checked_items,
                    )
                    base_text = (
                        bg_save
                        + " "
                        + (ans or "")
                        + " "
                        + (seungwa or "")
                    )
                    length_score = min(5, max(1, (len(base_text) // 30) + 1))
                    all_kw = set(GLOSSARY.keys())
                    for meta in NCS_DB.values():
                        all_kw.update(meta.get("keywords", []))
                    for phrases, _, _ in COLLOQUIAL_TO_NCS:
                        all_kw.update(phrases)
                    term_hits = sum(1 for w in all_kw if w in base_text)
                    term_score = min(5, max(1, term_hits + 1))
                    safety_hits = sum(
                        base_text.count(k)
                        for k in ["안전", "접지", "감전", "보호구", "LOTO", "ELB", "차단기"]
                    )
                    safety_score = min(5, max(1, safety_hits + 1))

                    if isinstance(practice_date, datetime.date):
                        selected_log_date = practice_date.isoformat()
                    else:
                        selected_log_date = str(practice_date or app_today())[:10]

                    log = {
                        "date": selected_log_date,
                        "bsr": bsr_final,
                        "ncs": draft_save["unit"],
                    }
                    ncs_ratio = _compute_ncs_term_ratio(bsr_final)

                    # 증거 사진(첫 장)을 base64로 인코딩해 DB에 저장.
                    # (현재 image_b64 스키마는 1장만 보관하므로 대표 사진을 첫 번째로 사용.
                    #  여러 장이 올라온 경우 image_note에 총 장수를 기록한다.)
                    evidence_b64: str | None = None
                    primary_img = imgs[0] if imgs else None
                    if primary_img is not None:
                        try:
                            primary_img.seek(0)
                        except Exception:
                            pass
                        evidence_b64 = _photo_to_base64(primary_img, max_side=720, for_sheet=True)

                    if imgs:
                        image_note_text = (
                            f"사진 {len(imgs)}장 업로드됨" if len(imgs) > 1 else "사진 업로드됨"
                        )
                    else:
                        image_note_text = None

                    try:
                        with st.spinner(
                            "데이터를 안전하게 저장하는 중입니다... 잠시만 기다려주세요 🚀"
                        ):
                            add_log(
                                uid=uid,
                                date=log["date"],
                                ncs_unit=log["ncs"],
                                bsr=log["bsr"],
                                image_note=image_note_text,
                                image_b64=evidence_b64,
                                ncs_term_ratio=ncs_ratio,
                            )
                            progress_gain = min(
                                8, max(2, (length_score + term_score + safety_score) // 2)
                            )
                            current = int(
                                (st.session_state.ncs_progress or {}).get(draft_save["unit"], 0)
                            )
                            new_val = min(current + progress_gain, 100)
                            st.session_state.ncs_progress[draft_save["unit"]] = new_val
                            update_progress(uid, draft_save["unit"], new_val)
                            st.session_state[draft_key] = None
                            st.session_state[_practice_date_key] = app_today()
                            _reset_scaffolding_chat(uid)
                        st.success("성공적으로 저장되었습니다!")
                        time.sleep(1)
                        st.rerun()
                    except DuplicateLogError:
                        st.warning("⚠️ 이미 동일한 내용의 일지가 방금 저장되었습니다.")
                    except Exception:
                        st.error(
                            "일시적인 네트워크 지연이 발생했습니다. 5초 뒤에 다시 시도해 주세요."
                        )

            # ─────────────────────────────────────────────────
            # 오늘의 실습 기록 현황 — Step 3 저장 폼 직하단 피드백
            # ─────────────────────────────────────────────────
            _render_today_practice_timeline(uid)

        # 레거시 작성 화면은 비활성화되어 이 분기는 실행되지 않는다.
        with col_side:
            st.empty()

    elif nav == NAV_OPTIONS[2]:
        # ─── 페이지 상단: 안내 헤더 ───
        _render_page_header(
            eyebrow="MY PRACTICE LOGS",
            title="실습 이력 관리",
            desc=(
                "지금까지 작성한 실습 일지를 일괄 조회하고, NCS 능력단위별 실무 경험 축적 현황을 확인할 수 있습니다. "
                "<strong>CSV 다운로드</strong>로 기록을 내려받을 수 있습니다."
            ),
        )

        logs = list_logs(uid)
        if not logs:
            # 작성 화면에만 있는 미저장 초안이 있으면 별도 안내
            _draft_open = bool(st.session_state.get(f"draft_{uid}"))
            _has_bsr_text = any(
                [
                    (st.session_state.get(f"content_{uid}") or "").strip(),
                    (st.session_state.get(f"ans_haegyul_{uid}") or "").strip(),
                    (st.session_state.get(f"ans_seungwa_{uid}") or "").strip(),
                ]
            )
            if _draft_open or _has_bsr_text:
                st.warning(
                    "**아직 이력에 저장되지 않은 작성 내용이 있습니다.**  \n"
                    "사이드바 **[실습 일지 작성]**으로 돌아가서 "
                    "**「최종 일지 저장하기」** 버튼을 눌러 저장을 완료하십시오. "
                    "그 버튼을 눌러 **저장됨** 메시지가 나와야 [실습 이력 관리]에 표시됩니다.",
                    icon=":material/save:",
                )
            # ─── Empty State ───
            st.info(
                "**저장된 실습 기록이 없습니다.**  \n"
                "일지는 **[실습 일지 작성]**에서 성찰 대화를 마친 뒤 **「최종 일지 저장하기」**를 눌렀을 때만 "
                "이 화면에 쌓입니다. AI 가이드·초안만 받은 것은 아직 저장이 아닙니다.  \n"
                "또한 **[작성 가이드 받기 / 새로고침]**을 실행해 NCS 단위가 잡힌 뒤에만 저장할 수 있습니다.",
                icon=":material/info:",
            )
            st.caption("Step 1·2·3을 진행한 뒤, 반드시 최종 저장 버튼까지 눌러 주십시오.")
        else:
            # ═══════════════════════════════════════════════════════
            # 1) 상단 대시보드 칩 — 한눈에 보이는 핵심 지표
            # ═══════════════════════════════════════════════════════
            _first_log = logs[0]
            _last_date = (_first_log.get("date") or "—") if logs else "—"
            _ca = str(_first_log.get("created_at") or "").strip()
            if _ca and logs:
                if " " in _ca:
                    _last_date = f"{_first_log.get('date') or '—'} · {_ca.split(' ', 1)[1][:8]}"
                elif "T" in _ca:
                    _last_date = f"{_first_log.get('date') or '—'} · {_ca.split('T', 1)[1][:8]}"
            _ncs_counts: dict[str, int] = {}
            for _r in logs:
                _u = _clean_ncs_unit_name(_r.get("ncs_unit", "") or "") or "기타"
                _ncs_counts[_u] = _ncs_counts.get(_u, 0) + 1
            _top_ncs = max(_ncs_counts.items(), key=lambda x: x[1])[0] if _ncs_counts else "—"
            _avg_len = int(
                sum(len(_bsr_preview_snippet(r.get("bsr", "") or "", max_len=10000) or "") for r in logs)
                / max(len(logs), 1)
            )
            _render_dash_chips([
                {"label": "누적 일지", "value": f"{len(logs)} 건"},
                {"label": "마지막 작성", "value": _last_date},
                {"label": "주력 NCS 단위", "value": _top_ncs[:14] + ("…" if len(_top_ncs) > 14 else "")},
                {"label": "평균 글자수", "value": f"{_avg_len} 자"},
            ])

            # ═══════════════════════════════════════════════════════
            # 2) 상단 액션 스트립 — 안내(st.info) + 다운로드 + 보기 모드 토글
            #    모바일에서는 CSS가 자동으로 세로 스택해 줍니다.
            # ═══════════════════════════════════════════════════════
            st.info(
                "**일지별 상세 보기**  \n"
                "아래 **[날짜 - 주요 성과]** 제목의 목록을 펼치면 해당 일지의 **성찰 일지 전문**, **증거 사진**, "
                "**NCS 전문가 톤 변환**을 한곳에서 확인할 수 있습니다. "
                "목록은 **저장 시각 기준 최신순**입니다.",
                icon=":material/folder_open:",
            )
            _act_b, _act_c = st.columns([1, 1])
            with _act_b:
                csv_bytes = (
                    "id,date,ncs_unit,bsr\n"
                    + "\n".join(
                        '"{}","{}","{}","{}"'.format(
                            row.get("id", ""),
                            row.get("date", ""),
                            row.get("ncs_unit", ""),
                            get_reflection_body(row.get("bsr", "") or "").replace('"', '""'),
                        )
                        for row in logs
                    )
                ).encode("utf-8-sig")
                st.download_button(
                    "CSV 다운로드",
                    data=csv_bytes,
                    file_name=f"{uid}_logs.csv",
                    mime="text/csv",
                    key=f"csv_dl_{uid}",
                    width="stretch",
                    icon=":material/download:",
                )
            with _act_c:
                _toggle_key = f"history_show_table_{uid}"
                if _toggle_key not in st.session_state:
                    st.session_state[_toggle_key] = False
                _is_table_now = st.session_state[_toggle_key]
                if st.button(
                    "표 보기" if not _is_table_now else "일지 expander 보기",
                    key=f"history_view_toggle_{uid}",
                    width="stretch",
                    icon=":material/table_chart:" if not _is_table_now else ":material/unfold_more:",
                    help="일지별 expander 보기와 표 형식 요약 간 전환합니다.",
                ):
                    st.session_state[_toggle_key] = not st.session_state[_toggle_key]
                    st.rerun()

            # ─── 일지 삭제 (상단에서 바로 사용 — 연습 기록 정리) ───
            with st.container(border=True):
                st.markdown(
                    "**일지 삭제** · 잘못 올리거나 연습용으로 쓴 기록은 여기서 지울 수 있습니다. "
                    "삭제한 데이터는 **복구되지 않습니다.**"
                )
                manage_options: list[tuple[Any, str]] = []
                for row in logs:
                    mdate = row.get("date", "")
                    mncs = _clean_ncs_unit_name(row.get("ncs_unit", "") or "") or "—"
                    msnippet = _bsr_preview_snippet(row.get("bsr", "") or "", max_len=40) or "—"
                    manage_options.append(
                        (row.get("id"), f"#{row.get('id')} [{mdate}] {mncs} — {msnippet}")
                    )
                mcol_a, mcol_b = st.columns([4, 1], gap="small")
                with mcol_a:
                    manage_selected = st.selectbox(
                        "삭제할 일지 선택",
                        options=manage_options,
                        format_func=lambda x: x[1],
                        key=f"manage_del_sel_{uid}",
                        label_visibility="collapsed",
                    )
                with mcol_b:
                    st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                    if st.button(
                        "삭제 확인…",
                        key=f"manage_del_btn_{uid}",
                        width="stretch",
                        type="secondary",
                        icon=":material/help_outline:",
                        help="한 번 더 확인한 뒤 삭제합니다. 백업 CSV를 받을 수 있습니다.",
                    ):
                        if manage_selected:
                            lid = int(manage_selected[0])
                            row_del = next((r for r in logs if r.get("id") == lid), None)
                            if row_del:
                                st.session_state["_stu_dlg_del_one"] = {
                                    "uid": uid,
                                    "row": copy_log_row(row_del),
                                }
                                st.rerun()

                st.divider()
                st.caption(
                    "아래는 **이 계정의 실습 일지를 전부** 지웁니다."
                )
                confirm_all = st.checkbox("모든 일지 삭제에 동의합니다", key=f"confirm_clear_{uid}")
                if st.button(
                    "삭제 확인 화면 열기 (전체)",
                    disabled=not confirm_all,
                    key=f"clear_all_{uid}",
                    width="stretch",
                    icon=":material/warning:",
                    help="백업 CSV와 최종 확인 창이 열립니다.",
                ):
                    st.session_state["_stu_dlg_clear_all"] = {
                        "uid": uid,
                        "rows": [copy_log_row(r) for r in logs],
                    }
                    st.rerun()

            _run_student_log_delete_dialogs(uid, logs)

            # ═══════════════════════════════════════════════════════
            # 3) 표 보기 ↔ 일지별 expander (저장 시각 최신순)
            # ═══════════════════════════════════════════════════════
            if st.session_state.get(f"history_show_table_{uid}", False):
                display_logs = []
                for r in logs:
                    bsr_clean = _bsr_preview_snippet(r.get("bsr", "") or "", max_len=60) or "—"
                    display_logs.append(
                        {
                            "ID": r.get("id"),
                            "작성시각": (r.get("created_at") or "") or "—",
                            "날짜": r.get("date", "") or "—",
                            "NCS 능력단위": _clean_ncs_unit_name(r.get("ncs_unit", "") or "") or "—",
                            "요약": bsr_clean,
                        }
                    )
                st.dataframe(
                    display_logs,
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.markdown("##### 일지별 상세")
                st.caption(
                    "작성·저장 시각이 **가장 최근인 일지**가 위에 옵니다. "
                    "같은 날 여러 건을 작성해도 모두 따로 저장되어 아래에서 각각 확인할 수 있습니다."
                )
                for r in logs:
                    lid = int(r.get("id") or 0)
                    exp_title = _journal_expander_title_from_row(r)
                    with st.expander(exp_title, expanded=False, key=f"stu_pr_hist_{uid}_{lid}"):
                        _render_student_practice_log_detail(uid, r)

            st.divider()
            _render_ncs_progress_section(uid, compact=False)

    elif nav == NAV_OPTIONS[3]:
        # ─── 페이지 상단: 안내 헤더 ───
        _render_page_header(
            eyebrow="AI GROWTH REPORT",
            title="AI 성장 진단 보고서",
            desc=(
                "작성된 실습 일지를 AI가 종합적으로 분석합니다. "
                "핵심 지표를 확인한 후 세부 항목별 상세 분석을 검토하시기 바랍니다."
            ),
        )

        logs = list_logs(uid)
        if not logs:
            # ─── Empty State ───
            st.info(
                "**분석 가능한 실습 일지가 존재하지 않습니다.**  \n"
                "사이드바의 [실습 일지 작성] 메뉴에서 첫 일지를 기록한 후 본 보고서를 이용하시기 바랍니다.",
                icon=":material/info:",
            )
            st.caption(
                "일지가 한 건 이상 저장되면 AI 성장 총평, 메타인지 코멘트, "
                "베스트 실습·레벨업 미션이 본 화면에 표시됩니다."
            )
        else:
            growth_key = f"ai_growth_{uid}"
            meta_key = f"ai_meta_coach_{uid}"

            # ═══════════════════════════════════════════════════════
            # 1) 핵심 지표 대시보드 (항상 최상단에 보이도록)
            # ═══════════════════════════════════════════════════════
            length_list, term_list, safety_list, reflect_list = [], [], [], []
            for row in logs:
                s = _log_competency_scores(str(row.get("bsr", "")))
                length_list.append(s.get("구체성", 0))
                term_list.append(s.get("전문용어", 0))
                safety_list.append(s.get("안전", 0))
                reflect_list.append(s.get("성찰", 0))
            avg_len = round(sum(length_list) / max(len(length_list), 1), 1)
            avg_term = round(sum(term_list) / max(len(term_list), 1), 1)
            avg_safe = round(sum(safety_list) / max(len(safety_list), 1), 1)
            avg_reflect = round(sum(reflect_list) / max(len(reflect_list), 1), 1)

            # 최근 3개와 최초 3개의 평균 비교 (변화 추세)
            def _avg_dim(log_list: list[dict]) -> dict[str, float]:
                if not log_list:
                    return {"구체성": 0.0, "전문용어": 0.0, "안전": 0.0, "성찰": 0.0}
                acc = {"구체성": 0.0, "전문용어": 0.0, "안전": 0.0, "성찰": 0.0}
                for r in log_list:
                    s = _log_competency_scores(str(r.get("bsr", "")))
                    for d in acc:
                        acc[d] += s.get(d, 0)
                return {d: acc[d] / max(len(log_list), 1) for d in acc}
            recent_avg = _avg_dim(logs[:3])
            first_avg = _avg_dim(list(reversed(logs))[:3])
            total_recent = sum(recent_avg.values())
            total_first = sum(first_avg.values())
            if total_recent > total_first + 0.1:
                _trend_label = f"↑ 최근 3개 평균 {round(total_recent,1)}점 (이전 {round(total_first,1)}점)"
                _trend_kind = "up"
            elif total_recent < total_first - 0.1:
                _trend_label = f"↓ 최근 3개 평균 {round(total_recent,1)}점 (이전 {round(total_first,1)}점)"
                _trend_kind = "down"
            else:
                _trend_label = f"≈ 최근 3개 평균 {round(total_recent,1)}점 (이전 {round(total_first,1)}점)"
                _trend_kind = ""

            _render_dash_chips([
                {"label": "누적 일지", "value": f"{len(logs)} 건", "trend": _trend_label, "trend_kind": _trend_kind},
                {"label": "구체성 평균", "value": f"{avg_len} / 5"},
                {"label": "전문용어 평균", "value": f"{avg_term} / 5"},
                {"label": "성찰 평균", "value": f"{avg_reflect} / 5"},
            ])

            # ═══════════════════════════════════════════════════════
            # 2) AI 진단 액션 영역 (성장 총평 / 메타인지 코멘트)
            # ═══════════════════════════════════════════════════════
            report_existing = st.session_state.get(growth_key)
            meta_existing = st.session_state.get(meta_key)
            growth_sections = (
                _parse_ai_growth_report_sections(report_existing)
                if report_existing
                else {}
            )
            growth_summary = (
                _growth_summary_for_card(report_existing) if report_existing else ""
            )

            col_growth, col_meta = st.columns([1, 1], gap="medium")

            with col_growth:
                with st.container(border=True):
                    _render_step_head(
                        num=1,
                        title="AI 맞춤형 성장 총평",
                        sub="전체 실습 일지를 종합하여 강점과 보완점을 분석합니다.",
                        status="생성됨" if report_existing else "대기",
                        status_kind="ok" if report_existing else "",
                    )
                    if report_existing:
                        # 긴 AI 보고서를 expander로 감싸 한 화면을 차지하지 않도록 함
                        with st.expander(
                            "성장 총평 상세 보기",
                            expanded=False,
                            icon=":material/description:",
                        ):
                            st.success(growth_summary)
                        if st.button(
                            "다시 분석하기",
                            key=f"growth_refresh_{uid}",
                            width="stretch",
                            icon=":material/refresh:",
                        ):
                            with st.spinner("AI가 실습 이력을 다시 분석하는 중..."):
                                report = _get_ai_growth_report(logs)
                            if report:
                                st.session_state[growth_key] = report
                                st.rerun()
                            else:
                                st.warning(
                                    "API를 사용할 수 없습니다. API 키 설정을 확인하시기 바랍니다.",
                                    icon=":material/warning:",
                                )
                    else:
                        st.info(
                            "Gemini가 실습 일지 전체를 분석하여 성장 보고서를 작성합니다. "
                            "분석에는 통상 10초 이내가 소요됩니다.",
                            icon=":material/smart_toy:",
                        )
                        if st.button(
                            "AI 성장 총평 생성",
                            key=f"growth_refresh_{uid}",
                            type="primary",
                            width="stretch",
                            icon=":material/auto_awesome:",
                        ):
                            with st.spinner("AI가 실습 이력을 분석하는 중..."):
                                report = _get_ai_growth_report(logs)
                            if report:
                                st.session_state[growth_key] = report
                                st.rerun()
                            else:
                                st.warning(
                                    "API를 사용할 수 없습니다. API 키 설정을 확인하시기 바랍니다.",
                                    icon=":material/warning:",
                                )

            with col_meta:
                with st.container(border=True):
                    _render_step_head(
                        num=2,
                        title="메타인지 성장 코멘트",
                        sub="최근 3개 일지의 성찰 깊이와 전문 용어 변화를 비교 분석합니다.",
                        status="생성됨" if meta_existing else "대기",
                        status_kind="ok" if meta_existing else "",
                    )
                    if meta_existing:
                        with st.expander(
                            "메타인지 코멘트 상세 보기",
                            expanded=False,
                            icon=":material/description:",
                        ):
                            st.success(meta_existing)
                        if st.button(
                            "다시 분석하기",
                            key=f"meta_coach_btn_{uid}",
                            width="stretch",
                            icon=":material/refresh:",
                        ):
                            with st.spinner("최근 일지를 분석하여 메타인지 코멘트를 작성하는 중..."):
                                mc = _get_ai_meta_coach_comment(logs)
                            if mc:
                                st.session_state[meta_key] = mc
                                st.rerun()
                            else:
                                st.warning(
                                    "API를 사용할 수 없습니다. API 키 설정을 확인하시기 바랍니다.",
                                    icon=":material/warning:",
                                )
                    else:
                        st.info(
                            "AI가 최근 3개 일지를 비교하여 향후 개선 방향을 제시합니다.",
                            icon=":material/psychology:",
                        )
                        if st.button(
                            "메타인지 코멘트 생성",
                            key=f"meta_coach_btn_{uid}",
                            type="primary",
                            width="stretch",
                            icon=":material/auto_awesome:",
                        ):
                            with st.spinner("최근 일지를 분석하여 메타인지 코멘트를 작성하는 중..."):
                                mc = _get_ai_meta_coach_comment(logs)
                            if mc:
                                st.session_state[meta_key] = mc
                                st.rerun()
                            else:
                                st.warning(
                                    "API를 사용할 수 없습니다. API 키 설정을 확인하시기 바랍니다.",
                                    icon=":material/warning:",
                                )

            # ═══════════════════════════════════════════════════════
            # (보조) 성찰 변화 가시화 — 레이다 차트 (번호 카드 외)
            # ═══════════════════════════════════════════════════════
            with st.expander(
                "성찰 변화 추이 (레이다 차트)",
                expanded=False,
                icon=":material/show_chart:",
            ):
                st.caption("최초 3개 일지 vs 최근 3개 일지의 역량 성장 곡선을 레이다 차트로 비교합니다.")
                if len(logs) >= 2:
                    reversed_logs = list(reversed(logs))
                    first3 = reversed_logs[:3]
                    recent3 = logs[:3]

                    def _avg_scores(log_list: list[dict]) -> list[float]:
                        if not log_list:
                            return [0.0] * 4
                        by_dim: dict[str, list[float]] = {"구체성": [], "전문용어": [], "안전": [], "성찰": []}
                        for row in log_list:
                            s = _log_competency_scores(row.get("bsr") or "")
                            for d in by_dim:
                                by_dim[d].append(s.get(d, 0))
                        return [sum(by_dim[d]) / max(len(by_dim[d]), 1) for d in by_dim]

                    first_vals = _avg_scores(first3)
                    recent_vals = _avg_scores(recent3)
                    dims = ["구체성", "전문용어", "안전", "성찰"]
                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=first_vals + [first_vals[0]], theta=dims + [dims[0]], fill="toself",
                        name="최초 3개 일지", line={"color": _CHART_ACCENT}
                    ))
                    fig_radar.add_trace(go.Scatterpolar(
                        r=recent_vals + [recent_vals[0]], theta=dims + [dims[0]], fill="toself",
                        name="최근 3개 일지", line={"color": _CHART_PRIMARY}
                    ))
                    fig_radar.update_layout(
                        polar={"radialaxis": {"visible": True, "range": [0, 5]}},
                        showlegend=True, height=320, margin=dict(l=40, r=40, t=20, b=20),
                        paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                    )
                    st.plotly_chart(fig_radar, width="stretch")
                    if sum(recent_vals) > sum(first_vals):
                        st.success(
                            "최근 일지에서 성찰 및 전문 용어 점수가 향상되었습니다.",
                            icon=":material/celebration:",
                        )
                else:
                    st.info(
                        "비교 차트는 일지가 2건 이상 저장된 경우 표시됩니다. 현재 1건이 저장되어 있으며, "
                        "한 건 더 작성하시면 성장 추이가 표시됩니다.",
                        icon=":material/info:",
                    )

            # ═══════════════════════════════════════════════════════
            # 3) 베스트 실습 순간 + 4) 레벨업 미션 (AI 성장 총평 [3][4] 파싱)
            # ═══════════════════════════════════════════════════════
            col_best, col_mission = st.columns([1, 1], gap="medium")

            with col_best:
                with st.container(border=True):
                    _render_step_head(
                        num=3,
                        title="🏆 나의 베스트 실습 순간",
                        sub="What–So What–Now What 연결과 판단이 돋보인 최고의 실습을 AI가 선정합니다.",
                        status="생성됨" if growth_sections.get("best_moment") else "대기",
                        status_kind="ok" if growth_sections.get("best_moment") else "",
                    )
                    best_moment = growth_sections.get("best_moment")
                    if best_moment:
                        st.success(best_moment)
                    elif report_existing:
                        st.caption(
                            "총평은 생성되었으나 베스트 실습 구간을 구분하지 못했습니다. "
                            "[다시 분석하기]를 눌러 주세요."
                        )
                    else:
                        st.caption(
                            "상단 [AI 성장 총평 생성]을 실행하면 이곳에 표시됩니다."
                        )

            with col_mission:
                with st.container(border=True):
                    _render_step_head(
                        num=4,
                        title="🚀 레벨업 미션",
                        sub="다음 실습 현장에서 바로 시도할 수 있는 행동 목표를 제안합니다.",
                        status="생성됨" if growth_sections.get("mission") else "대기",
                        status_kind="ok" if growth_sections.get("mission") else "",
                    )
                    level_up_mission = growth_sections.get("mission")
                    if level_up_mission:
                        st.info(level_up_mission)
                    elif report_existing:
                        st.caption(
                            "총평은 생성되었으나 레벨업 미션 구간을 구분하지 못했습니다. "
                            "[다시 분석하기]를 눌러 주세요."
                        )
                    else:
                        st.caption(
                            "상단 [AI 성장 총평 생성]을 실행하면 이곳에 표시됩니다."
                        )

    elif nav == NAV_OPTIONS[4]:
        _show_digital_portfolio(uid)


def _photo_to_base64(
    uploaded_file,
    max_side: int = 720,
    *,
    for_sheet: bool = False,
) -> str | None:
    """
    업로드된 사진을 최대 변 max_side 이하로 줄인 뒤 JPEG base64 data URI 반환. 실패 시 None.

    for_sheet=True이면 Google Sheets 셀 한도(~49k자)를 넘기지 않도록 품질·크기를 추가로 낮춘다.
    """
    try:
        import base64

        img_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else bytes(uploaded_file)
        if not img_bytes:
            return None
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        w, h = img.size
        side_cap = min(max_side, 512) if for_sheet else max_side
        if max(w, h) > side_cap:
            ratio = side_cap / float(max(w, h))
            img = img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))))
        buf = io.BytesIO()
        q = 78 if for_sheet else 88
        img.save(buf, format="JPEG", quality=int(q), optimize=True)
        raw = buf.getvalue()
        # data:image/jpeg;base64, 접두사 23자 + base64 본문이 셀 한도 이내가 되도록(여유 800자)
        max_total = 48200
        while len(raw) > 34000 and q >= 22:
            q -= 10
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=int(q), optimize=True)
            raw = buf.getvalue()
        enc = base64.b64encode(raw).decode()
        out = f"data:image/jpeg;base64,{enc}"
        if for_sheet and len(out) > max_total:
            factor = 0.82
            while len(out) > max_total and min(img.width, img.height) > 96:
                img = img.resize(
                    (max(1, int(img.width * factor)), max(1, int(img.height * factor))),
                    Image.Resampling.LANCZOS,
                )
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=max(q, 20), optimize=True)
                raw = buf.getvalue()
                enc = base64.b64encode(raw).decode()
                out = f"data:image/jpeg;base64,{enc}"
        return out
    except Exception:
        return None


def _show_profile_management(uid: str) -> None:
    """학생 프로필 관리 화면 — 이력서·취업 포트폴리오의 1페이지 데이터를 직접 편집."""
    profile = get_student_profile(uid)

    # ─── 페이지 상단: 안내 헤더 ───
    _render_page_header(
        eyebrow="RESUME PROFILE",
        title="이력서 정보 관리",
        desc=(
            "본 페이지에 입력된 정보는 <strong>NCS 종합 직무 포트폴리오</strong>의 표지 및 이력서 페이지에 자동 반영됩니다. "
            "사진 → 기본 정보 → 학력 → 경력 → 자격증 → 수상 → 기술 스택 순으로 작성하시기 바랍니다."
        ),
    )

    # ─── 프로필 완성도 칩 (한눈에 보이는 진행 현황) ───
    _photo_done = bool((profile.get("photo_b64") or "").strip())
    _basic_done = bool((profile.get("full_name") or "").strip())
    _edu_count = len([r for r in (profile.get("educations") or []) if any(str(v).strip() for v in r.values())])
    _car_count = len([r for r in (profile.get("careers") or []) if any(str(v).strip() for v in r.values())])
    _cert_count = len([r for r in (profile.get("certificates") or []) if any(str(v).strip() for v in r.values())])
    _award_count = len([r for r in (profile.get("awards") or []) if any(str(v).strip() for v in r.values())])
    _tech_count = len(profile.get("tech_stack") or [])
    # 6개 섹션(사진/기본/학력/경력/자격/기술스택) 중 채워진 비율
    _filled = sum([
        1 if _photo_done else 0,
        1 if _basic_done else 0,
        1 if _edu_count > 0 else 0,
        1 if _car_count > 0 else 0,
        1 if _cert_count > 0 else 0,
        1 if _tech_count > 0 else 0,
    ])
    _percent = int(round(_filled / 6 * 100))
    _render_dash_chips([
        {"label": "프로필 완성도", "value": f"{_percent} %",
         "trend": f"6개 섹션 중 {_filled}개 작성", "trend_kind": "up" if _percent >= 70 else ""},
        {"label": "사진", "value": "등록됨" if _photo_done else "필요"},
        {"label": "자격증", "value": f"{_cert_count} 개"},
        {"label": "기술 스택", "value": f"{_tech_count} 개"},
    ])

    with st.container(border=True):
        st.markdown("**이력서 데이터 전체 삭제** (연습 입력 지우기)")
        st.caption(
            "표지·학력·경력 등 **이 페이지에 저장된 내용만** 삭제합니다. "
            "**실습 일지**는 [실습 이력 관리] 메뉴에서 따로 삭제하세요."
        )
        pr_ok = st.checkbox("이력서 전체 삭제에 동의합니다", key=f"confirm_profile_reset_{uid}")
        if st.button(
            "삭제 확인 화면 열기 (이력서)",
            disabled=not pr_ok,
            key=f"profile_reset_all_{uid}",
            width="stretch",
            icon=":material/warning:",
        ):
            st.session_state["_stu_dlg_profile"] = uid
            st.rerun()

    if st.session_state.get("_stu_dlg_profile") == uid:
        _dlg_student_clear_profile(uid)

    # ─── 1. 사진 + 기본 인적사항 ───
    with st.container(border=True):
        _render_step_head(
            num=1,
            title="프로필 사진 · 기본 인적사항",
            sub="이력서 1페이지 상단에 들어갈 사진과 핵심 인적사항입니다.",
            status="등록됨" if _basic_done else "작성 전",
            status_kind="ok" if _basic_done else "",
        )
        col_photo, col_info = st.columns([1, 2])
        with col_photo:
            current_photo = profile.get("photo_b64") or ""
            if current_photo:
                st.markdown(
                    f"<div style='width:100%;aspect-ratio:3/4;background:#f1f5f9;"
                    f"border-radius:12px;overflow:hidden;border:1px solid #e2e8f0;'>"
                    f"<img src='{current_photo}' alt='프로필 사진' "
                    "style='width:100%;height:100%;object-fit:cover;' /></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='width:100%;aspect-ratio:3/4;background:#f1f5f9;"
                    "border-radius:12px;display:flex;align-items:center;justify-content:center;"
                    "color:#94a3b8;font-size:0.85rem;border:1px dashed #cbd5e1;'>"
                    "사진 미등록</div>",
                    unsafe_allow_html=True,
                )
            new_photo = st.file_uploader(
                "사진 교체",
                type=["jpg", "jpeg", "png"],
                key=f"profile_photo_{uid}",
            )
            # 모바일에서는 사진 저장·삭제 버튼이 자연스럽게 세로 스택 (CSS 자동 처리)
            cph_a, cph_b = st.columns(2)
            with cph_a:
                if st.button(
                    "사진 저장",
                    key=f"profile_photo_save_{uid}",
                    width="stretch",
                    icon=":material/photo_camera:",
                ):
                    if new_photo is None:
                        st.warning(
                            "먼저 사진을 선택하시기 바랍니다.",
                            icon=":material/warning:",
                        )
                    else:
                        b64 = _photo_to_base64(new_photo)
                        if b64:
                            profile["photo_b64"] = b64
                            save_student_profile(uid, profile)
                            st.success(
                                "사진이 저장되었습니다.",
                                icon=":material/check_circle:",
                            )
                            st.rerun()
                        else:
                            st.error(
                                "사진 처리에 실패하였습니다. JPG 또는 PNG 형식의 파일로 재시도하시기 바랍니다.",
                                icon=":material/error:",
                            )
            with cph_b:
                if current_photo and st.button(
                    "사진 삭제",
                    key=f"profile_photo_clear_{uid}",
                    width="stretch",
                    icon=":material/delete:",
                ):
                    profile["photo_b64"] = ""
                    save_student_profile(uid, profile)
                    st.rerun()

        with col_info:
            colA, colB = st.columns(2)
            with colA:
                full_name = st.text_input(
                    "이름",
                    value=profile.get("full_name", ""),
                    key=f"profile_name_{uid}",
                    placeholder="홍길동",
                )
                birth_date = st.text_input(
                    "생년월일 (YYYY-MM-DD)",
                    value=profile.get("birth_date", ""),
                    key=f"profile_birth_{uid}",
                    placeholder="2007-03-15",
                )
                phone = st.text_input(
                    "연락처",
                    value=profile.get("phone", ""),
                    key=f"profile_phone_{uid}",
                    placeholder="010-1234-5678",
                )
            with colB:
                email = st.text_input(
                    "이메일",
                    value=profile.get("email", ""),
                    key=f"profile_email_{uid}",
                    placeholder="student@example.com",
                )
                motto = st.text_area(
                    "좌우명 / 한 줄 소개",
                    value=profile.get("motto", ""),
                    key=f"profile_motto_{uid}",
                    height=110,
                    placeholder="현장에서 통하는 전기·전자 엔지니어를 향해 매일 1가지씩 배워 갑니다.",
                )

    # ─── 2. 학력 ───
    with st.container(border=True):
        _render_step_head(
            num=2,
            title="학력 사항",
            sub="기간(예: 2023.03 ~ 재학)·학교명·학과·재학상태를 입력하세요.",
            status=f"{_edu_count}건 등록" if _edu_count > 0 else "작성 전",
            status_kind="ok" if _edu_count > 0 else "",
        )
        edu_template = pd.DataFrame(profile.get("educations") or [])
        if edu_template.empty:
            edu_template = pd.DataFrame(
                [{"period": "", "school": "", "dept": "", "status": ""}]
            )
        else:
            for col in ["period", "school", "dept", "status"]:
                if col not in edu_template.columns:
                    edu_template[col] = ""
            edu_template = edu_template[["period", "school", "dept", "status"]]
        edu_df = st.data_editor(
            edu_template,
            num_rows="dynamic",
            width="stretch",
            key=f"profile_edu_{uid}",
            column_config={
                "period": st.column_config.TextColumn("기간", width="medium"),
                "school": st.column_config.TextColumn("학교명", width="medium"),
                "dept": st.column_config.TextColumn("학과/전공", width="medium"),
                "status": st.column_config.TextColumn("재학 상태", width="small"),
            },
        )

    # ─── 3. 경력 / 산학 도제 ───
    with st.container(border=True):
        _render_step_head(
            num=3,
            title="경력 · 산학일체형 도제 활동",
            sub="기간·기업(또는 기관)·역할·담당 업무 요약을 입력하세요.",
            status=f"{_car_count}건 등록" if _car_count > 0 else "작성 전",
            status_kind="ok" if _car_count > 0 else "",
        )
        car_template = pd.DataFrame(profile.get("careers") or [])
        if car_template.empty:
            car_template = pd.DataFrame(
                [{"period": "", "company": "", "role": "", "description": ""}]
            )
        else:
            for col in ["period", "company", "role", "description"]:
                if col not in car_template.columns:
                    car_template[col] = ""
            car_template = car_template[["period", "company", "role", "description"]]
        car_df = st.data_editor(
            car_template,
            num_rows="dynamic",
            width="stretch",
            key=f"profile_car_{uid}",
            column_config={
                "period": st.column_config.TextColumn("기간", width="small"),
                "company": st.column_config.TextColumn("회사 / 기관", width="medium"),
                "role": st.column_config.TextColumn("직무", width="small"),
                "description": st.column_config.TextColumn("담당 업무 요약", width="large"),
            },
        )

    # ─── 4. 자격증 ───
    with st.container(border=True):
        _render_step_head(
            num=4,
            title="자격증",
            sub="취득일(YYYY-MM)·자격증명·발급기관을 입력하세요.",
            status=f"{_cert_count}건 등록" if _cert_count > 0 else "작성 전",
            status_kind="ok" if _cert_count > 0 else "",
        )
        cert_template = pd.DataFrame(profile.get("certificates") or [])
        if cert_template.empty:
            cert_template = pd.DataFrame([{"date": "", "name": "", "issuer": ""}])
        else:
            for col in ["date", "name", "issuer"]:
                if col not in cert_template.columns:
                    cert_template[col] = ""
            cert_template = cert_template[["date", "name", "issuer"]]
        cert_df = st.data_editor(
            cert_template,
            num_rows="dynamic",
            width="stretch",
            key=f"profile_cert_{uid}",
            column_config={
                "date": st.column_config.TextColumn("취득일", width="small"),
                "name": st.column_config.TextColumn("자격증명", width="medium"),
                "issuer": st.column_config.TextColumn("발급 기관", width="medium"),
            },
        )

    # ─── 5. 수상 실적 ───
    with st.container(border=True):
        _render_step_head(
            num=5,
            title="수상 · 활동 실적",
            sub="일자(YYYY-MM)·수상명/활동명·주관 기관을 입력하세요.",
            status=f"{_award_count}건 등록" if _award_count > 0 else "작성 전",
            status_kind="ok" if _award_count > 0 else "",
        )
        award_template = pd.DataFrame(profile.get("awards") or [])
        if award_template.empty:
            award_template = pd.DataFrame([{"date": "", "title": "", "organizer": ""}])
        else:
            for col in ["date", "title", "organizer"]:
                if col not in award_template.columns:
                    award_template[col] = ""
            award_template = award_template[["date", "title", "organizer"]]
        award_df = st.data_editor(
            award_template,
            num_rows="dynamic",
            width="stretch",
            key=f"profile_award_{uid}",
            column_config={
                "date": st.column_config.TextColumn("일자", width="small"),
                "title": st.column_config.TextColumn("수상명 / 활동명", width="medium"),
                "organizer": st.column_config.TextColumn("주관 기관", width="medium"),
            },
        )

    # ─── 6. 기술 스택 (0~100 점수) ───
    with st.container(border=True):
        _render_step_head(
            num=6,
            title="기술 스택 (Tech Stack)",
            sub="전기·전자과 핵심 스킬을 0~100점으로 자기 평가하세요. 0점은 자동 제외, 1점 이상만 포트폴리오 막대그래프에 표시됩니다.",
            status=f"{_tech_count}개 평가됨" if _tech_count > 0 else "작성 전",
            status_kind="ok" if _tech_count > 0 else "",
        )

        # 기존 점수 → 빠른 조회용 dict
        existing_scores: dict[str, int] = {}
        for r in profile.get("tech_stack") or []:
            try:
                existing_scores[str(r.get("skill") or "").strip()] = int(r.get("score") or 0)
            except (TypeError, ValueError):
                pass

        # 전기·전자과 NCS 직무 핵심 스킬 (사전 정의)
        predefined_skills = [
            "납땜",
            "회로 조립",
            "OrCAD / PCB 설계",
            "오실로스코프 측정",
            "멀티미터 / 계측",
            "Arduino",
            "STM32 / 임베디드",
            "PLC 시퀀스 제어",
            "Modbus / RS-485 통신",
            "센서 응용",
            "전기안전 (LOTO)",
        ]

        slider_scores: dict[str, int] = {}
        cols = st.columns(2)
        for i, skill in enumerate(predefined_skills):
            with cols[i % 2]:
                slider_scores[skill] = st.slider(
                    skill,
                    min_value=0,
                    max_value=100,
                    value=int(existing_scores.get(skill, 0)),
                    step=5,
                    key=f"profile_tech_slider_{uid}_{i}",
                )

        # 사용자가 추가 스킬을 입력할 수 있도록 확장 영역 제공
        with st.expander("사용자 정의 스킬 추가 (선택)", expanded=False):
            st.caption(
                "위 목록에 없는 스킬을 자유롭게 추가하세요. 빈 행은 자동 제외됩니다."
            )
            custom_existing = [
                {"skill": s, "score": v}
                for s, v in existing_scores.items()
                if s not in predefined_skills
            ]
            custom_template = pd.DataFrame(custom_existing or [{"skill": "", "score": 0}])
            for col in ("skill", "score"):
                if col not in custom_template.columns:
                    custom_template[col] = "" if col == "skill" else 0
            custom_template = custom_template[["skill", "score"]]
            custom_tech_df = st.data_editor(
                custom_template,
                num_rows="dynamic",
                width="stretch",
                key=f"profile_tech_custom_{uid}",
                column_config={
                    "skill": st.column_config.TextColumn("사용자 스킬", width="medium"),
                    "score": st.column_config.NumberColumn(
                        "점수 (0~100)", min_value=0, max_value=100, step=5, width="small"
                    ),
                },
            )

    # ─── 저장 버튼 (모바일: 전체 폭 / 데스크톱: 자연스럽게 폭만큼) ───
    if st.button(
        "프로필 저장",
        key=f"profile_save_{uid}",
        type="primary",
        width="stretch",
        icon=":material/save:",
    ):
        def _df_to_records(df: pd.DataFrame, required_keys: list[str]) -> list[dict]:
            if df is None or df.empty:
                return []
            rows: list[dict] = []
            for _, r in df.iterrows():
                rec = {k: ("" if pd.isna(r.get(k)) else r.get(k)) for k in required_keys}
                if any(str(v).strip() for v in rec.values()):
                    rows.append({k: str(v).strip() if isinstance(v, str) else v for k, v in rec.items()})
            return rows

        tech_records: list[dict] = []
        # 사전 정의 스킬: 점수가 1점 이상인 것만 저장 (0점은 미사용으로 간주)
        for skill, score in slider_scores.items():
            if int(score or 0) > 0:
                tech_records.append({"skill": skill, "score": int(score)})
        # 사용자 정의 스킬: 점수>0이고 스킬명이 비어있지 않은 것만
        if custom_tech_df is not None and not custom_tech_df.empty:
            seen_names = {r["skill"] for r in tech_records}
            for _, r in custom_tech_df.iterrows():
                skill = "" if pd.isna(r.get("skill")) else str(r.get("skill")).strip()
                raw_score = r.get("score")
                try:
                    score = int(0 if pd.isna(raw_score) else float(raw_score))
                except (TypeError, ValueError):
                    score = 0
                score = max(0, min(100, score))
                if skill and score > 0 and skill not in seen_names:
                    tech_records.append({"skill": skill, "score": score})
                    seen_names.add(skill)

        updated = {
            "full_name": st.session_state.get(f"profile_name_{uid}", ""),
            "birth_date": st.session_state.get(f"profile_birth_{uid}", ""),
            "email": st.session_state.get(f"profile_email_{uid}", ""),
            "phone": st.session_state.get(f"profile_phone_{uid}", ""),
            "motto": st.session_state.get(f"profile_motto_{uid}", ""),
            "photo_b64": profile.get("photo_b64", ""),
            "educations": _df_to_records(edu_df, ["period", "school", "dept", "status"]),
            "careers": _df_to_records(car_df, ["period", "company", "role", "description"]),
            "certificates": _df_to_records(cert_df, ["date", "name", "issuer"]),
            "awards": _df_to_records(award_df, ["date", "title", "organizer"]),
            "tech_stack": tech_records,
        }
        save_student_profile(uid, updated)
        st.success(
            "프로필이 저장되었습니다. [NCS 종합 직무 포트폴리오] 메뉴에서 확인하실 수 있습니다.",
            icon=":material/check_circle:",
        )


def _esc(s: Any) -> str:
    """간단한 HTML 이스케이프 (None/숫자도 안전 변환)."""
    return html.escape("" if s is None else str(s))


def _build_resume_page_html(uid: str, profile: dict, prog: dict, logs: list[dict]) -> str:
    """포트폴리오 1페이지: 비주얼 이력서 (사진+인적사항 + 경력/학력/자격/기술스택)."""
    photo_b64 = profile.get("photo_b64") or ""
    full_name = (profile.get("full_name") or "").strip() or student_label(uid)
    motto = (profile.get("motto") or "").strip()
    birth = (profile.get("birth_date") or "").strip()
    email = (profile.get("email") or "").strip()
    phone = (profile.get("phone") or "").strip()

    # ── 좌측 컬럼: 사진 + 연락처 ──
    if photo_b64:
        photo_html = f"<img class='resume-photo' src='{photo_b64}' alt='profile' />"
    else:
        photo_html = (
            "<div class='resume-photo resume-photo--placeholder'>"
            "<span>PHOTO</span></div>"
        )

    contact_rows = []
    if birth:
        contact_rows.append(("Birth", _esc(birth)))
    if phone:
        contact_rows.append(("Phone", _esc(phone)))
    if email:
        contact_rows.append(("Email", _esc(email)))
    contact_rows.append(("School", "용산철도고등학교 · 산학일체형 도제학교"))
    contact_rows.append(("Track", "전기·전자과 / NCS 기반 직무 포트폴리오"))

    contact_html = "".join(
        f"<li><span class='label'>{lab}</span><span class='value'>{val}</span></li>"
        for lab, val in contact_rows
    )

    motto_html = (
        f"<p class='resume-motto'>“{_esc(motto)}”</p>" if motto else ""
    )

    # ── 우측: 경력 / 학력 / 자격증 / 수상 / 기술스택 ──
    careers = profile.get("careers") or []
    educations = profile.get("educations") or []
    certs = profile.get("certificates") or []
    awards = profile.get("awards") or []
    tech_stack = profile.get("tech_stack") or []

    def _timeline_html(rows: list[dict], primary: str, sub: str, body: str) -> str:
        items = []
        for r in rows:
            p = _esc(r.get(primary, ""))
            s = _esc(r.get(sub, ""))
            b = _esc(r.get(body, ""))
            if not (p or s or b):
                continue
            items.append(
                "<li class='timeline-item'>"
                f"<div class='timeline-period'>{p}</div>"
                "<div class='timeline-body'>"
                f"<div class='timeline-title'>{s}</div>"
                f"<div class='timeline-desc'>{b}</div>"
                "</div></li>"
            )
        if not items:
            return "<p class='resume-empty'>등록된 항목이 없습니다.</p>"
        return f"<ul class='timeline-list'>{''.join(items)}</ul>"

    careers_html = _timeline_html(careers, "period", "company", "description") if careers else (
        "<p class='resume-empty'>등록된 경력이 없습니다.</p>"
    )
    # 경력은 회사명 옆에 직무도 보이도록 별도 구성
    if careers:
        items = []
        for r in careers:
            period = _esc(r.get("period", ""))
            company = _esc(r.get("company", ""))
            role = _esc(r.get("role", ""))
            desc = _esc(r.get("description", ""))
            if not (period or company or role or desc):
                continue
            role_chip = (
                f"<span class='timeline-role-chip'>{role}</span>" if role else ""
            )
            items.append(
                "<li class='timeline-item'>"
                f"<div class='timeline-period'>{period}</div>"
                "<div class='timeline-body'>"
                f"<div class='timeline-title'>{company} {role_chip}</div>"
                f"<div class='timeline-desc'>{desc}</div>"
                "</div></li>"
            )
        careers_html = (
            f"<ul class='timeline-list'>{''.join(items)}</ul>"
            if items
            else "<p class='resume-empty'>등록된 경력이 없습니다.</p>"
        )

    if educations:
        items = []
        for r in educations:
            period = _esc(r.get("period", ""))
            school = _esc(r.get("school", ""))
            dept = _esc(r.get("dept", ""))
            status = _esc(r.get("status", ""))
            if not (period or school or dept or status):
                continue
            status_chip = (
                f"<span class='timeline-role-chip'>{status}</span>" if status else ""
            )
            items.append(
                "<li class='timeline-item'>"
                f"<div class='timeline-period'>{period}</div>"
                "<div class='timeline-body'>"
                f"<div class='timeline-title'>{school} {status_chip}</div>"
                f"<div class='timeline-desc'>{dept}</div>"
                "</div></li>"
            )
        educations_html = (
            f"<ul class='timeline-list'>{''.join(items)}</ul>"
            if items
            else "<p class='resume-empty'>등록된 학력이 없습니다.</p>"
        )
    else:
        educations_html = "<p class='resume-empty'>등록된 학력이 없습니다.</p>"

    if certs:
        cert_rows = "".join(
            f"<tr><td>{_esc(r.get('date',''))}</td><td>{_esc(r.get('name',''))}</td>"
            f"<td>{_esc(r.get('issuer',''))}</td></tr>"
            for r in certs
            if any(str(r.get(k, '')).strip() for k in ('date', 'name', 'issuer'))
        )
        certs_html = (
            "<table class='resume-table'><thead><tr>"
            "<th>취득일</th><th>자격증명</th><th>발급기관</th>"
            f"</tr></thead><tbody>{cert_rows}</tbody></table>"
            if cert_rows
            else "<p class='resume-empty'>등록된 자격증이 없습니다.</p>"
        )
    else:
        certs_html = "<p class='resume-empty'>등록된 자격증이 없습니다.</p>"

    if awards:
        award_rows = "".join(
            f"<tr><td>{_esc(r.get('date',''))}</td><td>{_esc(r.get('title',''))}</td>"
            f"<td>{_esc(r.get('organizer',''))}</td></tr>"
            for r in awards
            if any(str(r.get(k, '')).strip() for k in ('date', 'title', 'organizer'))
        )
        awards_html = (
            "<table class='resume-table'><thead><tr>"
            "<th>일자</th><th>수상명/활동</th><th>주관 기관</th>"
            f"</tr></thead><tbody>{award_rows}</tbody></table>"
            if award_rows
            else "<p class='resume-empty'>등록된 수상 실적이 없습니다.</p>"
        )
    else:
        awards_html = "<p class='resume-empty'>등록된 수상 실적이 없습니다.</p>"

    # ── 기술 스택: 가로 막대 차트 ──
    if tech_stack:
        bars = []
        for r in sorted(tech_stack, key=lambda x: -int(x.get("score") or 0)):
            skill = _esc(r.get("skill", ""))
            try:
                score = int(r.get("score") or 0)
            except (TypeError, ValueError):
                score = 0
            score = max(0, min(100, score))
            bars.append(
                "<div class='skill-row'>"
                f"<div class='skill-name'>{skill}</div>"
                "<div class='skill-bar-track'>"
                f"<div class='skill-bar-fill' style='width:{score}%;'></div>"
                "</div>"
                f"<div class='skill-score'>{score}</div>"
                "</div>"
            )
        tech_html = "".join(bars)
    else:
        tech_html = "<p class='resume-empty'>등록된 기술 스택이 없습니다.</p>"

    # ── NCS 상위 단위 요약 (소형 칩) ──
    top_ncs = sorted(prog.items(), key=lambda x: -x[1])[:6]
    ncs_chips = "".join(
        f"<span class='ncs-chip'>{_esc(format_ncs_unit(u))} <strong>{v}%</strong></span>"
        for u, v in top_ncs
        if v > 0
    )
    if not ncs_chips:
        ncs_chips = (
            "<span class='ncs-chip ncs-chip--muted'>실습 일지 누적 후 NCS 진도가 이곳에 표시됩니다.</span>"
        )

    n_logs = len(logs)
    avg_prog_v = round(sum(prog.values()) / max(len(prog), 1), 1) if prog else 0

    # ── 지도교사 종합의견 (확정본만 HTML/PDF에 포함) ──
    confirmed = get_confirmed_portfolio_comment(uid)
    if confirmed and (confirmed.get("comment_text") or "").strip():
        level = (confirmed.get("reflection_level") or "").strip()
        lvl_html = f"<span class='comment-level'>성찰 수준: {_esc(level)}</span>" if level else ""
        body_html = _esc(confirmed.get("comment_text") or "").replace("\n", "<br/>")
        teacher_comment_html = (
            "<section class='teacher-comment'>"
            f"<h2 class='resume-section-title'>지도교사 종합의견 {lvl_html}</h2>"
            f"<div class='teacher-comment-body'>{body_html}</div>"
            "</section>"
        )
    else:
        teacher_comment_html = ""

    # 페이지 조립
    return f"""
<section class='resume-page'>
  <header class='resume-header'>
    <div class='resume-header-bar'></div>
    <div class='resume-header-inner'>
      <div class='resume-name-block'>
        <p class='resume-eyebrow'>NCS 국가직무능력표준 기반 직무 포트폴리오</p>
        <h1 class='resume-name'>{_esc(full_name)}</h1>
        <p class='resume-subname'>{_esc(student_label(uid))} · 전기·전자과 산학일체형 도제생</p>
        {motto_html}
      </div>
      <div class='resume-quick-metrics'>
        <div class='qm'><span class='qm-num'>{n_logs}</span><span class='qm-lab'>실습 일지</span></div>
        <div class='qm'><span class='qm-num'>{avg_prog_v}%</span><span class='qm-lab'>NCS 평균 진도</span></div>
        <div class='qm'><span class='qm-num'>{len(prog)}</span><span class='qm-lab'>추적 단위</span></div>
      </div>
    </div>
  </header>

  <div class='resume-grid'>
    <aside class='resume-side'>
      {photo_html}
      <h3 class='side-h'>About Me</h3>
      <ul class='resume-contact-list'>{contact_html}</ul>

      <h3 class='side-h'>NCS Top Units</h3>
      <div class='ncs-chip-grid'>{ncs_chips}</div>

      <h3 class='side-h'>Tech Stack</h3>
      <div class='resume-skills'>{tech_html}</div>
    </aside>

    <main class='resume-main'>
      <h2 class='resume-section-title'>Career &amp; Apprenticeship</h2>
      {careers_html}

      <h2 class='resume-section-title'>Education</h2>
      {educations_html}

      <h2 class='resume-section-title'>Certifications</h2>
      {certs_html}

      <h2 class='resume-section-title'>Awards &amp; Activities</h2>
      {awards_html}

      {teacher_comment_html}
    </main>
  </div>
</section>
""".strip()


def _build_project_pages_html(selected_logs: list[dict]) -> str:
    """포트폴리오 2페이지+: 베스트 실습을 프로젝트 보고서 양식으로 출력."""
    if not selected_logs:
        return (
            "<section class='project-page'>"
            "<h2 class='project-section-title'>Best Practice Projects</h2>"
            "<p class='resume-empty'>좌측 화면에서 「베스트 실습」 항목을 선택하면 "
            "이 페이지부터 프로젝트 보고서 양식으로 자동 구성됩니다.</p>"
            "</section>"
        )

    pages: list[str] = []
    pages.append(
        "<section class='project-cover'>"
        "<p class='resume-eyebrow'>PORTFOLIO · PART 02</p>"
        "<h2 class='project-cover-title'>Best Practice Projects</h2>"
        "<p class='project-cover-sub'>NCS 직무 능력단위에 따라 실습 수행 경험과 핵심 판단, 향후 적용 계획을 정리한 프로젝트 보고서 모음입니다.</p>"
        "</section>"
    )

    for idx, row in enumerate(selected_logs, start=1):
        bsr_raw = str(row.get("bsr") or "")
        bsr_html = render_portfolio_entry_html(bsr_raw)
        ncs_display = format_ncs_unit(row.get("ncs_unit", ""))
        date_str = log_display_date(row)
        entry = generate_portfolio_entry(bsr_raw)
        chip_html = ""
        for chip in entry.get("chips") or []:
            chip_html += f"<span class='project-meta-chip'>{_esc(chip)}</span>"
        evidence_chips = ""
        if row.get("image_b64") or row.get("image_note"):
            evidence_chips = (
                "<span class='project-meta-chip project-meta-chip--evidence'>증거 사진 첨부</span>"
            )

        # ── 증거 사진 (있으면 본문 좌측에 고화질로 출력, 사진 없으면 1열 풀와이드) ──
        photo_b64 = row.get("image_b64") or ""
        photo_block = ""
        section_modifier = ""
        if photo_b64:
            photo_block = (
                "<figure class='project-photo'>"
                f"<img src='{photo_b64}' alt='실습 증거 사진' />"
                "<figcaption>실습 증거 사진</figcaption>"
                "</figure>"
            )
            section_modifier = " project-page--has-photo"

        pages.append(
            f"<section class='project-page{section_modifier}'>"
            "<header class='project-header'>"
            f"<div class='project-num'>Project · {idx:02d}</div>"
            f"<h2 class='project-title'>{_esc(ncs_display)}</h2>"
            f"<div class='project-meta'>"
            f"<span class='project-meta-chip'>{_esc(date_str)}</span>"
            f"{chip_html}"
            f"{evidence_chips}"
            "</div></header>"
            "<div class='project-body'>"
            f"{photo_block}"
            f"<div class='project-bsr'>{bsr_html}</div>"
            "</div></section>"
        )
    return "".join(pages)


def _portfolio_css() -> str:
    """포트폴리오 인쇄용 CSS — A4 1장째 이력서, 2장째부터 프로젝트 보고서."""
    return """
@page { size: A4; margin: 14mm 12mm 14mm 12mm; }
* { box-sizing: border-box; }
html, body { margin:0; padding:0; }
body {
  font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background:#eef2f7; color:#1e293b; line-height:1.55;
  -webkit-font-smoothing:antialiased;
}
.portfolio-print-wrapper { padding:1.5rem 0.5rem 3rem 0.5rem; }
.portfolio-doc { max-width:840px; margin:0 auto; background:#ffffff;
  box-shadow:0 8px 24px rgba(15, 23, 42, 0.08); border-radius:14px; overflow:hidden; }

/* ─────────────────────  Resume (page 1)  ───────────────────── */
.resume-page { padding:30px 32px 28px 32px; }
.resume-header { position:relative; margin-bottom:18px; }
.resume-header-bar {
  height:6px; border-radius:6px;
  background:linear-gradient(90deg, #0f766e 0%, #14b8a6 60%, #5eead4 100%);
  margin-bottom:14px;
}
.resume-header-inner { display:flex; align-items:flex-end; justify-content:space-between; gap:1.5rem; }
.resume-eyebrow { margin:0; font-size:0.72rem; letter-spacing:0.08em; text-transform:uppercase;
  color:#64748b; font-weight:600; }
.resume-name { margin:0.2rem 0 0.1rem 0; font-size:2.1rem; line-height:1.05; color:#0f172a;
  letter-spacing:-0.02em; font-weight:800; }
.resume-subname { margin:0; color:#475569; font-size:0.9rem; font-weight:500; }
.resume-motto { margin:0.4rem 0 0; padding:0.35rem 0.7rem; border-left:3px solid #14b8a6;
  color:#334155; font-size:0.9rem; font-style:italic; background:#f0fdfa; border-radius:0 6px 6px 0; }

.resume-quick-metrics { display:flex; gap:0.5rem; flex-shrink:0; }
.resume-quick-metrics .qm { min-width:70px; padding:0.5rem 0.7rem; border-radius:10px;
  background:#f8fafc; border:1px solid #e2e8f0; text-align:center; }
.resume-quick-metrics .qm-num { display:block; font-size:1.05rem; color:#0f766e; font-weight:800; }
.resume-quick-metrics .qm-lab { display:block; font-size:0.7rem; color:#64748b; margin-top:0.15rem; letter-spacing:0.04em; }

.resume-grid { display:grid; grid-template-columns:230px 1fr; gap:24px; margin-top:8px; }
.resume-side { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 14px 16px 14px; }
.resume-photo { width:100%; aspect-ratio:3/4; object-fit:cover; border-radius:8px;
  display:block; box-shadow:0 1px 4px rgba(15, 23, 42, 0.08); }
.resume-photo--placeholder { background:#e2e8f0; display:flex; align-items:center;
  justify-content:center; color:#94a3b8; font-size:0.78rem; letter-spacing:0.08em; }
.side-h { font-size:0.78rem; letter-spacing:0.08em; text-transform:uppercase; color:#0f766e;
  margin:14px 0 6px 0; font-weight:700; border-bottom:1px solid #cbd5e1; padding-bottom:3px; }
.resume-contact-list { list-style:none; padding:0; margin:0; }
.resume-contact-list li { display:flex; gap:6px; font-size:0.78rem; color:#334155;
  padding:3px 0; border-bottom:1px dashed #e2e8f0; }
.resume-contact-list li:last-child { border-bottom:none; }
.resume-contact-list .label { width:46px; flex-shrink:0; color:#94a3b8; font-weight:600; letter-spacing:0.04em; }
.resume-contact-list .value { color:#1e293b; word-break:break-all; }

.ncs-chip-grid { display:flex; flex-wrap:wrap; gap:4px; margin-top:4px; }
.ncs-chip { display:inline-flex; align-items:center; gap:4px;
  background:rgba(15,118,110,0.08); color:#0f766e; border:1px solid rgba(15,118,110,0.18);
  border-radius:999px; padding:3px 8px; font-size:0.7rem; font-weight:600; }
.ncs-chip strong { color:#115e59; font-weight:700; }
.ncs-chip--muted { background:transparent; color:#94a3b8; border:1px dashed #cbd5e1; font-style:italic; font-weight:500; }

.resume-skills { display:flex; flex-direction:column; gap:6px; margin-top:4px; }
.skill-row { display:grid; grid-template-columns:64px 1fr 24px; align-items:center; gap:6px; }
.skill-name { font-size:0.74rem; color:#0f172a; font-weight:600; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.skill-bar-track { height:7px; background:#e2e8f0; border-radius:999px; position:relative; overflow:hidden; }
.skill-bar-fill { height:100%; background:linear-gradient(90deg, #0f766e, #14b8a6, #5eead4);
  border-radius:999px; }
.skill-score { font-size:0.7rem; color:#475569; text-align:right; font-weight:700; }

.resume-main { display:flex; flex-direction:column; gap:16px; }
.resume-section-title { margin:0 0 6px 0; font-size:0.95rem; color:#0f766e; font-weight:700;
  letter-spacing:0.02em; padding-bottom:3px; border-bottom:1.5px solid #14b8a6; }
.timeline-list { list-style:none; padding:0; margin:0; }
.timeline-item { display:grid; grid-template-columns:118px 1fr; gap:10px; padding:8px 0;
  border-bottom:1px dashed #e2e8f0; }
.timeline-item:last-child { border-bottom:none; }
.timeline-period { font-size:0.78rem; color:#64748b; font-weight:600; padding-top:1px; }
.timeline-title { font-size:0.92rem; color:#0f172a; font-weight:700; margin-bottom:2px;
  display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
.timeline-role-chip { font-size:0.66rem; padding:1px 7px; background:#0f766e; color:#fff;
  border-radius:999px; font-weight:600; letter-spacing:0.04em; }
.timeline-desc { font-size:0.83rem; color:#334155; line-height:1.5; }
.resume-table { width:100%; border-collapse:collapse; font-size:0.83rem; }
.resume-table thead th { text-align:left; color:#0f766e; font-weight:700;
  border-bottom:1.5px solid #14b8a6; padding:5px 6px; font-size:0.78rem; letter-spacing:0.02em; }
.resume-table tbody td { padding:5px 6px; border-bottom:1px dashed #e2e8f0; color:#334155; }
.resume-empty { color:#94a3b8; font-style:italic; font-size:0.83rem; padding:6px 0 8px 0; margin:0; }

/* ─────────────────────  Project Pages (page 2+)  ───────────────────── */
.project-cover { padding:80px 40px 60px 40px; text-align:center;
  background:linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 100%);
  border-top:1px solid #e2e8f0; page-break-before:always; }
.project-cover-title { margin:0.4rem 0 0.6rem; font-size:2rem; color:#0f766e;
  letter-spacing:-0.02em; font-weight:800; }
.project-cover-sub { color:#475569; max-width:520px; margin:0 auto; font-size:0.92rem; }

.project-page { padding:30px 32px; border-top:1px solid #e2e8f0;
  page-break-before:always; page-break-inside:avoid; break-inside:avoid; }
.project-header { margin-bottom:14px; padding-bottom:10px; border-bottom:2px solid #0f766e; }
.project-num { font-size:0.72rem; letter-spacing:0.16em; color:#0f766e; font-weight:700;
  text-transform:uppercase; }
.project-title { margin:0.25rem 0 0.4rem; font-size:1.4rem; color:#0f172a; font-weight:700; letter-spacing:-0.01em; }
.project-meta { display:flex; gap:6px; flex-wrap:wrap; }
.project-meta-chip { display:inline-block; font-size:0.74rem; padding:3px 8px;
  background:#f1f5f9; color:#475569; border-radius:6px; font-weight:600; }
.project-meta-chip--evidence { background:rgba(15,118,110,0.1); color:#0f766e; }
.project-meta-chip--audio { background:rgba(20,184,166,0.12); color:#0d9488; }
.project-body { font-size:0.92rem; line-height:1.65; display:grid;
  grid-template-columns:1fr; gap:14px; }
.project-page--has-photo .project-body { grid-template-columns:minmax(220px, 38%) 1fr; }
.project-photo { margin:0; padding:0; }
.project-photo img { width:100%; height:auto; max-height:320px; object-fit:cover;
  border-radius:10px; border:1px solid #e2e8f0;
  box-shadow:0 2px 6px rgba(15, 23, 42, 0.06); display:block; }
.project-photo figcaption { font-size:0.74rem; color:#94a3b8; margin-top:5px;
  text-align:right; letter-spacing:0.04em; }
.project-bsr { padding:14px 16px; background:#f8fafc; border-radius:10px;
  border-left:4px solid #14b8a6; }
.project-bsr [data-section] { padding-bottom:5px; }

/* ─────────────────────  Print Styles  ───────────────────── */
@media print {
  @page { size: A4; margin: 12mm; }
  html, body { background:#ffffff !important; }
  body { -webkit-print-color-adjust:exact !important; print-color-adjust:exact !important; }
  .portfolio-print-wrapper { padding:0 !important; }
  .portfolio-doc { box-shadow:none !important; border-radius:0 !important; max-width:none !important; margin:0 !important; }

  .resume-page { padding:0 !important; page-break-after:always; page-break-inside:avoid; break-inside:avoid; }
  .resume-grid { grid-template-columns:200px 1fr !important; gap:14px !important; }
  .resume-side { padding:10px 10px 12px 10px !important; }

  .project-cover { padding:30mm 20mm 20mm 20mm !important; page-break-before:always; }
  .project-page { padding:0 !important; page-break-before:always !important;
    page-break-inside:avoid !important; break-inside:avoid !important; }

  /* 차트·표 페이지 분할 보호 */
  .skill-row, .timeline-item, .resume-table tr,
  .project-bsr, .project-header { page-break-inside:avoid; break-inside:avoid; }

  /* Streamlit 잡티 제거 (Streamlit 내 인쇄 시 안전장치) */
  [data-testid="stSidebar"], [data-testid="stToolbar"], header, footer,
  .stDeployButton, .stDownloadButton { display:none !important; }
}
""".strip()


def _portfolio_sel_key(uid: str, log_id: int | str) -> str:
    return f"port_sel_{uid}_{log_id}"


def _port_log_checkboxes_dict_key(uid: str, log_id: int) -> str:
    return f"{uid}_{log_id}"


def _init_portfolio_select_state() -> None:
    if "port_select_all_logs" not in st.session_state:
        st.session_state.port_select_all_logs = False
    if "port_log_checkboxes" not in st.session_state:
        st.session_state.port_log_checkboxes = {}


def _collect_portfolio_log_ids(logs: list[dict]) -> list[int]:
    log_ids: list[int] = []
    for row in logs:
        try:
            lid = int(row.get("id") or 0)
        except (TypeError, ValueError):
            continue
        if lid > 0:
            log_ids.append(lid)
    return log_ids


def _sync_portfolio_checkbox_states(uid: str, log_ids: list[int]) -> None:
    """포트폴리오 수록 체크박스 ↔ port_log_checkboxes·port_select_all_logs 동기화."""
    _init_portfolio_select_state()
    for lid in log_ids:
        cb_key = _portfolio_sel_key(uid, lid)
        dict_key = _port_log_checkboxes_dict_key(uid, lid)
        if cb_key in st.session_state:
            st.session_state.port_log_checkboxes[dict_key] = bool(st.session_state[cb_key])
        elif dict_key in st.session_state.port_log_checkboxes:
            st.session_state[cb_key] = st.session_state.port_log_checkboxes[dict_key]
    if log_ids:
        st.session_state.port_select_all_logs = all(
            st.session_state.port_log_checkboxes.get(
                _port_log_checkboxes_dict_key(uid, lid), False
            )
            for lid in log_ids
        )


def _on_toggle_portfolio_select_all(uid: str, log_ids: list[int]) -> None:
    """포트폴리오 수록 항목 전체 선택/해제."""
    _init_portfolio_select_state()
    new_val = not st.session_state.port_select_all_logs
    st.session_state.port_select_all_logs = new_val
    for lid in log_ids:
        dict_key = _port_log_checkboxes_dict_key(uid, lid)
        st.session_state.port_log_checkboxes[dict_key] = new_val
        st.session_state[_portfolio_sel_key(uid, lid)] = new_val


def _show_digital_portfolio(uid: str) -> None:
    """디지털 직무 포트폴리오 화면 — 1페이지 비주얼 이력서 + 2페이지 프로젝트 보고서."""

    profile = get_student_profile(uid)
    logs = list_logs(uid)
    prog = seed_progress_if_missing(uid, DEFAULT_NCS_PROGRESS)

    # ─── 페이지 상단: 안내 헤더 ───
    _render_page_header(
        eyebrow="MY DIGITAL PORTFOLIO",
        title="NCS 종합 직무 포트폴리오",
        desc=(
            "본 포트폴리오는 <strong>1페이지 이력서</strong>(About Me · Tech Stack · 학력 · 경력)와 "
            "<strong>2페이지 이후의 프로젝트 보고서</strong>로 구성됩니다. "
            "이력서 정보는 <strong>[내 프로필 관리]</strong> 메뉴에서, 프로젝트는 하단의 체크박스에서 선택하시기 바랍니다."
        ),
    )

    if not (profile.get("full_name") or "").strip():
        st.markdown(
            "<div class='empty-state' style='margin-bottom:1rem;'>"
            "<p class='empty-state__title'>프로필 정보가 입력되지 않았습니다</p>"
            "<p class='empty-state__desc'>"
            "[내 프로필 관리] 메뉴에서 이름·사진·경력·기술 스택을 입력하시면 정식 이력서가 자동으로 생성됩니다. "
            "현재는 학생 ID로 임시 표시됩니다."
            "</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    avg_prog = round(sum(prog.values()) / max(len(prog), 1), 1) if prog else 0
    _render_dash_chips([
        {"label": "누적 실습", "value": f"{len(logs)} 회"},
        {"label": "평균 NCS 진도", "value": f"{avg_prog} %"},
        {"label": "기술 스택", "value": f"{len(profile.get('tech_stack') or [])} 개"},
        {"label": "프로필 사진", "value": "있음" if (profile.get('photo_b64') or '').strip() else "없음"},
    ])

    # ── 베스트 실습 큐레이션 ──
    st.markdown(
        "<div class='action-strip'>"
        "<div class='action-strip__text'>"
        "<p class='action-strip__title'>포트폴리오 수록 실습 선택</p>"
        "<p class='action-strip__sub'>월별로 그룹화하여 표시합니다. 선택된 일지만 2페이지 이후의 프로젝트 보고서에 포함됩니다.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    month_groups: dict[str, list[dict]] = {}
    for row in logs:
        date_str = log_display_date(row)
        d = parse_calendar_date(date_str)
        if d is None:
            key = "0000-00"
            label = "날짜 미상"
            sort_date = datetime.date.min
        else:
            key = f"{d.year:04d}-{d.month:02d}"
            label = f"{d.year}년 {d.month}월"
            sort_date = d
        bucket = month_groups.setdefault(key, [])
        bucket.append(
            {
                "_row": row,
                "_sort_date": sort_date,
                "_label": label,
                "_created": str(row.get("created_at") or "").strip(),
                "_id": int(row.get("id") or 0),
            }
        )

    sorted_month_keys = sorted(month_groups.keys(), reverse=True)
    log_ids = _collect_portfolio_log_ids(logs)
    _sync_portfolio_checkbox_states(uid, log_ids)

    select_all_label = (
        "☐ 전체 해제" if st.session_state.port_select_all_logs else "☑️ 전체 선택"
    )
    st.button(
        select_all_label,
        key=f"port_select_all_btn_{uid}",
        disabled=not log_ids,
        on_click=_on_toggle_portfolio_select_all,
        args=(uid, log_ids),
    )

    selected_ids: list[int] = []
    for idx, mkey in enumerate(sorted_month_keys):
        entries = sorted(
            month_groups[mkey],
            key=lambda e: (e["_sort_date"], e["_created"], e["_id"]),
            reverse=True,
        )
        month_label = entries[0]["_label"] if entries else mkey
        is_first = idx == 0
        with st.expander(
            f"{month_label} 실습 기록 ({len(entries)}건)",
            expanded=is_first,
        ):
            for e in entries:
                row = e["_row"]
                try:
                    lid = int(row.get("id") or 0)
                except (TypeError, ValueError):
                    lid = 0
                d_sort = e["_sort_date"]
                if d_sort == datetime.date.min:
                    date_short = (row.get("date") or "—")
                else:
                    date_short = f"{d_sort.month:02d}.{d_sort.day:02d}"
                ncs_name = _clean_ncs_unit_name(row.get("ncs_unit", "") or "")
                snippet = _bsr_preview_snippet(row.get("bsr") or "", max_len=30)
                if ncs_name and snippet:
                    label = f"[{date_short}] {ncs_name} | {snippet}"
                elif ncs_name:
                    label = f"[{date_short}] {ncs_name}"
                elif snippet:
                    label = f"[{date_short}] {snippet}"
                else:
                    label = f"[{date_short}]"
                cb_key = _portfolio_sel_key(uid, lid)
                dict_key = _port_log_checkboxes_dict_key(uid, lid)
                if dict_key in st.session_state.port_log_checkboxes:
                    st.session_state[cb_key] = st.session_state.port_log_checkboxes[dict_key]
                checked = st.checkbox(label, key=cb_key)
                st.session_state.port_log_checkboxes[dict_key] = checked
                if checked:
                    selected_ids.append(lid)

    if log_ids:
        st.session_state.port_select_all_logs = all(
            st.session_state.port_log_checkboxes.get(
                _port_log_checkboxes_dict_key(uid, x), False
            )
            for x in log_ids
        )
    selected_logs = [r for r in logs if int(r.get("id") or 0) in selected_ids]

    # ── HTML 조립 ──
    resume_html = _build_resume_page_html(uid, profile, prog, logs)
    projects_html = _build_project_pages_html(selected_logs)
    portfolio_css = _portfolio_css()
    inner_html = (
        "<div class='portfolio-print-wrapper'><div class='portfolio-doc'>"
        f"{resume_html}"
        f"{projects_html}"
        "</div></div>"
    )

    full_html = (
        "<!DOCTYPE html><html lang='ko'><head><meta charset='utf-8'/>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'/>"
        "<title>NCS 직무 포트폴리오</title>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?"
        "family=Noto+Sans+KR:wght@400;500;600;700;800&display=swap' rel='stylesheet'>"
        f"<style>{portfolio_css}</style>"
        "</head><body>"
        + inner_html
        + "</body></html>"
    )

    st.download_button(
        label="포트폴리오 HTML 다운로드 (브라우저에서 Ctrl+P → PDF 저장)",
        data=full_html.encode("utf-8"),
        file_name=f"{uid}_portfolio.html",
        mime="text/html",
        key=f"portfolio_html_dl_{uid}",
        type="primary",
        width="stretch",
        icon=":material/download:",
    )
    st.caption(
        "다운로드한 HTML 파일을 브라우저에서 열고 Ctrl+P 인쇄 대화상자의 [PDF로 저장]을 선택하면 "
        "A4 인쇄 및 이메일 첨부 형식으로 활용할 수 있습니다."
    )

    st.subheader("포트폴리오 미리보기")
    st.caption("실제 다운로드되는 결과물과 동일한 형식으로 표시됩니다. 상단에서 실습 선택을 변경하면 즉시 반영됩니다.")
    render_portfolio_print_button(key=f"portfolio_print_{uid}")
    # st.markdown + unsafe_allow_html는 복잡한 HTML을 이스케이프해 태그가 그대로 보이는 경우가 있음.
    # st.html(1.33+) / components.html로 전체 DOM을 iframe에 렌더한다.
    _preview_html = f"<style>{portfolio_css}</style>{inner_html}"
    if hasattr(st, "html"):
        st.html(_preview_html)
    else:
        import streamlit.components.v1 as components

        components.html(_preview_html, height=1800, scrolling=True)

