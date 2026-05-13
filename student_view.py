import datetime
import hashlib
import html
import io
import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from bsr_utils import (
    GEMINI_TEXT_MODEL_CANDIDATES,
    GEMINI_VISION_MODEL_CANDIDATES,
    check_evidence_validity,
    extract_background_section,
    extract_bsr_section,
    gemini_generate_text,
    generate_bsr_draft_from_keywords,
    get_ai_scaffolding,
    get_reflection_example_sentence,
    radar_scores_from_logs,
    render_bsr_highlighted,
    resolve_google_api_key,
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

from db import (
    add_log,
    app_today,
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
from ui_style import P, render_password_change_expander

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


def _img_analysis_cache_sig(uid: str, img, *, use_real_ai: bool, content: str) -> str:
    """
    같은 사진(들)·같은 모드면 텍스트 입력만 바뀌어도 실제 Vision API는 재호출하지 않도록 시그니처.
    시뮬레이션 경로는 본문이 결과에 영향 → 본문 해시 포함.
    여러 장 업로드되면 모든 파일의 (name, size)를 시그니처에 누적한다.
    """
    files = _normalize_img_input(img)
    force_sim = st.session_state.get("analyze_force_sim_mode", False)
    parts: list[str] = []
    parts.append(f"n={len(files)}")
    for f in files:
        name = getattr(f, "name", "") or ""
        size = _evidence_file_size(f)
        parts.append(f"{name}#{size}")
    parts.append(str(bool(use_real_ai)))
    parts.append(str(bool(force_sim)))
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
    세션에 캐시된 시그니처와 같으면 analyze_image()를 호출하지 않고 캐시만 사용.
    img는 단일 파일/리스트/None 모두 받음 (내부에서 list로 정규화).
    반환: (detected, suggested_unit, safety_advice)
    """
    files = _normalize_img_input(img)
    sig_key = f"img_analysis_sig_{uid}"
    result_key = f"img_result_{uid}"
    sig = _img_analysis_cache_sig(uid, files, use_real_ai=use_real_ai, content=content)

    if st.session_state.get(sig_key) == sig and result_key in st.session_state:
        t = st.session_state[result_key]
        if isinstance(t, (list, tuple)) and len(t) >= 3:
            return list(t[0]), str(t[1]), str(t[2])
        if isinstance(t, (list, tuple)) and len(t) == 2:
            return list(t[0]), str(t[1]), ""

    force_sim = st.session_state.get("analyze_force_sim_mode", False)
    primary_name = getattr(files[0], "name", "") if files else ""
    result = analyze_image(
        files,
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
    sig = _img_analysis_cache_sig(uid, img, use_real_ai=use_real_ai, content=content)
    return st.session_state.get(sig_key) == sig and result_key in st.session_state


def _evidence_validity_sig(uid: str, img, *, use_real_ai: bool, content: str) -> str:
    """본문·이미지(들) 조합이 같을 때만 증거 연관성 점수를 캐시."""
    base = _img_analysis_cache_sig(uid, img, use_real_ai=use_real_ai, content=content)
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
학생이 작성한 일상적 말투의 실습 성찰(B-S-R)을 NCS 수행준거 양식의 격식 있는 문장으로 변환해 주세요.

【절대 규칙 — 위반 시 잘못된 응답으로 간주】
1. 절대로 별표 두 개, 밑줄 두 개 등 마크다운 강조 기호를 사용하지 말고 순수 텍스트만 출력할 것.
2. 반드시 [배경], [해결], [성과] 세 가지 태그를 모두 포함하여 단락을 나눌 것. 어느 하나라도 누락하면 안 됨.
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
- [배경], [해결], [성과] 순서로 각 태그 뒤에 한 칸 띄운 뒤 본문을 쓸 것
- [체크리스트: …]가 입력에 있으면 그대로 유지
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
        prompt = _build_polish_prompt(bsr_text, ncs_unit, ncs_element)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.3, "max_output_tokens": 2048},
        )
        if out:
            return _strip_polish_markdown(out.strip())
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


def _discover_gemini_generate_model_ids(genai) -> list[str]:
    """API에 등록된 generateContent용 gemini 모델 id (404 대비 2차 시도)."""
    found: list[str] = []
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" not in methods:
                continue
            raw = getattr(m, "name", "") or ""
            name = raw.split("/", 1)[-1] if "/" in raw else raw
            low = name.lower()
            if "gemini" not in low:
                continue
            if any(
                x in low
                for x in (
                    "embed",
                    "embedding",
                    "tts",
                    "imagen",
                    "veo",
                    "lyria",
                    "music",
                    "robotics",
                    "computer-use",
                )
            ):
                continue
            found.append(name)
    except Exception:
        pass
    seen: set[str] = set()
    uniq: list[str] = []
    for n in found:
        if n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq


def _gemini_vision_generate(genai, pil_imgs, prompt: str) -> tuple[str, str]:
    """이미지(여러 장 가능)+프롬프트로 텍스트 응답. (text, 사용한 모델명).

    pil_imgs: 단일 PIL.Image 또는 List[PIL.Image]. 둘 다 허용한다.
    여러 장이면 [prompt, *pil_imgs] 순서로 Gemini에 전달해 모든 사진을 한 번에 분석한다.
    모두 실패 시 누적 오류를 담아 raise.
    """
    if not isinstance(pil_imgs, (list, tuple)):
        pil_imgs = [pil_imgs]
    pil_imgs = [p for p in pil_imgs if p is not None]
    if not pil_imgs:
        raise ValueError("분석할 이미지가 없습니다.")

    last_err: Exception | None = None
    attempt_logs: list[str] = []
    tried: set[str] = set()
    ordered = list(GEMINI_VISION_MODEL_CANDIDATES) + [
        m for m in _discover_gemini_generate_model_ids(genai) if m not in GEMINI_VISION_MODEL_CANDIDATES
    ]
    for model_name in ordered:
        if model_name in tried:
            continue
        tried.add(model_name)
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                [prompt, *pil_imgs],
                generation_config={"temperature": 0.2, "max_output_tokens": 1024},
            )
            text = ""
            try:
                text = (response.text or "").strip()
            except ValueError:
                if response.candidates:
                    parts = getattr(response.candidates[0].content, "parts", None) or []
                    for p in parts:
                        text += getattr(p, "text", "") or ""
                text = text.strip()
            if text:
                return text, model_name
            msg = f"{model_name}: 응답이 비었거나 안전 필터로 차단되었을 수 있습니다."
            attempt_logs.append(msg)
            last_err = RuntimeError(msg)
        except Exception as e:
            attempt_logs.append(f"{model_name}: {e}")
            last_err = e
            continue
    detail = "\n".join(attempt_logs) if attempt_logs else "(시도 로그 없음)"
    raise RuntimeError(
        "모든 Gemini 이미지 모델에서 실패했습니다.\n\n" + detail
    ) from last_err


def analyze_image(image_file, use_real_api: bool = True, content: str = "", file_name: str = "") -> tuple[list[dict], str, str]:
    """
    실습 사진 분석. use_real_api=False이거나 Quota 초과 시 시뮬레이션 모드로 자동 전환.
    image_file은 단일 UploadedFile 또는 List[UploadedFile] 모두 허용.
    여러 장이 들어오면 모든 사진을 같은 실습 컨텍스트로 묶어 한 번에 Gemini에 전달한다.
    반환: (탐지된 장비 목록, 추천 NCS 단위, 안전 조언)
    """
    files = _normalize_img_input(image_file)
    primary_name = file_name or (getattr(files[0], "name", "") if files else "")

    use_sim_key = "analyze_force_sim_mode"
    if st.session_state.get(use_sim_key, False):
        use_real_api = False
    if not use_real_api:
        sim_text = _get_simulation_response(primary_name, content)
        return _parse_ai_response(sim_text)

    if not files:
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
        from PIL import Image

        # 모든 사진을 PIL로 디코드. 일부만 손상돼도 가능한 사진들로 분석 진행.
        pil_images: list = []
        failed_count = 0
        for f in files:
            try:
                f.seek(0)
                img_bytes = f.read()
                pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
            except (OSError, ValueError):
                failed_count += 1
                continue

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

        if failed_count:
            st.caption(
                f"※ 업로드한 {len(files)}장 중 {failed_count}장은 불러오지 못해 제외했습니다."
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
            return (
                [{"객체": "분석 결과 없음", "신뢰도": "—"}],
                "전자부품장착",
                "AI가 응답을 생성하지 못하였습니다. 다른 사진으로 재시도하시기 바랍니다.",
            )

        detected, suggested_unit, safety_advice = _parse_ai_response(response_text)
        st.session_state.pop("analyze_force_sim_mode", None)
        return detected, suggested_unit, safety_advice

    except Exception as e:
        err_msg = str(e).lower()
        with st.expander("오류 상세 (원인 확인용 — 반드시 펼쳐서 확인)", expanded=True):
            st.code(str(e)[:4000])
            st.caption(
                "Google Cloud 콘솔에서 **Generative Language API** 사용 설정·결제·할당량을 확인하세요. "
                "키에 **앱/웹 제한**이 걸려 있으면 로컬 Streamlit에서 막힐 수 있습니다."
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
        st.error(
            "**이미지 분석 API를 사용할 수 없습니다.**\n\n"
            "할당량 초과·네트워크 오류·인증 오류 등으로 요청이 완료되지 않았을 수 있습니다. "
            "**관리자에게 API 설정을 확인하세요.**"
        )
        # API 실패 시 견고한 방어: 시뮬레이션 모드로 전환해 시연 연속성 확보
        if (
            "quota" in err_msg
            or "resource exhausted" in err_msg
            or "network" in err_msg
            or "connection" in err_msg
            or "timeout" in err_msg
            or "429" in err_msg
            or "503" in err_msg
        ):
            st.session_state["analyze_force_sim_mode"] = True
            st.info("안정적인 시연을 위해 로컬 분석 모드로 전환합니다.")
            sim_text = _get_simulation_response(primary_name, content)
            return _parse_ai_response(sim_text)
        st.session_state["analyze_force_sim_mode"] = True
        st.info("안정적인 시연을 위해 로컬 분석 모드로 전환합니다.")
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


def _build_bsr_string(background: str, haegyul: str, seungwa: str, checked_items: list[str]) -> str:
    """표준 BSR 문자열 조합: [배경][해결][성과][체크리스트]"""
    parts = [f"[배경] {background or ''}"]
    if haegyul:
        parts.append(f"[해결] {haegyul}")
    if seungwa:
        parts.append(f"[성과] {seungwa}")
    if checked_items:
        parts.append(f"[체크리스트: {'; '.join(checked_items)}]")
    return " ".join(parts)


def _render_bsr_reflection_card_html(
    background: str,
    haegyul: str,
    seungwa: str,
    checked_items: list[str],
    polished: str | None,
) -> str:
    """원문 vs AI 다듬기 2열 + 단계별 화살표 (교육용 BSR 미리보기)."""
    pol = (polished or "").strip()
    pb = extract_bsr_section(pol, "배경") if pol else ""
    ph = extract_bsr_section(pol, "해결") if pol else ""
    ps = extract_bsr_section(pol, "성과") if pol else ""
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

    chunks.append(_pair("배경 · 문제", f"[배경] {background or ''}", f"[배경] {pb}" if pol else ""))
    chunks.append("<div class='bsr-flow-divider' aria-hidden='true'></div>")
    chunks.append(_pair("해결 과정", f"[해결] {haegyul or ''}", f"[해결] {ph}" if pol else ""))
    chunks.append("<div class='bsr-flow-divider' aria-hidden='true'></div>")
    chunks.append(_pair("성과 · 깨달음", f"[성과] {seungwa or ''}", f"[성과] {ps}" if pol else ""))

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
5. 입력에 [배경][해결][성과] 구조가 있으면 각 구간을 유지하며 다듬되, 원문에 없는 내용은 추가하지 말 것.

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


AI_GROWTH_PROMPT = """당신은 공업고등학교 NCS 직무 역량 코치입니다. 학생의 실습 BSR(배경-해결-성과) 이력을 분석하여 맞춤형 성장 조언을 작성해 주세요.

다음 3가지를 **전문가 톤**으로 조언해 주세요. 각 항목당 2~3문장.
1. 현재 가장 뛰어난 직무 강점 (구체적 사례 기반)
2. 보완이 필요한 성찰 포인트 (메타인지·과정 서술 강화 방향)
3. 다음 실습 시 도전해볼 '미션' (구체적 행동 제안 1가지)

BSR 이력:
---
{bsr_history}
---

조언만 출력. 마크다운 제목·번호 없이 본문만."""


def _get_ai_growth_report(bsr_logs: list[dict]) -> str | None:
    """Gemini API로 BSR 이력 기반 AI 맞춤형 성장 총평 생성. 실패 시 None."""
    if not bsr_logs:
        return None
    api_key = _get_google_api_key()
    if not api_key:
        return None
    history = "\n\n".join(
        f"[{r.get('date','')}] {r.get('ncs_unit','')}\n{str(r.get('bsr',''))[:800]}"
        for r in bsr_logs[:15]
    )
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        prompt = AI_GROWTH_PROMPT.format(bsr_history=history)
        out = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.5, "max_output_tokens": 1024},
        )
        if out:
            return out
    except Exception:
        pass
    return None


def _seungwa_char_count(bsr: str) -> int:
    """[성과] 구간 글자 수 (성찰 깊이 근사)."""
    m = re.search(r"\[성과\]\s*(.*?)(?=\[|$)", (bsr or ""), re.DOTALL)
    return len(m.group(1).strip()) if m else 0


def _professional_term_hits(bsr: str) -> int:
    """BSR에 등장하는 서로 다른 GLOSSARY·NCS 키워드 개수(빈도 근사)."""
    t = bsr or ""
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
        bsr = str(row.get("bsr") or "")
        date = row.get("date", "")
        unit = row.get("ncs_unit", "")
        sc = _seungwa_char_count(bsr)
        th = _professional_term_hits(bsr)
        lines.append(
            f"일지 {i} [{date}] 능력단위:{unit}\n"
            f"  - [성과] 구간 글자 수(성찰 깊이 근사): {sc}자\n"
            f"  - 전문 용어 매칭 빈도(근사): {th}\n"
            f"  - BSR 앞부분 요약: {(bsr[:200] + '…') if len(bsr) > 200 else bsr}"
        )
    return "\n\n".join(lines)


AI_META_COACH_PROMPT = """당신은 공업고등학교 NCS 직무 역량을 지도하는 교육·메타인지 코치입니다.

아래는 한 학생의 **최근 3개 실습 일지**에 대해 산출한 지표입니다. 각 일지마다 [성과] 구간 글자 수(성찰 깊이 근사)와 전문 용어 매칭 빈도를 비교할 수 있습니다.

---
{stats_block}
---

**작성 지침**
1. 세 일지를 비교하여 **성장한 점**을 구체적으로 칭찬하세요 (성찰의 깊이, 전문 용어 사용 측면).
2. **보완할 점**을 구체적으로 제시하세요. 특히 [성과] 구간이 짧거나 메타인지적 표현(이유, 깨달음, 다음에는 등)이 부족한 경우를 짚어 주세요.
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
    text = (bsr_text or "").strip()
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
        m = re.search(r"\[성과\]\s*(.*?)(?=\[|$)", bsr, re.DOTALL)
        if m:
            seg = m.group(1).strip()
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
        return "—", "[성과] 구간이 없어 성찰 수준을 평가할 수 없습니다."
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
        text = (row.get("bsr") or "").strip()
        for term in sorted(all_terms, key=len, reverse=True):
            if term in text:
                used[term] = used.get(term, 0) + 1
    return [t for t, _ in sorted(used.items(), key=lambda x: -x[1])]


def _compute_ncs_term_ratio(bsr_text: str) -> float:
    """BSR 내 구어체 대비 NCS 표준 용어 사용 비율(0~100)."""
    if not (t := (bsr_text or "").strip()):
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


def _render_ncs_progress_section(uid: str, *, compact: bool = True) -> None:
    """실시간 NCS 이수 현황 — 우측 좁은 패널(col_side)용 기본 compact 렌더링."""
    st.markdown('<div class="ncs-block">', unsafe_allow_html=True)
    st.markdown("#### 실시간 NCS 이수 현황")
    prog = st.session_state.ncs_progress or {}
    logs_for_chart = list_logs(uid)

    if not logs_for_chart:
        st.info("저장된 실습일지가 없음. 일지를 저장하면 NCS 진행률 및 그래프가 표시됨.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    n_units = max(len(prog), 1)
    avg_p = round(sum(prog.values()) / n_units, 1) if prog else 0.0
    if compact:
        st.metric("평균 NCS 진도율", f"{avg_p}%")
        mc1, mc2 = st.columns(2)
        mc1.metric("능력단위", f"{len(prog)}개")
        mc2.metric("누적 일지", f"{len(logs_for_chart)}건")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("평균 NCS 진도율", f"{avg_p}%")
        m2.metric("추적 중인 능력단위", f"{len(prog)}개")
        m3.metric("누적 실습 일지", f"{len(logs_for_chart)}건")

    if prog:
        with st.expander("능력단위별 진행률 (상세)", expanded=compact):
            for unit, val in prog.items():
                st.caption(format_ncs_unit(unit))
                st.progress(val / 100)

        # 좁은 우측 패널에서는 차트를 세로로 스택해 가독성 확보
        if compact:
            col_bar = st.container()
            col_radar = st.container()
        else:
            col_bar, col_radar = st.columns(2)

        with col_bar:
            st.caption("단위별 진행률")
            bar_df = pd.DataFrame(
                {"단위": [format_ncs_unit(u) for u in prog.keys()], "진행률(%)": list(prog.values())}
            )
            bar_fig = px.bar(
                bar_df,
                x="단위",
                y="진행률(%)",
                color_discrete_sequence=[_CHART_PRIMARY],
            )
            bar_fig.update_layout(
                margin=dict(l=30, r=20, t=20, b=35) if compact else dict(l=40, r=40, t=30, b=40),
                showlegend=False,
                xaxis_title="",
                yaxis_title="진행률(%)",
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                height=220 if compact else 320,
                font=dict(size=10 if compact else 12),
            )
            bar_fig.update_traces(marker_line_width=0)
            st.plotly_chart(bar_fig, width="stretch")

        with col_radar:
            st.caption("직무 영역 레이더 (설계 / 제작 / 계측 / 제어 / 안전)")
            text_all = " ".join(str(r.get("bsr", "")) for r in logs_for_chart)
            axes = ["설계", "제작", "계측", "제어", "안전"]
            keywords = {
                "설계": ["설계", "회로도", "스키매틱", "시뮬레이션"],
                "제작": ["조립", "납땜", "배선", "배관", "장착"],
                "계측": ["측정", "멀티미터", "오실로스코프", "메거", "계측"],
                "제어": ["PLC", "인버터", "시퀀스", "프로그램", "모터제어"],
                "안전": ["안전", "접지", "감전", "보호구", "LOTO", "인터록"],
            }
            scores = []
            for a in axes:
                s = sum(text_all.count(k) for k in keywords[a])
                scores.append(s)
            if sum(scores) == 0:
                scores = [1, 1, 1, 1, 1]

            values = np.array(scores, dtype=float)
            values = values / values.max() * 100.0
            r_vals = list(values) + [values[0]]
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
                        tickfont=dict(size=11, color="#64748b"),
                        gridcolor=f"rgba({_teal}, 0.12)",
                        linecolor=f"rgba({_teal}, 0.15)",
                    ),
                    angularaxis=dict(
                        tickfont=dict(size=12, color=P["text"]),
                        gridcolor=f"rgba({_teal}, 0.12)",
                    ),
                    bgcolor="rgba(248, 250, 252, 0.6)",
                ),
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                margin=dict(l=30, r=30, t=25, b=25) if compact else dict(l=70, r=70, t=50, b=50),
                showlegend=False,
                height=240 if compact else 340,
            )
            st.plotly_chart(fig, width="stretch")
    else:
        st.caption("NCS 진행 데이터가 아직 없습니다. 일지를 저장하면 반영됩니다.")

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
    text = str(bsr_text)
    text = re.sub(r"\[체크리스트:[^\]]*\]", "", text)
    text = re.sub(r"\[(?:배경|해결|성과)\]", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s{2,}", " ", text).strip()
    if not text:
        return ""
    return (text[:max_len] + "...") if len(text) > max_len else text


# ═══════════════════════════════════════════════════════════════════
# 학생 화면 가독성 개선용 작은 UI 헬퍼들 (ui_style.py의 CSS와 짝을 이룸)
# ═══════════════════════════════════════════════════════════════════
def _render_page_header(eyebrow: str, title: str, desc: str) -> None:
    """학생 페이지 상단에 친근한 안내 헤더를 렌더한다 (학생들이 무엇을 하는 화면인지 1초 내 파악)."""
    st.markdown(
        f"<div class='student-page-header'>"
        f"<div class='student-page-header__eyebrow'>{html.escape(eyebrow)}</div>"
        f"<h2 class='student-page-header__title'>{html.escape(title)}</h2>"
        f"<p class='student-page-header__desc'>{desc}</p>"
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
        f"<p class='step-card__sub'>{html.escape(sub)}</p>" if sub else ""
    )
    st.markdown(
        f"<div class='step-card__head'>"
        f"<span class='step-card__num'>{num}</span>"
        f"<div class='step-card__head-text'>"
        f"<p class='step-card__title'>{html.escape(title)}</p>"
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
        _render_page_header(
            eyebrow="STEP-BY-STEP",
            title="실습 일지 작성",
            desc=(
                "<strong>1</strong> 사진과 메모 업로드, "
                "<strong>2</strong> AI 초안 자동 생성, "
                "<strong>3</strong> 내 표현으로 정제하여 저장."
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
                    title="AI 초안 자동 생성",
                    sub="Step 1의 메모와 사진을 기반으로 AI가 [배경]·[해결]·[성과] 초안을 작성합니다.",
                    status=_step2_status[0],
                    status_kind=_step2_status[1],
                )
                # 모바일에서 가운데 정렬 컬럼이 좁아 보이므로, 전체 폭 버튼으로 단순화.
                do_ai_draft = st.button(
                    "AI BSR 초안 자동 생성",
                    key=f"bsr_ai_draft_{uid}",
                    type="primary",
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
                        with st.spinner("BSR 초안을 생성하는 중..."):
                            draft_d = generate_bsr_draft_from_keywords(
                                memo_raw,
                                detected_list,
                                _get_google_api_key() or "",
                            )
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
                                "초안 생성에 실패하였습니다. API 키 설정을 확인하거나 잠시 후 다시 시도해 주십시오.",
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
                        "생성된 BSR 초안 미리보기",
                        expanded=False,
                        icon=":material/list_alt:",
                    ):
                        if bg_state:
                            st.info(f"**[배경] 실습 상황**\n\n{bg_state}")
                        else:
                            st.caption("[배경] —")
                        if hg_state:
                            st.info(f"**[해결] 과정 및 해결 방법**\n\n{hg_state}")
                        else:
                            st.caption("[해결] —")
                        if sw_state:
                            st.success(f"**[성과] 학습 내용 및 성찰**\n\n{sw_state}")
                        else:
                            st.caption("[성과] —")

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
                        "[배경] 오늘의 실습 상황",
                        height=200,
                        placeholder=(
                            "예) 오늘은 PLC 시퀀스 실습을 수행하였습니다. "
                            "조립한 회로, 사용한 장비, 실습 목적 등을 기재하시기 바랍니다."
                        ),
                        key=f"content_{uid}",
                    )
                    ans = st.text_area(
                        "[해결] 과정 및 해결 방법",
                        height=200,
                        placeholder=(
                            "예) LED가 점등되지 않아 회로도를 재확인하고 극성을 점검하였습니다. "
                            "발생한 문제와 해결 절차를 단계별로 기재하시기 바랍니다."
                        ),
                        key=f"ans_haegyul_{uid}",
                    )
                    seungwa = st.text_area(
                        "[성과] 학습 내용 및 성찰",
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
                            st.caption("수행한 항목을 선택하면 저장 시 체크리스트가 BSR에 포함되어 기록됩니다.")
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
                            help="입력한 [배경]·[해결]·[성과]를 NCS 수행준거 양식의 정제된 문장으로 동시에 변환합니다.",
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
                        "다듬을 내용이 입력되지 않았습니다. [배경]·[해결]·[성과] 항목 중 "
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
                        _new_bg = extract_bsr_section(_polished, "배경") or _bg_now
                        _new_hg = extract_bsr_section(_polished, "해결") or _hg_now
                        _new_sw = extract_bsr_section(_polished, "성과") or _sw_now
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

                    log = {
                        "date": str(app_today()),
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
                        evidence_b64 = _photo_to_base64(primary_img, max_side=1080)

                    if imgs:
                        image_note_text = (
                            f"사진 {len(imgs)}장 업로드됨" if len(imgs) > 1 else "사진 업로드됨"
                        )
                    else:
                        image_note_text = None

                    add_log(
                        uid=uid,
                        date=log["date"],
                        ncs_unit=log["ncs"],
                        bsr=log["bsr"],
                        image_note=image_note_text,
                        image_b64=evidence_b64,
                        ncs_term_ratio=ncs_ratio,
                    )
                    progress_gain = min(8, max(2, (length_score + term_score + safety_score) // 2))
                    current = int((st.session_state.ncs_progress or {}).get(draft_save["unit"], 0))
                    new_val = min(current + progress_gain, 100)
                    st.session_state.ncs_progress[draft_save["unit"]] = new_val
                    update_progress(uid, draft_save["unit"], new_val)
                    st.session_state[draft_key] = None
                    st.success(
                        "실습 일지가 성공적으로 저장되었습니다.",
                        icon=":material/check_circle:",
                    )

        # 3. 오른쪽 좁은 영역 (col_side) — NCS 이수 현황
        with col_side:
            _render_ncs_progress_section(uid)

    elif nav == NAV_OPTIONS[2]:
        # ─── 페이지 상단: 안내 헤더 ───
        _render_page_header(
            eyebrow="MY PRACTICE LOGS",
            title="실습 이력 관리",
            desc=(
                "지금까지 작성한 실습 일지를 일괄 조회하고, 개별 일지의 상세 내용을 확인할 수 있습니다. "
                "<strong>CSV 다운로드</strong>로 기록을 내려받을 수 있습니다."
            ),
        )

        logs = list_logs(uid)
        if not logs:
            # ─── Empty State ───
            st.info(
                "**작성된 실습 기록이 존재하지 않습니다.**  \n"
                "사이드바의 **[실습 일지 작성]** 메뉴에서 첫 기록을 작성하시기 바랍니다.  \n"
                "Step 1·2·3 절차를 따라 NCS 기반 일지를 작성할 수 있습니다.",
                icon=":material/info:",
            )
            st.caption("일지가 한 건 이상 저장되면, 이 화면에 카드 형식으로 누적된 기록이 표시됩니다.")
        else:
            # ═══════════════════════════════════════════════════════
            # 1) 상단 대시보드 칩 — 한눈에 보이는 핵심 지표
            # ═══════════════════════════════════════════════════════
            _last_date = (logs[0].get("date") or "—") if logs else "—"
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
                "**나의 실습 카드**  \n"
                "카드 목록에서 일지를 개관하고, 하단의 [자세히 보기] 영역에서 개별 일지의 상세 내용을 확인하시기 바랍니다.",
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
                            (row.get("bsr", "") or "").replace('"', '""'),
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
                    "표 보기" if not _is_table_now else "카드 보기",
                    key=f"history_view_toggle_{uid}",
                    width="stretch",
                    icon=":material/table_chart:" if not _is_table_now else ":material/view_module:",
                    help="카드 그리드와 표 형식 간 보기 모드를 전환합니다.",
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
                    mbsr = (row.get("bsr", "") or "").replace("\n", " ")
                    msnippet = (mbsr[:40] + "…") if len(mbsr) > 40 else mbsr
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
                        "선택 일지 삭제",
                        key=f"manage_del_btn_{uid}",
                        width="stretch",
                        type="secondary",
                        icon=":material/delete:",
                    ):
                        if manage_selected:
                            delete_log(uid, int(manage_selected[0]))
                            st.success("선택한 일지가 삭제되었습니다.", icon=":material/check_circle:")
                            st.rerun()

                st.divider()
                st.caption(
                    "아래는 **이 계정의 실습 일지를 전부** 지웁니다. "
                    "NCS 이수 진행률 막대도 함께 초기화됩니다."
                )
                confirm_all = st.checkbox("모든 일지 삭제에 동의합니다", key=f"confirm_clear_{uid}")
                if st.button(
                    "모든 일지 삭제",
                    disabled=not confirm_all,
                    key=f"clear_all_{uid}",
                    width="stretch",
                    icon=":material/delete_forever:",
                ):
                    clear_logs(uid)
                    st.session_state.ncs_progress = seed_progress_if_missing(uid, DEFAULT_NCS_PROGRESS)
                    st.success("모든 일지와 진행률이 초기화되었습니다.", icon=":material/check_circle:")
                    st.rerun()

            # ═══════════════════════════════════════════════════════
            # 3) 카드 그리드 (또는 표 보기)
            # ═══════════════════════════════════════════════════════
            if st.session_state.get(f"history_show_table_{uid}", False):
                display_logs = []
                for r in logs:
                    bsr_clean = _bsr_preview_snippet(r.get("bsr", "") or "", max_len=60) or "—"
                    display_logs.append(
                        {
                            "ID": r.get("id"),
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
                cards_html: list[str] = []
                for r in logs:
                    rid = html.escape(str(r.get("id", "")))
                    rdate = html.escape(str(r.get("date") or "—"))
                    rncs = html.escape(_clean_ncs_unit_name(r.get("ncs_unit", "") or "") or "기록")
                    rbsr_full = r.get("bsr", "") or ""
                    rexcerpt_raw = _bsr_preview_snippet(rbsr_full, max_len=140) or "—"
                    rexcerpt = html.escape(rexcerpt_raw)
                    chip_imgs = ""
                    if r.get("image_note") or r.get("image_b64"):
                        chip_imgs += "<span class='history-card__chip history-card__chip--evidence'>사진</span>"
                    cards_html.append(
                        "<div class='history-card'>"
                        "<div class='history-card__top'>"
                        f"<span class='history-card__date'>{rdate}</span>"
                        f"<span class='history-card__id'>#{rid}</span>"
                        "</div>"
                        f"<p class='history-card__ncs'>{rncs}</p>"
                        f"<p class='history-card__excerpt'>{rexcerpt}</p>"
                        f"<div class='history-card__foot'>{chip_imgs}</div>"
                        "</div>"
                    )
                st.markdown(
                    "<div class='history-card-grid'>" + "".join(cards_html) + "</div>",
                    unsafe_allow_html=True,
                )

            # ═══════════════════════════════════════════════════════
            # 4) 상세 보기 & AI 변환 — 모바일에서 화면이 꽉 차지 않게 expander로 접음
            # ═══════════════════════════════════════════════════════
            with st.expander(
                "자세히 보기 및 AI 전문가 톤 변환",
                expanded=False,
                icon=":material/menu_book:",
            ):
                detail_options = [
                    (
                        r.get("id"),
                        f"[{r.get('date','—')}] {_clean_ncs_unit_name(r.get('ncs_unit','') or '') or '기록'}",
                    )
                    for r in logs
                ]
                selected_id = st.selectbox(
                    "상세 보기할 기록",
                    options=[o[0] for o in detail_options],
                    format_func=lambda x: next((o[1] for o in detail_options if o[0] == x), str(x)),
                    key=f"bsr_detail_{uid}",
                )
                selected_row = next((r for r in logs if r.get("id") == selected_id), None)

                if selected_row:
                    date_val = selected_row.get("date") or "—"
                    ncs_name = _clean_ncs_unit_name(selected_row.get("ncs_unit", "") or "") or "—"
                    bsr_raw = str(selected_row.get("bsr") or "")

                    # BSR 구조 파싱 (배경/해결/성과/체크리스트)
                    bg_sec = extract_bsr_section(bsr_raw, "배경")
                    hg_sec = extract_bsr_section(bsr_raw, "해결")
                    sw_sec = extract_bsr_section(bsr_raw, "성과")
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
                            f"letter-spacing:0.02em;margin-bottom:0.2rem;\">{label}</div>"
                            f"<div style=\"color:#1f2937;line-height:1.7;font-size:0.95rem;\">"
                            f"{body_safe}</div></div>"
                        )

                    sections_html = (
                        _detail_para("실습 배경 및 목표", bg_sec, "#2563eb")
                        + _detail_para("기술적 문제 해결 및 수행 과정", hg_sec, "#0f766e")
                        + _detail_para("직무 역량 성장 및 성찰", sw_sec, "#b45309")
                    )
                    if not sections_html:
                        sections_html = (
                            "<div style=\"color:#6b7280;font-size:0.9rem;padding:0.4rem 0;\">"
                            "구조화된 BSR 내용이 없습니다.</div>"
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
                        "<div style=\"display:flex;align-items:baseline;gap:0.6rem;"
                        "padding-bottom:0.55rem;margin-bottom:0.2rem;"
                        "border-bottom:1px solid #e2e8f0;\">"
                        f"<span style=\"font-size:0.85rem;color:#475569;"
                        f"background:#f1f5f9;padding:0.15rem 0.55rem;border-radius:4px;\">"
                        f"{html.escape(date_val)}</span>"
                        f"<span style=\"font-size:1.05rem;font-weight:600;color:#0f172a;\">"
                        f"{html.escape(ncs_name)}</span></div>"
                    )

                    st.markdown(
                        "<div style=\"background:#ffffff;border:1px solid #e2e8f0;"
                        "border-radius:10px;padding:1rem 1.25rem;margin-top:0.5rem;"
                        "box-shadow:0 1px 2px rgba(15,23,42,0.04);\">"
                        f"{title_html}{sections_html}{checklist_html}"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    # ── AI 변환 버튼 (모바일 친화: 전체 폭) ──
                    convert_clicked = st.button(
                        "AI 기반 NCS 전문가 톤 변환",
                        key=f"ncs_bsr_btn_{uid}_{selected_id}",
                        width="stretch",
                        icon=":material/auto_awesome:",
                        help="선택된 실습 내용을 NCS 표준 용어로 정제하여 표시합니다.",
                    )
                    st.caption("버튼을 누르면 선택된 실습 내용이 NCS 표준 용어 버전으로 변환됩니다.")

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
                            "border-radius:6px;color:#334155;line-height:1.7;\">"
                            f"{safe_rew}</div>",
                            unsafe_allow_html=True,
                        )

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
                "일지가 한 건 이상 저장되면 AI 성장 총평, 메타인지 코멘트, 역량 레이다 차트가 본 화면에 표시됩니다."
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
                            st.success(report_existing)
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
            # 3) 성찰 변화 가시화 (모바일에서 차트가 화면을 다 차지하지 않게 expander로 접음)
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
            # 4) 성찰 수준 자가 진단 + 마스터 전문 용어 (좌우)
            # ═══════════════════════════════════════════════════════
            col_reflect, col_terms = st.columns([1, 1], gap="medium")

            with col_reflect:
                with st.container(border=True):
                    _render_step_head(
                        num=4,
                        title="성찰 수준 자가 진단",
                        sub="작성한 [성과] 문장들의 깊이를 자동으로 평가합니다.",
                    )
                    level, comment = _evaluate_seungwa_reflection(logs)
                    if level == "높음":
                        st.success(
                            f"**현재 수준: 높음** — {comment}",
                            icon=":material/trending_up:",
                        )
                    elif level == "보통":
                        st.info(
                            f"**현재 수준: 보통** — {comment}",
                            icon=":material/trending_flat:",
                        )
                    else:
                        st.warning(
                            f"**현재 수준: 낮음** — {comment}",
                            icon=":material/trending_down:",
                        )
                    st.caption("'높음' 수준에 도달하기 위한 권장 성찰 키워드:")
                    recommend_kw = ["이유", "깨달음", "다음에는", "과정", "개선", "스스로", "이해", "알게"]
                    st.markdown(" ".join(f"`{k}`" for k in recommend_kw))

            with col_terms:
                with st.container(border=True):
                    _render_step_head(
                        num=5,
                        title="습득 전문 용어",
                        sub="작성된 일지에서 사용된 NCS 직무 전문 용어를 집계합니다.",
                    )
                    used_terms = _extract_used_professional_terms(logs)
                    if used_terms:
                        tags_html = " ".join(
                            f"<span style='display:inline-block;background:rgba(15,118,110,0.1);color:#0f766e;padding:0.25rem 0.55rem;margin:0.15rem;border-radius:999px;font-size:0.85rem;'>{html.escape(t)}</span>"
                            for t in used_terms[:40]
                        )
                        st.markdown(f"<div style='line-height:2;'>{tags_html}</div>", unsafe_allow_html=True)
                        st.caption(f"누적 사용 NCS 직무 전문 용어: {len(used_terms)}개")
                    else:
                        st.caption("매칭된 전문 용어가 존재하지 않습니다. NCS 직무 용어를 활용하시기 바랍니다.")

    elif nav == NAV_OPTIONS[4]:
        _show_digital_portfolio(uid)


def _photo_to_base64(uploaded_file, max_side: int = 720) -> str | None:
    """
    업로드된 사진을 정사각형에 가까운 형태로 보존한 채 최대 변 max_side 이하로 리사이즈하고
    JPEG로 base64 인코딩한 data URI를 반환. 실패 시 None.
    """
    try:
        from PIL import Image
        import base64

        img_bytes = uploaded_file.read() if hasattr(uploaded_file, "read") else bytes(uploaded_file)
        if not img_bytes:
            return None
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_side:
            ratio = max_side / float(max(w, h))
            img = img.resize((int(w * ratio), int(h * ratio)))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88, optimize=True)
        enc = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{enc}"
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
            "이력서 저장분 모두 지우기",
            disabled=not pr_ok,
            key=f"profile_reset_all_{uid}",
            width="stretch",
            icon=":material/delete_forever:",
        ):
            clear_student_profile(uid)
            st.success("이력서 저장 데이터를 삭제했습니다.", icon=":material/check_circle:")
            st.rerun()

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


def _logo_base64() -> str:
    """로고를 base64로 인코딩하여 HTML img src에 사용."""
    _APP_DIR = Path(__file__).resolve().parent
    for name in ["school_logo.png", "school_logo_placeholder.svg"]:
        p = _APP_DIR / "assets" / name
        if p.exists():
            try:
                import base64
                data = p.read_bytes()
                enc = base64.b64encode(data).decode()
                mime = "image/svg+xml" if name.endswith(".svg") else "image/png"
                return f"data:{mime};base64,{enc}"
            except Exception:
                pass
    return ""


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
        <p class='resume-subname'>{_esc(uid)} · 전기·전자과 산학일체형 도제생</p>
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
        "<p class='project-cover-sub'>NCS 직무 능력단위에 따라 실습 현장을 BSR(배경·해결·성과) 구조로 정리한 프로젝트 보고서 모음입니다.</p>"
        "</section>"
    )

    for idx, row in enumerate(selected_logs, start=1):
        bsr_raw = str(row.get("bsr") or "")
        bsr_html = render_bsr_highlighted(bsr_raw)
        ncs_display = format_ncs_unit(row.get("ncs_unit", ""))
        date_str = row.get("date", "")
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
        date_str = (row.get("date") or "").strip()
        try:
            d = datetime.date.fromisoformat(date_str)
            key = f"{d.year:04d}-{d.month:02d}"
            label = f"{d.year}년 {d.month}월"
            sort_date = d
        except ValueError:
            key = "0000-00"
            label = "날짜 미상"
            sort_date = datetime.date.min
        bucket = month_groups.setdefault(key, [])
        bucket.append({"_row": row, "_sort_date": sort_date, "_label": label})

    sorted_month_keys = sorted(month_groups.keys(), reverse=True)
    selected_ids: list[int] = []
    for idx, mkey in enumerate(sorted_month_keys):
        entries = sorted(month_groups[mkey], key=lambda e: e["_sort_date"], reverse=True)
        month_label = entries[0]["_label"] if entries else mkey
        is_first = idx == 0
        with st.expander(
            f"{month_label} 실습 기록 ({len(entries)}건)",
            expanded=is_first,
        ):
            for e in entries:
                row = e["_row"]
                lid = row.get("id")
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
                if st.checkbox(label, key=f"port_sel_{uid}_{lid}"):
                    selected_ids.append(lid)
    selected_logs = [r for r in logs if r.get("id") in selected_ids]

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

    st.markdown(
        "<h3 style='margin:1.4rem 0 0.5rem 0;font-size:1.12rem;color:#0f766e;'>"
        "포트폴리오 미리보기</h3>",
        unsafe_allow_html=True,
    )
    st.caption("실제 다운로드되는 결과물과 동일한 형식으로 표시됩니다. 상단에서 실습 선택을 변경하면 즉시 반영됩니다.")
    st.markdown(
        f"<style>{portfolio_css}</style>{inner_html}",
        unsafe_allow_html=True,
    )

