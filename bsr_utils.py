"""BSR 구조 시각화 공용 유틸. [배경][해결][성과] 구간별 색상 하이라이트 + NCS 전문 용어 강조."""
import hashlib
import io
import json
import logging
import os
import re
import time
from typing import Any

_log = logging.getLogger("ai_final.gemini")

# 레이더 역량 축 (학생·교사 뷰 공통)
RADAR_AXES: list[str] = ["설계", "제작", "계측", "제어", "안전"]

_RADAR_KEYWORDS: dict[str, list[str]] = {
    "설계": ["설계", "회로도", "스키매틱", "시뮬레이션"],
    "제작": ["조립", "납땜", "배선", "배관", "장착"],
    "계측": ["측정", "멀티미터", "오실로스코프", "메거", "계측"],
    "제어": ["PLC", "인버터", "시퀀스", "프로그램", "모터제어"],
    "안전": ["안전", "접지", "감전", "보호구", "LOTO", "인터록"],
}

# 실습 기록 수가 적을 때 한 축이 과도하게 100점이 되지 않도록 max 정규화 분모에 바닥을 둔다.
RADAR_MIN_LOGS_FOR_FULL_SCALE = 5
RADAR_MIN_MAX_DENOMINATOR = 5

# 음성(STT) 기능은 v2에서 제거됨 (사진 + 텍스트 입력으로 대체).
# 과거 호환을 위해 임포트 형태가 필요하면 student_view.py 등에서 정리할 것.


# 논문 시연 재현성: 핵심 AI 호출은 gemini-2.5-flash를 기본으로 고정한다.
# list_models()로 첫 모델을 고르지 않는다. 기본 모델 실패 시에만 아래 순서로 fallback.
GEMINI_PRIMARY_MODEL: str = "gemini-2.5-flash"
GEMINI_MODEL_TRY_ORDER: tuple[str, ...] = (
    GEMINI_PRIMARY_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
)
GEMINI_UNIFIED_MODEL: str = GEMINI_PRIMARY_MODEL
GEMINI_TEXT_MODEL_CANDIDATES: tuple[str, ...] = GEMINI_MODEL_TRY_ORDER
GEMINI_VISION_MODEL_CANDIDATES: tuple[str, ...] = GEMINI_MODEL_TRY_ORDER
LAST_GEMINI_MODEL: str | None = None
LAST_GEMINI_FALLBACK: bool = False
_PRIMARY_SKIP_UNTIL: float = 0.0

GEMINI_EMPTY_RESPONSE_MESSAGE: str = (
    "AI가 응답을 생성하지 못했습니다. 다른 사진이나 메모로 시도해 주세요."
)


def discover_gemini_generate_model_names(genai) -> list[str]:
    """현재 API 키로 ``list_models``에 노출된 ``generateContent`` 지원 모델 id(짧은 이름) 목록."""
    names: list[str] = []
    seen: set[str] = set()
    try:
        for m in genai.list_models():
            methods = getattr(m, "supported_generation_methods", None) or []
            if "generateContent" not in methods:
                continue
            raw = (getattr(m, "name", None) or "").strip()
            if not raw:
                continue
            short = raw.split("/", 1)[-1] if "/" in raw else raw
            low = short.lower()
            if "embed" in low or "bge" in low or "imagen" in low or "aqa" in low:
                continue
            if short not in seen:
                seen.add(short)
                names.append(short)
    except Exception:
        return []

    def _sort_key(s: str) -> tuple[int, int, str]:
        n = s.lower()
        if re.match(r"^gemini-2\.\d+-flash(?!-lite)", n):
            tier, sub = 0, 0
        elif "flash-lite" in n and "gemini-2" in n:
            tier, sub = 0, 1
        elif "gemini-2" in n and "flash" in n:
            tier, sub = 1, 0
        elif "gemini-1.5" in n and "flash" in n:
            tier, sub = 2, 0
        elif "gemini-1.5" in n and "pro" in n:
            tier, sub = 3, 0
        elif "gemini-2" in n:
            tier, sub = 4, 0
        elif "gemini" in n:
            tier, sub = 5, 0
        else:
            tier, sub = 9, 0
        preview = 1 if ("preview" in n or re.search(r"\bexp\b", n)) else 0
        return (tier, sub, preview, n)

    names.sort(key=_sort_key)
    return names


def mark_primary_unavailable(reason: str = "") -> None:
    """기본 모델이 429 등으로 막히면 짧은 시간 동안 fallback을 먼저 쓴다."""
    global _PRIMARY_SKIP_UNTIL
    _PRIMARY_SKIP_UNTIL = time.time() + 90.0
    _log.warning(
        "Gemini primary cooldown 90s: primary=%s reason=%s",
        GEMINI_PRIMARY_MODEL,
        reason or "unavailable",
    )


def _note_gemini_model(name: str, *, reason: str = "") -> None:
    """기본 모델이 아닌 경우에만 fallback 로그를 남긴다. UI에는 노출하지 않는다."""
    global LAST_GEMINI_MODEL, LAST_GEMINI_FALLBACK
    LAST_GEMINI_MODEL = name
    LAST_GEMINI_FALLBACK = name != GEMINI_PRIMARY_MODEL
    if LAST_GEMINI_FALLBACK:
        _log.warning(
            "Gemini fallback model used: primary=%s used=%s reason=%s",
            GEMINI_PRIMARY_MODEL,
            name,
            reason or "primary call failed",
        )
    else:
        _log.debug("Gemini model used: %s", name)


def resolved_gemini_model_candidates(
    genai=None,
    static_tail: tuple[str, ...] | None = None,
) -> list[str]:
    """``gemini-2.5-flash``를 항상 먼저 쓴다. list_models()로 순서를 바꾸지 않는다.

    ``genai`` 인자는 기존 호출부 호환용이며 선택 순서에 사용하지 않는다.
    """
    del genai  # 재현성을 위해 list_models 결과를 앞에 두지 않음
    tail = static_tail or GEMINI_MODEL_TRY_ORDER
    merged: list[str] = []
    seen: set[str] = set()
    skip_primary = time.time() < _PRIMARY_SKIP_UNTIL
    for n in (GEMINI_PRIMARY_MODEL, *tail):
        if skip_primary and n == GEMINI_PRIMARY_MODEL:
            continue
        if n and n not in seen:
            seen.add(n)
            merged.append(n)
    if skip_primary:
        _log.warning(
            "Gemini primary skipped (cooldown after 429/unavailable); using fallback first"
        )
    return merged


def gemini_safety_settings_block_none() -> list:
    """실습·공구·기계 묘사가 안전 필터에 걸려 빈 응답이 나오지 않도록 전 카테고리 BLOCK_NONE."""
    try:
        from google.generativeai.types import HarmBlockThreshold, HarmCategory

        t = HarmBlockThreshold.BLOCK_NONE
        out: list = []
        for attr in (
            "HARM_CATEGORY_HARASSMENT",
            "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "HARM_CATEGORY_DANGEROUS_CONTENT",
        ):
            cat = getattr(HarmCategory, attr, None)
            if cat is not None:
                out.append({"category": cat, "threshold": t})
        return out
    except Exception:
        return []


def extract_generate_content_text(response) -> str:
    """``response.text``만 쓰지 않고 candidates/parts를 훑어 본문을 수집한다(차단·빈 후보 대비).

    Gemini 2.x thinking 모델의 ``thought`` 파트는 제외하고 최종 답변 텍스트만 모은다.
    """
    if response is None:
        return ""

    def _part_text(part) -> str:
        # thinking/reasoning 파트는 화면에 쓰지 않음
        if getattr(part, "thought", False):
            return ""
        return getattr(part, "text", None) or ""

    chunks: list[str] = []
    try:
        for cand in getattr(response, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                t = _part_text(part)
                if t:
                    chunks.append(t)
    except Exception:
        pass
    if chunks:
        return "".join(chunks).strip()
    try:
        return (response.text or "").strip()
    except (ValueError, AttributeError, TypeError):
        pass
    try:
        for part in getattr(response, "parts", None) or []:
            t = _part_text(part)
            if t:
                chunks.append(t)
    except Exception:
        pass
    return "".join(chunks).strip()


def get_gemini_model(genai, model_name: str | None = None):
    """GenerativeModel 생성.

    - ``model_name``이 있으면 해당 이름만 시도하고, 실패 시 ``None``을 반환한다.
    - ``model_name``이 없으면 ``gemini-2.5-flash``를 먼저 시도하고,
      실패 시에만 정적 fallback 목록을 사용한다.
    """
    if model_name:
        try:
            return genai.GenerativeModel(model_name)
        except Exception:
            return None
    try:
        import streamlit as st

        _st = st
    except ImportError:
        _st = None
    for mn in resolved_gemini_model_candidates(genai):
        try:
            return genai.GenerativeModel(mn)
        except Exception:
            continue
    msg = "사용 가능한 Gemini 모델을 찾을 수 없습니다. API 키 상태를 확인하세요."
    if _st is not None:
        _st.error(msg)
        _st.stop()
    raise RuntimeError(msg)


def _finish_reason_is_max_tokens(finish_reason) -> bool:
    """finish_reason이 MAX_TOKENS(잘림)인지 판별."""
    if finish_reason is None:
        return False
    s = str(finish_reason).upper()
    return "MAX_TOKEN" in s or s.endswith("2") or s == "2"


def gemini_generate_text(genai, prompt: str, *, generation_config: dict | None = None) -> str | None:
    """generateContent 지원 모델을 순서대로 시도. 전부 실패 시 None.

    Gemini 2.x에서 ``max_output_tokens``가 부족하면 문장 중간에 끊긴 텍스트가
    반환될 수 있다. 그런 잘림(MAX_TOKENS) 응답은 다음 모델로 넘기고,
    모두 잘리기만 하면 그중 가장 긴 후보를 반환한다.
    """
    gc = dict(generation_config or {})
    # 호출부에서 너무 작게 준 경우 thinking 모델에서 본문이 잘리므로 하한 보정
    try:
        mot = int(gc.get("max_output_tokens") or 0)
    except (TypeError, ValueError):
        mot = 0
    if 0 < mot < 512:
        gc["max_output_tokens"] = 1024
    safety = gemini_safety_settings_block_none()
    kwargs: dict = {"generation_config": gc}
    if safety:
        kwargs["safety_settings"] = safety
    truncated_best: str | None = None
    last_err = ""
    for name in resolved_gemini_model_candidates(genai):
        try:
            model = get_gemini_model(genai, name)
            if model is None:
                last_err = f"{name}: GenerativeModel init failed"
                if name == GEMINI_PRIMARY_MODEL:
                    _log.warning("Gemini primary unavailable: %s", last_err)
                continue
            response = model.generate_content(prompt, **kwargs)
            text = extract_generate_content_text(response)
            if not text:
                last_err = f"{name}: empty response"
                if name == GEMINI_PRIMARY_MODEL:
                    _log.warning("Gemini primary empty response")
                continue
            fr = None
            try:
                cands = getattr(response, "candidates", None) or []
                if cands:
                    fr = getattr(cands[0], "finish_reason", None)
            except Exception:
                fr = None
            if _finish_reason_is_max_tokens(fr):
                last_err = f"{name}: MAX_TOKENS truncated"
                if truncated_best is None or len(text) > len(truncated_best):
                    truncated_best = text
                if name == GEMINI_PRIMARY_MODEL:
                    _log.warning("Gemini primary truncated (MAX_TOKENS); trying fallback")
                continue
            _note_gemini_model(name, reason=last_err)
            return text
        except Exception as e:
            last_err = f"{name}: {e}"
            if name == GEMINI_PRIMARY_MODEL:
                _log.warning("Gemini primary failed: %s", e)
                if "429" in str(e) or "quota" in str(e).lower():
                    mark_primary_unavailable(str(e)[:180])
            continue
    if truncated_best:
        _note_gemini_model("truncated-fallback", reason=last_err)
    return truncated_best


def resolve_google_api_key(explicit: str | None = None) -> str | None:
    """Streamlit secrets 우선, 없으면 환경 변수."""
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    try:
        import streamlit as st

        if hasattr(st, "secrets") and st.secrets.get("GOOGLE_API_KEY"):
            return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass
    return os.environ.get("GOOGLE_API_KEY")


# ── Gemini 비전: 업로드 이미지 리사이즈·JPEG 압축 (전송량·지연 감소) ─────────────────
VISION_MAX_IMAGE_SIDE: int = 1024
VISION_MAX_JPEG_BYTES: int = 500 * 1024


def pil_image_to_gemini_jpeg_bytes(im) -> bytes:
    """PIL 이미지를 최대 변 1024px 이하로 줄인 뒤 JPEG로 인코딩한다. 목표 용량 500KB 이하."""
    from PIL import Image

    img = im.convert("RGB")
    w, h = img.size
    max_side = VISION_MAX_IMAGE_SIDE
    if max(w, h) > max_side:
        r = max_side / float(max(w, h))
        img = img.resize(
            (max(1, int(round(w * r))), max(1, int(round(h * r)))),
            Image.Resampling.LANCZOS,
        )
    max_b = VISION_MAX_JPEG_BYTES
    best = b""
    q = 88
    while q >= 18:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(q), optimize=True)
        data = buf.getvalue()
        best = data
        if len(data) <= max_b:
            return data
        q -= 12
    factor = 0.87
    while min(img.width, img.height) >= 160:
        img = img.resize(
            (max(1, int(img.width * factor)), max(1, int(img.height * factor))),
            Image.Resampling.LANCZOS,
        )
        for q in (78, 68, 58, 48, 38, 28, 20, 18):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q, optimize=True)
            data = buf.getvalue()
            best = data
            if len(data) <= max_b:
                return data
    return best


def uploaded_files_to_gemini_pil_images(files: list) -> tuple[list, str]:
    """업로드 파일마다 Gemini 비전용 JPEG(500KB 이하 목표)로 압축한 PIL 목록과 핑거프린트를 반환한다.

    핑거프린트는 최종 JPEG 바이트열의 SHA-256(파일 구분자 포함)으로, 동일 사진 재업로드 시 동일 값이 된다.
    """
    from PIL import Image

    pils: list = []
    h = hashlib.sha256()
    for f in files:
        f.seek(0)
        raw = f.read()
        f.seek(0)
        try:
            src = Image.open(io.BytesIO(raw)).convert("RGB")
        except (OSError, ValueError):
            continue
        jpeg_bytes = pil_image_to_gemini_jpeg_bytes(src)
        if not jpeg_bytes:
            continue
        h.update(jpeg_bytes)
        h.update(b"\n---\n")
        pils.append(Image.open(io.BytesIO(jpeg_bytes)).convert("RGB"))
    if not pils:
        return [], ""
    return pils, h.hexdigest()


def pil_images_to_gemini_inline_parts(pil_images: list) -> list:
    """PIL 이미지를 Gemini ``generate_content``가 받는 ``inline_data`` Part dict 목록으로 변환한다.

    ``google-generativeai``는 ``{"inline_data": {"mime_type": "...", "data": <bytes>}}`` 형태를
    일반적으로 수용한다(Protobuf Blob). JPEG로 재인코딩해 픽셀 버퍼를 직접 넘기지 않는다.
    """
    from PIL import Image

    parts: list = []
    for img in pil_images or []:
        if img is None:
            continue
        im = img.convert("RGB").copy()
        im.load()
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=88, optimize=True)
        jpeg_bytes = buf.getvalue()
        if not jpeg_bytes:
            continue
        # Protobuf Blob.data는 bytes. (일부 예제는 base64 문자열을 쓰지만 SDK가 bytes를 기대하는 경우가 많다.)
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes}})
    return parts


def radar_scores_from_logs(logs: list[dict]) -> tuple[list[str], list[float]]:
    """일지 목록에서 역량 레이다용 축과 점수(0~100) 추출. 키워드 빈도 정규화."""
    axes = list(RADAR_AXES)
    text_all = " ".join(get_reflection_body(str(r.get("bsr", ""))) for r in logs)
    scores = [sum(text_all.count(k) for k in _RADAR_KEYWORDS[a]) for a in axes]
    if sum(scores) == 0:
        scores = [1, 1, 1, 1, 1]
    raw_max = max(scores)
    n_logs = len(logs)
    # 기록이 적을 때는 분모에 최소 기준(가상 '5회' 분량)을 두어 한 축이 쉽게 100점이 되지 않게 한다.
    if n_logs < RADAR_MIN_LOGS_FOR_FULL_SCALE:
        m = max(raw_max, RADAR_MIN_MAX_DENOMINATOR)
    else:
        m = max(raw_max, 1)
    values = [round(s / m * 100.0, 2) for s in scores]
    return axes, values


def extract_background_section(content: str) -> str:
    """What 또는 레거시 [배경] 구간. 없으면 전체를 사용 (사진-본문 대조용)."""
    rec = parse_reflection_record(content or "")
    for key in ("what", "legacy_background"):
        v = str(rec.get(key) or "").strip()
        if v:
            return v
    body = content or ""
    for tag in (_REFLECTION_SECTION_TAGS + (REFLECTION_META_TAG,)):
        if tag in body:
            body = body.split(tag, 1)[0]
    return body.strip() or (content or "").strip()


def extract_bsr_section(bsr_text: str, section: str) -> str:
    """
    일지 문자열에서 구간 본문을 추출한다.

    - 신규: ``What`` / ``So What`` / ``Now What`` (및 한글 별칭 경험·의미·적용)
    - 레거시: ``배경`` / ``해결`` / ``성과`` (기존 저장본 호환, 의미가 동일하다고 보지 않음)
    """
    rec = parse_reflection_record(bsr_text)
    alias = {
        "What": "what",
        "So What": "so_what",
        "Now What": "now_what",
        "경험": "what",
        "의미": "so_what",
        "적용": "now_what",
        "배경": "legacy_background",
        "해결": "legacy_solution",
        "성과": "legacy_reflection",
    }
    key = alias.get(section)
    if not key:
        return ""
    val = str(rec.get(key) or "").strip()
    if val:
        return val
    # 신규 일지를 옛 키로 요청하면 표시용으로만 대응 (1:1 의미 치환은 아님)
    if section == "배경":
        return str(rec.get("what") or "").strip()
    if section == "해결":
        return str(rec.get("so_what") or "").strip()
    if section == "성과":
        return str(rec.get("now_what") or rec.get("so_what") or "").strip()
    return ""


# ── What–So What–Now What 성찰 엔진 ────────────────────────────────
TASK_TYPES: tuple[str, ...] = (
    "troubleshooting",
    "measurement",
    "assembly",
    "design",
    "embedded_programming",
    "general",
)
REFLECTION_META_TAG = "[성찰메타]"
_REFLECTION_SECTION_TAGS: tuple[str, ...] = (
    "[What]",
    "[So What]",
    "[Now What]",
    "[경험]",
    "[의미]",
    "[적용]",
    "[배경]",
    "[해결]",
    "[성과]",
    REFLECTION_META_TAG,
)
_PROBLEM_RE = re.compile(
    r"켜지지\s*않|안\s*켜|동작하지\s*않|안되|안\s*되|안됨|오류|불량|고장|"
    r"쇼트|실패|원인\s*분석|안\s*나와|안나와|멈췄|오동작|문제\s*가|"
    r"문제가\s|문제점|이상\s*동|이상했"
)
_NO_PROBLEM_RE = re.compile(r"문제\s*없|이상\s*없|정상\s*동작|잘\s*됨|잘됨")
_MEASURE_KW = (
    "오실로", "oscillo", "파형", "멀티미터", "측정", "전압", "전류", "주파수",
    "계측", "메거", "프로브", "파형",
)
_ASSEMBLY_KW = ("납땜", "솔더", "조립", "장착", "배선", "기판", "PCB", "부품 삽입", "브레드")
_DESIGN_KW = ("회로도", "설계", "저항값", "부품 선정", "스키매틱", "시뮬레이션", "정수 선정")
_EMBED_KW = ("아두이노", "arduino", "mcu", "코딩", "펌웨어", "임베디드", "스케치", "디버깅", "마이크로")
_KNOWN_EQUIPMENT = (
    "오실로스코프", "멀티미터", "인두", "납땜기", "전원장치", "함수발생기",
    "아두이노", "PLC", "브레드보드", "PCB", "로직분석기",
)


def _text_has_problem(text: str) -> bool:
    t = text or ""
    if _NO_PROBLEM_RE.search(t):
        return False
    return bool(_PROBLEM_RE.search(t) or ("문제" in t and "문제없" not in t.replace(" ", "")))


def _first_matching_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(k.lower() in low for k in keywords)


def _heuristic_task_type(memo: str, problem_occurred: bool) -> str:
    t = memo or ""
    if problem_occurred:
        return "troubleshooting"
    if _first_matching_keyword(t, _EMBED_KW):
        return "embedded_programming"
    if _first_matching_keyword(t, _DESIGN_KW):
        return "design"
    if _first_matching_keyword(t, _MEASURE_KW):
        return "measurement"
    if _first_matching_keyword(t, _ASSEMBLY_KW):
        return "assembly"
    return "general"


def _photo_confidence_ok(conf: object) -> bool:
    """사진 인식 신뢰도가 질문의 '사실'로 쓸 만큼 높은지."""
    s = str(conf or "").strip().replace("%", "")
    if not s or s in ("—", "-", "없음", "미상"):
        return False
    try:
        return float(s) >= 70.0
    except ValueError:
        return "높" in s


def _equipment_from_inputs(memo: str, detected_tools: list | None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    blob = memo or ""
    blob_l = blob.lower()
    memo_type = _heuristic_task_type(blob, _text_has_problem(blob))

    def add(name: str) -> None:
        n = (name or "").strip()
        if not n or n in seen:
            return
        dummy = {"사진 없음", "이미지 로드 실패", "API 키 미설정", "분석 결과 없음", "이미지 분석 완료"}
        if n in dummy:
            return
        seen.add(n)
        found.append(n)

    for eq in _KNOWN_EQUIPMENT:
        if eq.lower() in blob_l or eq in blob:
            add(eq)
    if "오실로" in blob and "오실로스코프" not in seen:
        add("오실로스코프")

    for d in detected_tools or []:
        if not isinstance(d, dict):
            continue
        name = str(d.get("객체") or d.get("name") or "").strip()
        if not name or not _photo_confidence_ok(d.get("신뢰도")):
            continue
        if name in blob or name.lower() in blob_l:
            add(name)
        elif memo_type == "general":
            # 메모가 빈약할 때만 고신뢰 사진 장비를 보조 근거로 사용
            add(name)
    return found[:8]


def _task_label_from_memo(memo: str, task_type: str) -> str:
    s = re.sub(r"\s+", " ", (memo or "").strip())
    if s:
        return s[:80]
    return {
        "troubleshooting": "문제 확인 작업",
        "measurement": "측정 작업",
        "assembly": "조립 작업",
        "design": "회로 설계 작업",
        "embedded_programming": "임베디드 프로그래밍 작업",
        "general": "전자 실무 작업",
    }.get(task_type, "실습 작업")


def heuristic_practice_analysis(
    memo: str,
    detected_tools: list | None = None,
    ncs_unit: str = "",
) -> dict[str, Any]:
    """API 없이 메모·인식 장비만으로 안정적인 분석 JSON을 만든다."""
    raw = (memo or "").strip()
    problem = _text_has_problem(raw)
    task_type = _heuristic_task_type(raw, problem)
    equipment = _equipment_from_inputs(raw, detected_tools)
    task = _task_label_from_memo(raw, task_type)
    focus = {
        "troubleshooting": "원인 확인 순서와 판단 이유",
        "measurement": "측정값·파형을 정상으로 본 기준",
        "assembly": "작업 품질을 높이기 위해 신경 쓴 점",
        "design": "회로·부품을 선택한 조건",
        "embedded_programming": "의도대로 동작하는지 확인한 방법",
        "general": "결과에 영향을 준 작업 방법이나 판단",
    }.get(task_type, "작업 판단 기준")
    evidence = raw[:180] if raw else "(학생 입력 없음)"
    return {
        "task_type": task_type,
        "problem_occurred": bool(problem),
        "task": task,
        "equipment": equipment,
        "ncs_unit": (ncs_unit or "").strip(),
        "evidence": evidence,
        "reflection_focus": focus,
        "raw_input": raw,
        "image_analysis": [str(x) for x in (equipment or [])],
    }


def parse_reflection_record(text: str) -> dict[str, Any]:
    """저장된 일지 문자열을 신규/레거시 포맷으로 파싱. 레거시 구간을 신규와 동일 의미로 치환하지 않는다."""
    raw = str(text or "")
    meta: dict[str, Any] = {}
    body = raw
    mi = raw.find(REFLECTION_META_TAG)
    if mi >= 0:
        blob = raw[mi + len(REFLECTION_META_TAG) :].strip()
        body = raw[:mi].rstrip()
        try:
            start = blob.find("{")
            end = blob.rfind("}") + 1
            if start >= 0 and end > start:
                obj = json.loads(blob[start:end])
                if isinstance(obj, dict):
                    meta = obj
        except (ValueError, json.JSONDecodeError, TypeError):
            meta = {}

    def _take(tag: str) -> str:
        i = body.find(tag)
        if i < 0:
            return ""
        segment = body[i + len(tag) :]
        ends = [segment.find(t) for t in _REFLECTION_SECTION_TAGS if t != tag]
        ends = [p for p in ends if p >= 0]
        end = min(ends) if ends else len(segment)
        return segment[:end].strip()

    what = _take("[What]") or _take("[경험]")
    so_what = _take("[So What]") or _take("[의미]")
    now_what = _take("[Now What]") or _take("[적용]")
    legacy_bg = _take("[배경]")
    legacy_sol = _take("[해결]")
    legacy_ref = _take("[성과]")
    fmt = "wswnw" if (what or so_what or now_what) else (
        "legacy_bsr" if (legacy_bg or legacy_sol or legacy_ref) else "plain"
    )
    return {
        "format": fmt,
        "what": what,
        "so_what": so_what,
        "now_what": now_what,
        "legacy_background": legacy_bg,
        "legacy_solution": legacy_sol,
        "legacy_reflection": legacy_ref,
        "meta": meta,
        "raw": raw,
    }


def parse_reflection_log(text: str) -> dict[str, Any]:
    """``parse_reflection_record``의 공개 별칭."""
    return parse_reflection_record(text)


def strip_reflection_meta(text: str) -> str:
    """``[성찰메타]`` JSON 블록을 제거한 문자열."""
    raw = str(text or "")
    i = raw.find(REFLECTION_META_TAG)
    if i < 0:
        return raw.strip()
    return raw[:i].rstrip()


def get_reflection_meta(text: str) -> dict[str, Any]:
    """연구·내부 추적용 메타만 반환. UI/LLM 본문에는 쓰지 않는다."""
    rec = parse_reflection_record(text)
    return dict(rec.get("meta") or {})


def get_reflection_body(text: str) -> str:
    """사용자·포트폴리오·세특·교사의견에 넣을 성찰 본문. 메타 JSON 제외.

    신규는 What/So What/Now What, 레거시는 배경/해결/성과를 그대로 유지한다.
    """
    rec = parse_reflection_record(text)
    if rec["format"] == "wswnw":
        parts: list[str] = []
        if rec.get("what"):
            parts.append(f"[What] {rec['what']}")
        if rec.get("so_what"):
            parts.append(f"[So What] {rec['so_what']}")
        if rec.get("now_what"):
            parts.append(f"[Now What] {rec['now_what']}")
        chk = re.search(r"\[체크리스트:[^\]]*\]", str(text or ""))
        if chk:
            parts.append(chk.group(0))
        return "\n".join(parts).strip()
    if rec["format"] == "legacy_bsr":
        parts = []
        if rec.get("legacy_background"):
            parts.append(f"[배경] {rec['legacy_background']}")
        if rec.get("legacy_solution"):
            parts.append(f"[해결] {rec['legacy_solution']}")
        if rec.get("legacy_reflection"):
            parts.append(f"[성과] {rec['legacy_reflection']}")
        chk = re.search(r"\[체크리스트:[^\]]*\]", str(text or ""))
        if chk:
            parts.append(chk.group(0))
        return "\n".join(parts).strip() if parts else strip_reflection_meta(text)
    return strip_reflection_meta(text)


def reflection_display_sections(text: str) -> list[tuple[str, str, str]]:
    """UI용 (짧은 제목, 설명, 본문) 목록. 레거시 일지는 옛 라벨을 유지한다."""
    rec = parse_reflection_record(text)
    if rec["format"] == "legacy_bsr":
        return [
            ("배경 · 상황", "이전 형식", rec["legacy_background"]),
            ("해결 · 과정", "이전 형식", rec["legacy_solution"]),
            ("성과 · 성찰", "이전 형식", rec["legacy_reflection"]),
        ]
    if rec["format"] == "wswnw":
        return [
            ("What — 실무 경험", "실무 경험", rec["what"]),
            ("So What — 판단 및 성찰", "판단 및 성찰", rec["so_what"]),
            ("Now What — 향후 적용", "향후 적용", rec["now_what"]),
        ]
    body = (text or "").strip()
    return [("실습 기록", "", body)] if body else []


def build_reflection_string(
    what: str,
    so_what: str,
    now_what: str,
    *,
    meta: dict | None = None,
    checked_items: list[str] | None = None,
) -> str:
    parts = [
        f"[What] {(what or '').strip()}",
        f"[So What] {(so_what or '').strip()}",
        f"[Now What] {(now_what or '').strip()}",
    ]
    if checked_items:
        parts.append(f"[체크리스트: {'; '.join(checked_items)}]")
    if meta:
        compact = {
            k: meta.get(k)
            for k in (
                "task_type",
                "problem_occurred",
                "task",
                "equipment",
                "ncs_unit",
                "turn1_question",
                "turn1_answer",
                "turn2_question",
                "turn2_answer",
                "raw_input",
                "reflection_focus",
                "evidence",
                "image_analysis",
            )
            if meta.get(k) not in (None, "", [])
        }
        try:
            parts.append(REFLECTION_META_TAG + json.dumps(compact, ensure_ascii=False))
        except (TypeError, ValueError):
            pass
    return "\n".join(parts)


def _parse_analysis_json(raw: str) -> dict[str, Any] | None:
    if not (raw or "").strip():
        return None
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        start = t.index("{")
        end = t.rindex("}") + 1
        obj = json.loads(t[start:end])
    except (ValueError, json.JSONDecodeError, KeyError):
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _grounded_equipment(candidates: list, memo: str, detected: list | None) -> list[str]:
    allowed = set(_equipment_from_inputs(memo, detected))
    blob = (memo or "").lower()
    out: list[str] = []
    for c in candidates or []:
        name = str(c or "").strip()
        if not name:
            continue
        if name in allowed or name.lower() in blob or name in (memo or ""):
            if name not in out:
                out.append(name)
    return out[:8]


def merge_practice_analysis(heuristic: dict[str, Any], model: dict | None) -> dict[str, Any]:
    """우선순위: 학생 원문 > 고신뢰 사진 장비 > 휴리스틱 > LLM.

    LLM이 troubleshooting을 넣어도 원문에 문제 서술이 없으면 채택하지 않는다.
    LLM 장비는 원문 또는 이미 허용된 장비 목록에 있을 때만 남긴다.
    """
    out = dict(heuristic)
    if not model:
        return out
    memo = str(heuristic.get("raw_input") or "")
    detected = heuristic.get("image_analysis") or heuristic.get("equipment")
    # 문제 발생 여부는 학생 텍스트만 따른다.
    out["problem_occurred"] = bool(heuristic.get("problem_occurred"))
    if out["problem_occurred"]:
        out["task_type"] = "troubleshooting"
    else:
        m_type = str(model.get("task_type") or "").strip()
        h_type = str(heuristic.get("task_type") or "general")
        if m_type == "troubleshooting":
            pass
        elif h_type != "general":
            out["task_type"] = h_type
        else:
            # 원문에 유형 키워드가 없으면 LLM이 assembly/measurement 등으로 올리지 않는다.
            out["task_type"] = "general"
    eq = _grounded_equipment(model.get("equipment") or [], memo, detected)
    if eq:
        # 휴리스틱(원문+고신뢰 사진) 목록을 우선하고, 그 안에서만 보강
        base = list(heuristic.get("equipment") or [])
        merged_eq: list[str] = []
        for n in base + eq:
            if n and n not in merged_eq:
                merged_eq.append(n)
        out["equipment"] = merged_eq[:8]
    task = str(model.get("task") or "").strip()
    if task and (task[:12] in memo or any(w and w in memo for w in re.findall(r"[가-힣A-Za-z0-9]{2,}", task)[:6])):
        out["task"] = task[:120]
    ncs = str(model.get("ncs_unit") or "").strip()
    if ncs and not out.get("ncs_unit"):
        out["ncs_unit"] = ncs
    return out


def analyze_practice_experience(
    memo: str,
    detected_tools: list | None = None,
    ncs_unit: str = "",
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """사진·메모에서 구조화 JSON을 만든다. 실패 시 휴리스틱만 반환."""
    heur = heuristic_practice_analysis(memo, detected_tools, ncs_unit)
    key = resolve_google_api_key(api_key)
    if not key or not (memo or "").strip():
        return heur
    tools = _detected_tools_to_str(detected_tools)
    prompt = f"""공업고 전자 실습 일지를 분석한다. 학생 입력에 없는 사실·장비·문제를 만들지 마라.
출력은 JSON 하나만. 키:
task_type (troubleshooting|measurement|assembly|design|embedded_programming|general),
problem_occurred (boolean, 메모에 오류·불량·미동작이 명시된 경우만 true),
task, equipment (배열, 입력/사진에 있는 것만), ncs_unit, evidence, reflection_focus.

[학생 메모]
{(memo or '')[:2000]}

[사진 인식]
{tools}

[NCS]
{ncs_unit or '미정'}
"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        raw = gemini_generate_text(
            genai,
            prompt,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 1024,
                "response_mime_type": "application/json",
            },
        )
        parsed = _parse_analysis_json(raw or "")
        if parsed is None:
            raw2 = gemini_generate_text(
                genai,
                prompt,
                generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            )
            parsed = _parse_analysis_json(raw2 or "")
        return merge_practice_analysis(heur, parsed)
    except Exception:
        return heur


def fallback_so_what_question(analysis: dict[str, Any]) -> str:
    memo = str(analysis.get("raw_input") or "").strip()
    task = (analysis.get("task") or "").strip() or memo or "오늘 작업"
    eq = analysis.get("equipment") or []
    eq_s = eq[0] if eq else ""
    t = analysis.get("task_type") or "general"
    vague = t == "general" or len(memo) < 18
    if t == "troubleshooting" and analysis.get("problem_occurred"):
        target = eq_s or "연결·동작이 달랐던 부분"
        return (
            f"{target}을 확인할 때 어떤 부분을 먼저 확인했고, "
            "왜 그 부분부터 확인했나요?"
        )
    if t == "measurement":
        if eq_s:
            return (
                f"{eq_s}로 측정하면서 정상 여부를 확인하기 위해 "
                "어떤 값을 중점적으로 확인했나요?"
            )
        return "출력 파형을 측정하면서 정상 여부를 확인하기 위해 어떤 값을 중점적으로 확인했나요?"
    if t == "assembly":
        named: list[str] = []
        if "저항" in memo:
            named.append("저항")
        if re.search(r"LED|led|엘이디", memo, re.I):
            named.append("LED")
        subject = "과 ".join(named) if named else (eq_s or "부품")
        return f"{subject}를 납땜할 때 접합 상태가 적절한지 어떤 부분을 확인했나요?"
    if t == "design":
        return "저항값을 변경할 때 어떤 조건을 기준으로 새로운 값을 선택했나요?"
    if t == "embedded_programming":
        return "프로그램이 의도한 대로 동작하는지 어떤 방법으로 직접 확인했나요?"
    if vague:
        return "오늘 수행한 작업에서 결과를 확인하기 위해 실제로 어떤 과정을 거쳤나요?"
    return f"오늘 수행한 {task}에서 결과를 확인하기 위해 실제로 어떤 과정을 거쳤나요?"


def fallback_now_what_question(analysis: dict[str, Any], answer1: str) -> str:
    a = re.sub(r"\s+", " ", (answer1 or "").strip())
    if len(a) < 12:
        return "다음에 비슷한 작업을 한다면 이번보다 더 잘하기 위해 한 가지 바꾸고 싶은 점은 무엇인가요?"
    snip = a[:36].rstrip() + ("…" if len(a) > 36 else "")
    t = analysis.get("task_type") or "general"
    if t == "troubleshooting":
        return (
            f"‘{snip}’라는 확인 방법을 다음 회로 작업에서 더 빨리 쓰기 위해 "
            "작업 중 어떤 점을 먼저 점검하고 싶나요?"
        )
    if t == "measurement":
        return (
            f"‘{snip}’을 다음 측정에서 더 정확하게 적용하기 위해 "
            "어떤 점을 보완하고 싶나요?"
        )
    if t == "assembly":
        return (
            f"‘{snip}’을 다음 조립에서 더 안정적으로 적용하려면 "
            "작업 과정에서 어떤 점을 확인하고 싶나요?"
        )
    if t == "design":
        return (
            f"‘{snip}’이라는 판단을 다음 설계에 적용한다면 "
            "값 선정이나 검증을 어떤 순서로 하고 싶나요?"
        )
    if t == "embedded_programming":
        return (
            f"‘{snip}’으로 확인했습니다. 다음 코딩·디버깅에서 "
            "어떤 테스트를 더 일찍 넣고 싶나요?"
        )
    return (
        f"‘{snip}’을 다음 실습에서 더 효과적으로 적용하기 위해 "
        "어떤 점을 보완하거나 확인하고 싶나요?"
    )


_INVENTED_FACT_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("오실로스코프", ("오실로", "oscillo", "파형")),
    ("멀티미터", ("멀티미터", "테스터", "multimeter")),
    ("납땜", ("납땜", "솔더", "solder")),
    ("아두이노", ("아두이노", "arduino", "mcu")),
    ("PCB", ("pcb", "스키매틱")),
    ("브레드보드", ("브레드", "breadboard")),
    ("인두", ("인두", "납땜기")),
)


def _question_invents_unknown_equipment(question: str, analysis: dict[str, Any]) -> bool:
    q = question or ""
    memo = str(analysis.get("raw_input") or "")
    ans = str(analysis.get("_turn1_answer") or "")
    known = " ".join(str(x) for x in (analysis.get("equipment") or [])) + " " + memo + " " + ans
    if "기판" in known:
        known += " PCB pcb"
    if "테스터" in known:
        known += " 멀티미터"
    known_l = known.lower()
    for eq in _KNOWN_EQUIPMENT:
        if eq in q and eq not in known and eq.lower() not in known_l:
            return True
    for _label, markers in _INVENTED_FACT_MARKERS:
        if any(m.lower() in q.lower() or m in q for m in markers):
            if not any(m.lower() in known_l or m in known for m in markers):
                return True
    if not analysis.get("problem_occurred"):
        if re.search(r"켜지지\s*않|고장|오류 원인|불량의 원인|오동작", q):
            return True
    for qty in ("전압", "전류", "주파수", "진폭"):
        if qty in q and qty not in known:
            return True
    for jargon in _UNGROUNDED_JARGON:
        if jargon.lower() in q.lower() and jargon.lower() not in known_l and jargon not in known:
            return True
    return False


_UNGROUNDED_JARGON = (
    "Time/Div", "Volt/Div", "V/div", "T/div", "트리거 레벨", "프로브 보정",
    "로직분석기", "스펙트럼", "FFT", "진폭", "왜곡", "오버슈트", "듀티비",
    "안정성", "노이즈", "오차", "불량률", "신뢰성", "리플",
)
_TURN2_FORBIDDEN_UNLESS_SAID = (
    "안정성", "왜곡", "진폭", "Time/Div", "Volt/Div", "V/div", "노이즈",
    "오차", "불량률", "오버슈트", "듀티", "트리거", "프로브 보정",
    "신뢰성", "리플", "정확도 저하",
)

_ABSTRACT_SO_WHAT_RE = re.compile(
    r"무엇을 배웠|어떻게 느꼈|무엇이 중요했|느낀 점은|배운 점은\s*무엇|"
    r"어떤 점을?\s*가장\s*중요|중요하게\s*생각|무엇이 중요하다고 생각"
)
_HYPOTHETICAL_SO_WHAT_RE = re.compile(
    r"판단할 수 있|확인할 수 있|알 수 있|일반적으로 어떤 방법|"
    r"어떤 방법이 있|어떻게 하면 되|이론상"
)
_FUTURE_NOW_WHAT_RE = re.compile(
    r"다음|향후|다음번|적용|보완|예방|개선|바꾸고|확인하고 싶|하고 싶"
)
_CONTENT_STOP = {
    "그리고", "그래서", "그러나", "이번", "오늘", "실습", "작업", "확인", "했습니다",
    "했어요", "생각", "것", "수", "때", "더", "가장", "부분", "위해", "하는",
}


def _content_tokens(text: str) -> set[str]:
    toks = set(re.findall(r"[가-힣A-Za-z0-9]{2,}", text or ""))
    return {t for t in toks if t not in _CONTENT_STOP and len(t) >= 2}


def _so_what_question_ok(q: str, analysis: dict[str, Any]) -> bool:
    if not q or ("?" not in q and "？" not in q):
        return False
    if _ABSTRACT_SO_WHAT_RE.search(q):
        return False
    if _HYPOTHETICAL_SO_WHAT_RE.search(q):
        return False
    if _question_invents_unknown_equipment(q, analysis):
        return False
    if not analysis.get("problem_occurred"):
        if re.search(r"켜지지\s*않|고장|오류 원인|불량의 원인", q):
            return False
    return True


def _now_what_adds_unsaid_concepts(q: str, answer1: str, memo: str) -> bool:
    known = f"{answer1 or ''} {memo or ''}"
    known_l = known.lower()
    for concept in _TURN2_FORBIDDEN_UNLESS_SAID:
        if concept.lower() in (q or "").lower() and concept.lower() not in known_l and concept not in known:
            return True
    if re.search(r"불량", q or "") and "불량" not in known:
        return True
    return False


def _now_what_question_ok(q: str, analysis: dict[str, Any], answer1: str) -> bool:
    if not q or ("?" not in q and "？" not in q):
        return False
    if _question_invents_unknown_equipment(q, {**analysis, "_turn1_answer": answer1}):
        return False
    if not _FUTURE_NOW_WHAT_RE.search(q):
        return False
    if _now_what_adds_unsaid_concepts(q, answer1, str(analysis.get("raw_input") or "")):
        return False
    a = (answer1 or "").strip()
    if len(a) >= 12:
        tokens = [t for t in _content_tokens(a) if len(t) >= 2]
        # 긴 토큰을 우선해 최소 1개는 질문에 등장해야 한다
        tokens.sort(key=len, reverse=True)
        if tokens and not any(t in q for t in tokens[:8]):
            return False
    return True


def generate_so_what_question(analysis: dict[str, Any], *, api_key: str | None = None) -> str:
    """Turn 1: So What? — 판단·기준·방법을 하나만 묻는다."""
    fb = fallback_so_what_question(analysis)
    key = resolve_google_api_key(api_key)
    if not key:
        return fb
    payload = json.dumps(
        {
            "task_type": analysis.get("task_type"),
            "problem_occurred": analysis.get("problem_occurred"),
            "task": analysis.get("task"),
            "equipment": analysis.get("equipment"),
            "raw_input": analysis.get("raw_input"),
        },
        ensure_ascii=False,
    )
    focus_hint = {
        "troubleshooting": "학생이 실제로 먼저 확인한 부분과 그 이유를 회상하도록 묻는다. 예: LED가 켜지지 않았을 때 어떤 부분을 먼저 확인했고, 왜 그 부분부터 확인했나요?",
        "measurement": "측정하면서 실제로 어떤 값을 확인했는지 묻는다. 예: 출력 파형을 측정하면서 정상 여부를 확인하기 위해 어떤 값을 중점적으로 확인했나요?",
        "assembly": "납땜·조립 중 실제로 살펴본 부분을 묻는다. 예: 저항과 LED를 납땜할 때 접합 상태가 적절한지 어떤 부분을 확인했나요? '어떻게 판단할 수 있나요'처럼 일반 지식 질문은 금지.",
        "design": "값을 바꾼 실제 이유와 선택 기준을 묻는다. 예: 저항값을 변경할 때 어떤 조건을 기준으로 새로운 값을 선택했나요?",
        "embedded_programming": "프로그램이 의도대로 동작하는지 직접 확인한 방법을 묻는다.",
        "general": "오늘 수행한 작업에서 결과를 확인하기 위해 실제로 거친 과정을 묻는다. 장비·납땜·고장을 만들지 마라.",
    }.get(str(analysis.get("task_type") or "general"), "")
    prompt = f"""공업고 전자과 실습 교사다. 아래 JSON은 학생이 실제로 적은 내용만 담는다.
So What? 질문은 이론 시험이 아니라, 학생이 실제로 한 행동·판단·기준을 회상하게 하는 한 문장이다.
구조: 학생의 실제 작업명 + 수행한 행동 + 판단/기준 한 가지.
우선 표현: 무엇을 확인했나요, 어떤 부분을 주의해서 작업했나요, 어떤 기준으로 상태를 확인했나요, 왜 그 부분을 중요하게 봤나요.
금지 표현: 어떻게 판단할 수 있나요, 일반적으로 어떤 방법이 있나요, 무엇이 중요하다고 생각하나요, 무엇을 배웠나요, 어떻게 느꼈나요.
이번 작업 유형 힌트: {focus_hint}
JSON에 없는 장비·고장·작업을 가정하지 마라. problem_occurred가 false이면 고장·오류 원인을 묻지 마라.
고등학생이 이해할 한 문장으로 물음표로 끝내라.

분석 JSON:
{payload}
"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        raw = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 512},
        )
        line = " ".join(x.strip() for x in (raw or "").splitlines() if x.strip())
        if "?" in line or "？" in line:
            q_idx = max(line.rfind("?"), line.rfind("？"))
            line = line[: q_idx + 1].strip()
        if _so_what_question_ok(line, analysis):
            return line
    except Exception:
        pass
    return fb


def generate_now_what_question(
    analysis: dict[str, Any],
    turn1_question: str,
    turn1_answer: str,
    *,
    api_key: str | None = None,
) -> str:
    """Turn 2: Now What? — Turn 1 답변의 핵심을 다음 실습으로 옮긴다. 실패 시 1회 재시도."""
    fb = fallback_now_what_question(analysis, turn1_answer)
    key = resolve_google_api_key(api_key)
    a1 = (turn1_answer or "").strip()
    core = re.sub(r"\s+", " ", a1)[:80]
    if not key:
        return fb

    def _ask(extra: str) -> str | None:
        payload = json.dumps(
            {
                "task_type": analysis.get("task_type"),
                "task": analysis.get("task"),
                "equipment": analysis.get("equipment"),
                "raw_input": analysis.get("raw_input"),
                "turn1_question": turn1_question,
                "turn1_answer": a1,
            },
            ensure_ascii=False,
        )
        prompt = f"""공업고 전자과 실습 교사다. Now What? 질문 1개만 출력한다.
학생 Turn 1 답변에 실제로 나온 핵심(방법·기준·비교한 값)만 질문에 반영하고,
다음 실습에서 그 방법을 어떻게 적용·보완할지 묻는다.
답변에 없는 개념을 새로 붙이지 마라. 금지 예: 안정성, 왜곡, 진폭, Time/Div, 노이즈, 오차, 불량, 새 장비.
금지: '다음에 무엇을 개선하고 싶나요?' 고정문구, 형식적 칭찬.
한 문장, 물음표로 끝.
{extra}

맥락 JSON:
{payload}
"""
        import google.generativeai as genai

        genai.configure(api_key=key)
        raw = gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": 0.2, "max_output_tokens": 512},
        )
        line = " ".join(x.strip() for x in (raw or "").splitlines() if x.strip())
        if "?" in line or "？" in line:
            q_idx = max(line.rfind("?"), line.rfind("？"))
            line = line[: q_idx + 1].strip()
        return line or None

    try:
        q = _ask("")
        if q and _now_what_question_ok(q, analysis, a1):
            return q
        retry = _ask(f"반드시 학생 답변의 이 핵심을 질문에 포함하라: {core}")
        if retry and _now_what_question_ok(retry, analysis, a1):
            return retry
    except Exception:
        pass
    return fb


def generate_reflection_draft(
    analysis: dict[str, Any],
    turn1_answer: str,
    turn2_answer: str,
    *,
    api_key: str | None = None,
) -> dict[str, str]:
    """What / So What / Now What 초안. 입력에 없는 사실은 추가하지 않는다."""
    empty = {"what": "", "so_what": "", "now_what": ""}
    memo = str(analysis.get("raw_input") or "").strip()
    a1 = (turn1_answer or "").strip()
    a2 = (turn2_answer or "").strip()
    if not (memo or a1 or a2):
        return dict(empty)
    key = resolve_google_api_key(api_key)
    fallback = {
        "what": memo or str(analysis.get("task") or ""),
        "so_what": a1,
        "now_what": a2,
    }
    if not key:
        return fallback
    payload = json.dumps(
        {
            "raw_input": memo,
            "equipment": analysis.get("equipment"),
            "turn1_answer": a1,
            "turn2_answer": a2,
        },
        ensure_ascii=False,
    )
    prompt = f"""공업고 전자 실습 일지를 What–So What–Now What 구조로 정리한다.
JSON만 출력. 키: what, so_what, now_what (한국어 순수 텍스트).
- what: 학생 메모에 적힌 작업과 상황만 객관적으로.
- so_what: Turn 1 답변을 중심으로 판단·기준·이유.
- now_what: Turn 2 답변을 중심으로 다음 적용·보완.
학생 메모와 두 답변에 없는 장비·고장·성과·학습 내용을 만들지 마라.
NCS 용어는 경험을 왜곡하지 않는 범위에서만 다듬어라.

자료:
{payload}
"""
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        raw = gemini_generate_text(
            genai,
            prompt,
            generation_config={
                "temperature": 0.2,
                "max_output_tokens": 2048,
                "response_mime_type": "application/json",
            },
        )
        obj = _parse_analysis_json(raw or "")
        if not obj:
            raw = gemini_generate_text(
                genai,
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 2048},
            )
            obj = _parse_analysis_json(raw or "") or {}
        else:
            obj = obj or {}
        out = {
            "what": str(obj.get("what") or "").strip(),
            "so_what": str(obj.get("so_what") or "").strip(),
            "now_what": str(obj.get("now_what") or "").strip(),
        }
        if not (out["what"] or out["so_what"] or out["now_what"]):
            return fallback
        return out
    except Exception:
        return fallback



def _strip_magic_draft_markdown(s: str) -> str:
    """Magic Draft·JSON 값에서 마크다운 강조 기호를 제거."""
    if not s:
        return s
    out = s
    out = re.sub(r"\*\*([^*]+)\*\*", r"\1", out)
    out = re.sub(r"\*([^*]+)\*", r"\1", out)
    out = re.sub(r"__([^_]+)__", r"\1", out)
    return out


def _parse_magic_draft_json_or_tags(raw: str) -> dict[str, str]:
    """모델 출력(JSON 우선, 실패 시 [배경] 태그 문자열)을 dict로 정규화."""
    empty = {"background": "", "solution": "", "reflection": ""}
    if not (raw or "").strip():
        return dict(empty)
    t = raw.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        start = t.index("{")
        end = t.rindex("}") + 1
        obj = json.loads(t[start:end])
        return {
            "background": _strip_magic_draft_markdown(str(obj.get("background", "") or "")),
            "solution": _strip_magic_draft_markdown(str(obj.get("solution", "") or "")),
            "reflection": _strip_magic_draft_markdown(str(obj.get("reflection", "") or "")),
        }
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    if "[배경]" in t or "[해결]" in t or "[성과]" in t:
        return {
            "background": _strip_magic_draft_markdown(extract_bsr_section(t, "배경")),
            "solution": _strip_magic_draft_markdown(extract_bsr_section(t, "해결")),
            "reflection": _strip_magic_draft_markdown(extract_bsr_section(t, "성과")),
        }
    return dict(empty)


def generate_bsr_draft_from_keywords(
    raw_text: str,
    detected_tools: list,
    api_key: str,
) -> dict[str, str]:
    """
    짧은 메모·키워드와 사진 인식으로 What–So What–Now What 초안을 생성한다.
    구 호출부를 위해 background/solution/reflection 키도 함께 반환한다.
    """
    key = (api_key or "").strip() or resolve_google_api_key()
    empty = {"background": "", "solution": "", "reflection": "", "what": "", "so_what": "", "now_what": ""}
    if not key or not (raw_text or "").strip():
        return dict(empty)

    analysis = analyze_practice_experience(raw_text, detected_tools or [], "", api_key=key)
    draft = generate_reflection_draft(analysis, "", "", api_key=key)
    if not (draft.get("what") or draft.get("so_what") or draft.get("now_what")):
        raise RuntimeError(
            "성찰 JSON 파싱 후 What·So What·Now What이 모두 비었습니다."
        )
    return {
        "what": draft.get("what", ""),
        "so_what": draft.get("so_what", ""),
        "now_what": draft.get("now_what", ""),
        "background": draft.get("what", ""),
        "solution": draft.get("so_what", ""),
        "reflection": draft.get("now_what", ""),
    }


def _parse_evidence_score_0_100(text: str | None) -> float | None:
    """모델 출력에서 0~100 점수 파싱."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```\s*$", "", t)
    try:
        start = t.index("{")
        end = t.rindex("}") + 1
        obj = json.loads(t[start:end])
        s = obj.get("score")
        if s is not None:
            v = float(s)
            return min(100.0, max(0.0, v))
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    m = re.search(r"(?:score|점수)\s*[:=]\s*(\d{1,3})", t, re.I)
    if m:
        return min(100.0, max(0.0, float(m.group(1))))
    m2 = re.search(r"\b(\d{1,3})\b", t)
    if m2:
        v = int(m2.group(1))
        if 0 <= v <= 100:
            return float(v)
    return None


def check_evidence_validity(
    image_file,
    content: str,
    *,
    api_key: str | None = None,
) -> float:
    """
    실습 사진(들)과 [배경] 글의 증거 적합성을 0~100으로 추정.
    image_file은 단일 UploadedFile 또는 List[UploadedFile] 모두 허용.
    여러 장이 들어오면 모든 사진을 같은 실습 컨텍스트로 묶어 한 번에 평가한다.
    API 실패·이미지 오류 시 중립값(75)을 반환해 UI가 과도하게 경고하지 않게 한다.
    """
    key = resolve_google_api_key(api_key)
    bg = extract_background_section(content)
    if not key or not bg.strip():
        return 75.0

    files = image_file if isinstance(image_file, (list, tuple)) else [image_file]
    files = [f for f in files if f is not None]
    if not files:
        return 75.0

    try:
        pil_imgs, _fp = uploaded_files_to_gemini_pil_images(files)
        if not pil_imgs:
            return 75.0
    except Exception:
        return 75.0

    photo_phrase = (
        f"**사진 {len(pil_imgs)}장(같은 실습 시간에 찍은 사진들)**"
        if len(pil_imgs) > 1
        else "**사진 한 장**"
    )
    multi_extra = (
        "\n사진이 여러 장이라면, 모두를 **같은 실습 상황의 증거**로 보고 종합해서 판단하라."
        " 한 장이라도 본문과 잘 맞으면 점수를 너무 박하게 주지 말 것."
        if len(pil_imgs) > 1
        else ""
    )
    prompt = f"""당신은 공업고 전기·전자과 실습 평가를 돕는 조교이다.
학생이 제출한 {photo_phrase}과 **실습 일지 본문**이 서로 **적절한 증거 관계**인지 평가하라.

본문에 서술된 활동·장비·상황이 사진에 보이는 내용과 논리적으로 맞는가?
(예: 본문은 PLC 실습인데 사진만 납땜이면 낮은 점수){multi_extra}

[학생 실습 기록]
{bg[:6000]}

출력 규칙: **JSON 한 줄만** 출력한다.
형식: {{"score": 정수(0~100), "reason": "한 줄 한국어 이유"}}
score 기준: 80~100 매우 일치, 50~79 부분 일치, 0~49 사진이 본문 증거로 부적절"""

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        gc = {"temperature": 0.15, "max_output_tokens": 256}
        safety = gemini_safety_settings_block_none()
        gen_kwargs: dict = {"generation_config": gc}
        if safety:
            gen_kwargs["safety_settings"] = safety
        raw = ""
        used_model = ""
        for name in resolved_gemini_model_candidates(genai):
            try:
                model = get_gemini_model(genai, name)
                if model is None:
                    if name == GEMINI_PRIMARY_MODEL:
                        _log.warning("Gemini evidence-score primary unavailable: %s", name)
                    continue
                pil_rgb = [p.convert("RGB").copy() for p in pil_imgs]
                for p in pil_rgb:
                    try:
                        p.load()
                    except Exception:
                        pass
                payload = [prompt, *pil_rgb]
                response = model.generate_content(payload, **gen_kwargs)
                raw = extract_generate_content_text(response)
                if raw:
                    used_model = name
                    _note_gemini_model(name, reason="evidence-score")
                    break
            except Exception as e:
                if name == GEMINI_PRIMARY_MODEL:
                    _log.warning("Gemini evidence-score primary failed: %s", e)
                continue
        parsed = _parse_evidence_score_0_100(raw)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    return 75.0


def _compress_bsr_for_school_record_summary(bsr: str, *, max_per_section: int = 180) -> str:
    """세특·교사 의견용으로 성찰 본문만 짧게 압축. 메타 JSON 제외."""
    rec = parse_reflection_record(bsr)
    if rec["format"] == "wswnw":
        pairs = (("What", rec.get("what")), ("So What", rec.get("so_what")), ("Now What", rec.get("now_what")))
    elif rec["format"] == "legacy_bsr":
        pairs = (
            ("배경", rec.get("legacy_background")),
            ("해결", rec.get("legacy_solution")),
            ("성과", rec.get("legacy_reflection")),
        )
    else:
        pairs = ()
    parts: list[str] = []
    for label, t in pairs:
        body = str(t or "").strip()
        if not body:
            continue
        one = re.sub(r"\s+", " ", body)
        if len(one) > max_per_section:
            one = one[: max_per_section - 1] + "…"
        parts.append(f"{label}:{one}")
    if parts:
        return " | ".join(parts)
    flat = re.sub(r"\s+", " ", get_reflection_body(bsr))
    if len(flat) > 420:
        return flat[:419] + "…"
    return flat


def _select_logs_for_school_record_summary(logs: list[dict], *, max_entries: int = 32) -> list[dict]:
    """일지가 많을 때 최근·능력단위별 대표 샘플만 선택."""
    if len(logs) <= max_entries:
        return list(logs)

    def _sort_key(r: dict) -> tuple[str, int]:
        d = str(r.get("date") or "").strip()
        lid = int(r.get("id") or 0) if str(r.get("id") or "").isdigit() else 0
        return (d, lid)

    sorted_logs = sorted(logs, key=_sort_key)
    recent = sorted_logs[-12:]
    unit_counts: dict[str, int] = {}
    for r in sorted_logs:
        u = str(r.get("ncs_unit") or "").strip() or "(미분류)"
        unit_counts[u] = unit_counts.get(u, 0) + 1
    top_units = sorted(unit_counts.keys(), key=lambda u: (-unit_counts[u], u))[:6]

    picked: list[dict] = []
    seen_ids: set[int | str] = set()
    for u in top_units:
        unit_logs = [r for r in sorted_logs if (str(r.get("ncs_unit") or "").strip() or "(미분류)") == u]
        for r in unit_logs[-2:]:
            rid = r.get("id", id(r))
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            picked.append(r)
    for r in recent:
        rid = r.get("id", id(r))
        if rid in seen_ids:
            continue
        seen_ids.add(rid)
        picked.append(r)
        if len(picked) >= max_entries:
            break
    picked.sort(key=_sort_key)
    return picked[:max_entries]


def summarize_logs_for_school_record(
    logs: list[dict],
    *,
    max_chars: int = 10000,
) -> tuple[str, dict[str, Any]]:
    """
    AI 전달용 실습 일지 요약본과 통계 메타데이터를 반환한다.

    반환: (요약 텍스트, {"total_logs", "sampled_logs", "unit_stats", ...})
    """
    try:
        from constants import NCS_DB
    except ImportError:
        NCS_DB = {}

    if not logs:
        return "", {"total_logs": 0, "sampled_logs": 0, "unit_stats": []}

    unit_counts: dict[str, int] = {}
    for r in logs:
        u = str(r.get("ncs_unit") or "").strip() or "(미분류)"
        unit_counts[u] = unit_counts.get(u, 0) + 1

    unit_stats: list[dict[str, Any]] = []
    for u, cnt in sorted(unit_counts.items(), key=lambda x: (-x[1], x[0])):
        meta = NCS_DB.get(u, {}) if u != "(미분류)" else {}
        unit_stats.append(
            {
                "unit": u,
                "count": cnt,
                "code": meta.get("code", ""),
                "keywords": (meta.get("keywords") or [])[:8],
            }
        )

    sampled = _select_logs_for_school_record_summary(logs)
    lines: list[str] = []
    lines.append("[능력단위별 실습 빈도]")
    for st in unit_stats[:10]:
        code = st.get("code") or "코드 미상"
        kws = ", ".join(st.get("keywords") or []) or "키워드 없음"
        lines.append(f"- {st['unit']} ({code}): {st['count']}회 · 키워드: {kws}")

    lines.append("")
    lines.append(f"[샘플 실습 일지 {len(sampled)}건 / 전체 {len(logs)}건]")
    for i, r in enumerate(sampled, 1):
        date = str(r.get("date") or "")
        unit = str(r.get("ncs_unit") or "").strip() or "(미분류)"
        bsr = (r.get("bsr") or "").strip()
        ratio = r.get("ncs_term_ratio")
        ratio_s = f" · NCS용어비율:{ratio:.0f}%" if isinstance(ratio, (int, float)) else ""
        compact = _compress_bsr_for_school_record_summary(bsr)
        lines.append(f"--- {i}. {date} | {unit}{ratio_s} ---")
        lines.append(compact if compact else "(본문 없음)")

    corpus = "\n".join(lines).strip()
    if len(corpus) > max_chars:
        corpus = corpus[: max_chars - 20] + "\n…[요약 길이 제한]"

    meta = {
        "total_logs": len(logs),
        "sampled_logs": len(sampled),
        "unit_stats": unit_stats,
        "top_unit": unit_stats[0]["unit"] if unit_stats else "",
    }
    return corpus, meta


def generate_seuteuk_from_bsr_logs(
    logs: list[dict],
    student_label: str,
    *,
    api_key: str | None = None,
) -> str | None:
    """
    BSR 로그를 요약·샘플링해 학교생활기록부용 세특(세부능력 및 특기사항) 서술 초안 생성.
    실패 시 None.
    """
    if not logs:
        return None
    corpus, meta = summarize_logs_for_school_record(logs)
    if not corpus.strip():
        return None
    top_unit = str(meta.get("top_unit") or "").strip() or "해당 NCS 능력단위"

    prompt = f"""당신은 고등학교 전기·전자과 담임 및 현장교사를 돕는 기술사이다.
아래는 한 학생의 NCS 기반 실습 성찰 일지를 토큰 절약을 위해 요약·샘플링한 자료이다.
이 자료를 바탕으로 **학교생활기록부의 「세부능력 및 특기사항」(세특)**에 들어갈 서술형 문단을 작성하라.

학생: {student_label}
전체 실습 횟수: {meta.get("total_logs", 0)}회
분석에 사용한 샘플: {meta.get("sampled_logs", 0)}건
일지 분석상 가장 많이 수행한 NCS 능력단위: {top_unit}

[실습 일지 요약]
{corpus}

작성 지침:
- 학생 일지 데이터를 분석하여 **가장 많이 수행한 NCS 능력단위**를 파악하고, 생성되는 세특 내용의 **첫 부분이나 핵심 문장 안에 반드시 대괄호**를 사용하여 **[{top_unit}]** 형태로 해당 능력단위명을 표기하라.
- 단순히 '무엇을 했다'는 활동 나열이 아니라, **오류·이상 징후·시운전 문제를 해결하는 과정**에서 드러난 **기술적 성장**과 **메타인지적 태도**(원인 가설, 점검 순서, 측정·대조, 개선)를 중심으로 서술한다.
- 전기·전자 실습에 맞는 용어(접지, 인터록, 파형, 쇼트 등)를 자연스럽게 쓴다.
- 2~5문장, 평서체·기재요령에 맞는 격식, 과장·미사여구 금지.
- 제목·번호·글머리표 없이 본문만 출력한다(능력단위 표기용 대괄호는 허용)."""

    return _gemini_text(prompt, api_key, temperature=0.42, max_tokens=900)


TEACHER_COMMENT_GEMINI_MODELS: tuple[str, ...] = (
    GEMINI_PRIMARY_MODEL,
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-002",
)


def _gemini_text_with_models(
    prompt: str,
    api_key: str | None,
    model_names: tuple[str, ...],
    *,
    temperature: float = 0.35,
    max_tokens: int = 768,
) -> str | None:
    """지정 모델 순서로 generateContent를 시도한다."""
    key = resolve_google_api_key(api_key)
    if not key or not (prompt or "").strip():
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        gc = {"temperature": temperature, "max_output_tokens": max_tokens}
        safety = gemini_safety_settings_block_none()
        kwargs: dict = {"generation_config": gc}
        if safety:
            kwargs["safety_settings"] = safety
        for name in model_names:
            try:
                model = get_gemini_model(genai, name)
                if model is None:
                    if name == GEMINI_PRIMARY_MODEL:
                        _log.warning("Gemini teacher-path primary unavailable: %s", name)
                    continue
                response = model.generate_content(prompt, **kwargs)
                text = extract_generate_content_text(response)
                if text:
                    _note_gemini_model(name, reason="teacher-path")
                    return text
            except Exception as e:
                if name == GEMINI_PRIMARY_MODEL:
                    _log.warning("Gemini teacher-path primary failed: %s", e)
                continue
        return gemini_generate_text(genai, prompt, generation_config=gc)
    except Exception:
        pass
    return None


def _teacher_comment_keyword_fallback(student_label: str, meta: dict[str, Any]) -> str:
    """Gemini 실패 시 태도·인성 중심 종합의견 폴백."""
    total = int(meta.get("total_logs") or 0)
    return (
        f"{student_label}은(는) 한 학기 동안 {total}회의 실습 활동에 성실히 참여하였다. "
        "실습 중 어려움이 생겨도 포기하지 않고 원인을 차근차근 확인하려는 끈기를 보였으며, "
        "측정·점검 과정에서 안전 수칙을 지키려는 태도가 점차 안정되었다. "
        "동료와의 협력 상황에서 맡은 역할을 다하고, 실수를 성찰해 다음 실습에 반영하려는 "
        "자세가 돋보였다. 앞으로도 책임감 있는 실습 태도를 유지하며 성장하기를 기대한다."
    )


def generate_teacher_comprehensive_comment_draft(
    logs: list[dict],
    student_label: str,
    *,
    api_key: str | None = None,
) -> str:
    """
    실습 일지를 요약·분석해 「행동 특성 및 종합의견」 초안(~500자) 생성.
    Gemini 1.5 Flash 우선, 태도·인성 중심.
    """
    if not logs:
        return "해당 학생의 저장된 실습 일지가 없어 종합의견 초안을 작성할 수 없습니다."

    corpus, meta = summarize_logs_for_school_record(logs)
    if not corpus.strip():
        return _teacher_comment_keyword_fallback(student_label, meta)

    prompt = f"""너는 직업계고 교사야. 학생의 한 학기 실습 일지 데이터를 분석해서 '행동 특성 및 종합의견' 초안을 작성해 줘.
세특처럼 기술적인 세부 내용(NCS)보다는 학생의 실습 태도, 끈기, 문제해결 과정, 팀워크, 안전 수칙 준수 등 인성 및 태도적 측면에서의 성장에 초점을 맞춰서 500자 내외로 작성해.

학생: {student_label}
전체 실습 횟수: {meta.get("total_logs", 0)}회
분석 샘플: {meta.get("sampled_logs", 0)}건

[실습 일지 요약]
{corpus}

출력 규칙:
- 「행동 특성 및 종합의견」에 바로 기재할 수 있는 평서체 한 단락만 출력한다.
- NCS 코드·기술 용어 나열은 최소화하고, 태도·인성·성장 중심으로 쓴다.
- 450~550자(공백 포함) 내외.
- 제목·번호·글머리표·따옴표 없이 본문만 출력한다."""

    raw = _gemini_text_with_models(
        prompt,
        api_key,
        TEACHER_COMMENT_GEMINI_MODELS,
        temperature=0.4,
        max_tokens=1024,
    )
    if raw and len(raw.strip()) >= 60:
        text = re.sub(r"\s+", " ", raw.strip())
        if len(text) > 580:
            text = text[:577] + "…"
        return text
    return _teacher_comment_keyword_fallback(student_label, meta)


def _school_record_keyword_fallback(student_label: str, meta: dict[str, Any]) -> str:
    """Gemini 실패 시 규칙 기반 생기부 초안."""
    top = meta.get("top_unit") or "전기·전자 실습"
    total = int(meta.get("total_logs") or 0)
    unit_stats = meta.get("unit_stats") or []
    top3 = ", ".join(f"{s['unit']}({s['count']}회)" for s in unit_stats[:3])
    return (
        f"{student_label}은(는) 학기 동안 {total}회의 NCS 기반 실습 일지를 작성하며 "
        f"주로 {top3 or top} 영역에서 활동하였다. "
        f"반복 실습을 통해 {top} 직무의 기본 절차와 안전 수칙을 익히고, "
        f"측정·점검·오류 원인 추적 과정에서 문제 해결 태도를 보였다. "
        f"실습 기록을 스스로 정리하며 기술적 성장과 메타인지적 성찰을 꾸준히 드러내 "
        f"진로(전기·전자) 탐색과 자기주도 학습 역량을 키워 나가고 있다."
    )


def generate_school_record_draft(
    logs: list[dict],
    student_label: str,
    *,
    api_key: str | None = None,
) -> str:
    """
    누적 실습 일지를 요약·분석해 학교생활기록부 「진로활동」 또는 「자율활동」 문구 초안(~500자) 생성.
    """
    corpus, meta = summarize_logs_for_school_record(logs)
    if not corpus:
        return "해당 학생의 저장된 실습 일지가 없어 생활기록부 초안을 작성할 수 없습니다."

    prompt = f"""당신은 공업고등학교 전기·전자과 담임교사를 돕는 생활기록부 작성 조교이다.
아래는 한 학생의 NCS 기반 실습 포트폴리오(일지)를 토큰 절약을 위해 요약·샘플링한 자료이다.

학생: {student_label}
전체 실습 횟수: {meta.get("total_logs", 0)}회
분석에 사용한 샘플: {meta.get("sampled_logs", 0)}건

[요약 자료]
{corpus}

다음 순서로 **내부 분석**을 수행한 뒤, 최종 출력만 작성하라.
1) 가장 많이 수행한 핵심 직무(NCS 능력단위·코드·키워드 근거)
2) 반복적으로 드러난 강점과 기술적 숙련도
3) 실습 태도·성찰·발전 과정 요약

[출력 규칙]
- **학교생활기록부 「진로활동」 또는 「자율활동」**에 들어갈 서술형 문단 **한 개**만 출력한다.
- 분석 과정·머리말·번호·글머리표·따옴표·마크다운 금지.
- **450~550자(공백 포함)** 내외의 평서체 한국어.
- 없는 사실을 지어내지 말고, 요약 자료에서 합리적으로 추론 가능한 범위만 서술한다.
- 전기·전자·PLC·안전 등 실습 맥락에 맞는 NCS 용어를 2~4개 자연스럽게 포함한다."""

    raw = _gemini_text(prompt, api_key, temperature=0.38, max_tokens=1024)
    if raw and len(raw.strip()) >= 80:
        text = re.sub(r"\s+", " ", raw.strip())
        if len(text) > 580:
            text = text[:577] + "…"
        return text
    return _school_record_keyword_fallback(student_label, meta)


def extract_weak_radar_dimensions(values: list[float]) -> list[dict]:
    """
    5축 점수와 동일한 순서(RADAR_AXES)의 값.
    약점: 해당 축 < 30 이거나, 나머지 4축 평균의 80% 이하(20% 이상 낮음).
    """
    out: list[dict] = []
    n = len(RADAR_AXES)
    if len(values) != n:
        return out
    for j, ax in enumerate(RADAR_AXES):
        v = float(values[j])
        others = [float(values[k]) for k in range(n) if k != j]
        mean_o = sum(others) / len(others)
        if v < 30:
            out.append(
                {"axis": ax, "reason": "30점 미만", "value": v, "others_avg": mean_o}
            )
        elif mean_o > 0 and v <= mean_o * 0.8:
            out.append(
                {
                    "axis": ax,
                    "reason": "타 영역 평균 대비 20% 이상 낮음",
                    "value": v,
                    "others_avg": mean_o,
                }
            )
    return out


def _get_ncs_terms() -> set[str]:
    """constants에서 NCS 전문 용어 수집 (키워드·용어·NCS 표준명)."""
    try:
        from constants import GLOSSARY, NCS_DB, COLLOQUIAL_TO_NCS
    except ImportError:
        return set()
    terms: set[str] = set(GLOSSARY.keys())
    for meta in NCS_DB.values():
        terms.update(meta.get("keywords", []))
    for phrases, ncs_term, _ in COLLOQUIAL_TO_NCS:
        terms.add(ncs_term)
        terms.update(phrases)
    return {t for t in terms if t and len(t) >= 2}


def _highlight_ncs_terms(text: str, terms: set[str]) -> str:
    """텍스트 내 NCS 전문 용어를 <strong>으로 강조. 플레이스홀더로 중첩 방지."""
    if not text or not terms:
        return text.replace("<", "&lt;").replace(">", "&gt;")
    escaped = text.replace("<", "&lt;").replace(">", "&gt;")
    markers: list[tuple[str, str]] = []
    for i, term in enumerate(sorted(terms, key=len, reverse=True)):
        if len(term) < 2 or term not in escaped:
            continue
        ph = f"\x00NCS{i}\x00"
        markers.append((ph, f"<strong style='color:#334155;font-weight:600;border-bottom:1px dotted #94a3b8;'>{term}</strong>"))
        escaped = escaped.replace(term, ph)
    for ph, tag in markers:
        escaped = escaped.replace(ph, tag)
    return escaped


def render_original_vs_refined(original: str, refined: str) -> str:
    """
    다듬기 전(Original)과 다듬은 후(AI Refined)를 나란히 보여주는 HTML.
    메타인지적 성찰 유도: 학생이 일상 언어→전문 용어 치환 과정을 학습.
    """
    orig_html = render_bsr_highlighted(original) if original else "<p style='color:#94a3b8;font-style:italic;'>(내용 없음)</p>"
    ref_html = render_bsr_highlighted(refined) if refined else "<p style='color:#94a3b8;font-style:italic;'>(다듬기 전 내용을 입력한 뒤 'AI 전문 문장으로 다듬기' 버튼을 누르세요)</p>"
    return (
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;margin:1rem 0;'>"
        "<div style='border:1px solid #e2e8f0;border-radius:8px;padding:1rem;background:#f8fafc;'>"
        "<p style='margin:0 0 0.75rem;font-weight:600;color:#64748b;font-size:0.9em;'>다듬기 전 (Original)</p>"
        f"<div style='font-size:0.9em;'>{orig_html}</div></div>"
        "<div style='border:1px solid #1e3a5f;border-radius:8px;padding:1rem;background:#f0f9ff;'>"
        "<p style='margin:0 0 0.75rem;font-weight:600;color:#1e3a5f;font-size:0.9em;'>다듬은 후 (AI Refined)</p>"
        f"<div style='font-size:0.9em;'>{ref_html}</div></div></div>"
    )


def render_bsr_highlighted(bsr_text: str, highlight_terms: bool = True) -> str:
    """
    BSR 텍스트를 '프로젝트 보고서'의 소제목(Sub-heading) 양식으로 렌더링한다.
    날것의 태그 대신 다음 라벨로 치환된다:
      [What] / [경험]     -> What — 실무 경험
      [So What] / [의미]  -> So What — 판단 및 성찰
      [Now What] / [적용] -> Now What — 향후 적용
      [배경] / [해결] / [성과] -> 이전 형식 라벨 (레거시 일지)
      [체크리스트:] -> NCS 수행준거 점검
      [성찰메타] -> 화면에서 숨김
    highlight_terms=True일 때 NCS 전문 용어를 굵게·밑줄로 강조한다.

    반환 HTML은 독립 실행(HTML 다운로드)·Streamlit 앱 양쪽에서 모두 동작하도록
    인라인 스타일만 사용한다. 소제목은 이모지 없이 좌측 컬러 바 + 하단 헤어라인으로
    구분감을 준다.
    """
    if not bsr_text:
        return ""
    escaped = lambda s: (s or "").replace("<", "&lt;").replace(">", "&gt;")
    ncs_terms = _get_ncs_terms() if highlight_terms else set()

    # 섹션별 색상 포인트 (아이콘 없음 — 진중한 텍스트 전용)
    section_defs = {
        "[What]": ("What — 실무 경험", "#1d4ed8"),
        "[경험]": ("What — 실무 경험", "#1d4ed8"),
        "[So What]": ("So What — 판단 및 성찰", "#b45309"),
        "[의미]": ("So What — 판단 및 성찰", "#b45309"),
        "[Now What]": ("Now What — 향후 적용", "#047857"),
        "[적용]": ("Now What — 향후 적용", "#047857"),
        "[배경]": ("이전 형식 · 실습 배경 및 목표", "#1d4ed8"),
        "[해결]": ("이전 형식 · 수행 과정", "#b45309"),
        "[성과]": ("이전 형식 · 성과 및 성찰", "#047857"),
    }

    # 소제목: 좌측 세로바(Border-left) + 하단 얇은 헤어라인만으로 구분
    def _title_style(accent: str) -> str:
        return (
            "display:block;"
            "font-family:'Noto Sans KR','Segoe UI',sans-serif;"
            f"color:{accent};"
            "font-size:1.02em;font-weight:700;letter-spacing:-0.01em;"
            "margin:1.05rem 0 0.5rem 0;"
            "padding:0.2rem 0 0.45rem 0.75rem;"
            f"border-left:3px solid {accent};"
            f"border-bottom:1px solid {accent}26;"
            "background:transparent;"
        )

    # 본문: 소제목 아래 살짝 들여쓰기하여 보고서처럼 이어짐
    body_style = (
        "padding:0.15rem 0 0.35rem 0.95rem;"
        "color:#334155;line-height:1.8;font-size:0.95em;word-wrap:break-word;"
        "border-left:2px solid #f1f5f9;margin:0 0 0.9rem 0.15em;"
    )
    section_wrap = "display:block;margin:0 0 0.35rem 0;"
    empty_placeholder = (
        "<span style=\"color:#94a3b8;font-style:italic;font-size:0.9em;\">"
        "(내용 없음)</span>"
    )

    def _section_html(tag_key: str, content: str) -> str:
        label, accent = section_defs[tag_key]
        cnt = _highlight_ncs_terms(content.strip(), ncs_terms) if ncs_terms else escaped(content.strip())
        if not (cnt or "").strip():
            cnt = empty_placeholder
        return (
            f"<section style='{section_wrap}'>"
            f"<h4 style='{_title_style(accent)}'>{label}</h4>"
            f"<div style='{body_style}'>{cnt}</div>"
            f"</section>"
        )

    def _checklist_html(raw: str) -> str:
        # raw: "[체크리스트: a; b; c]"
        accent = "#475569"
        inner = raw[len("[체크리스트:"):].rstrip("]").strip()
        items = [s.strip() for s in re.split(r"[;·,]", inner) if s.strip()]
        if items:
            items_html = (
                "<ul style=\"margin:0;padding:0 0 0 1.1rem;line-height:1.75;list-style:square;\">"
                + "".join(
                    f"<li style=\"margin:0.15rem 0;color:#334155;font-size:0.95em;\">"
                    f"{escaped(it)}</li>"
                    for it in items
                )
                + "</ul>"
            )
        else:
            items_html = empty_placeholder
        return (
            f"<section style='{section_wrap}'>"
            f"<h4 style='{_title_style(accent)}'>NCS 수행준거 점검</h4>"
            f"<div style='{body_style}'>{items_html}</div>"
            f"</section>"
        )

    parts = re.split(
        r"(\[What\]|\[So What\]|\[Now What\]|\[경험\]|\[의미\]|\[적용\]|"
        r"\[배경\]|\[해결\]|\[성과\]|\[성찰메타\]|\[체크리스트:[^\]]*\])",
        bsr_text,
    )
    result: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
        if p == "[성찰메타]":
            i += 2 if i + 1 < len(parts) else 1
            continue
        if p in section_defs:
            content = parts[i + 1] if i + 1 < len(parts) else ""
            result.append(_section_html(p, content))
            i += 2
        elif p.startswith("[체크리스트:"):
            result.append(_checklist_html(p))
            i += 1
        else:
            if p and p.strip():
                # 태그 밖의 자유 텍스트는 일반 본문 문단으로
                result.append(
                    f"<p style='color:#334155;line-height:1.75;font-size:0.95em;margin:0 0 0.5rem 0;'>"
                    f"{escaped(p).strip()}</p>"
                )
            i += 1

    return (
        "<div class='bsr-report' style=\"line-height:1.75;color:#1e293b;\">"
        + "".join(result).replace("\n", "<br/>")
        + "</div>"
    )


def _detected_tools_to_str(detected_tools: list[dict] | list[str] | None) -> str:
    """사진 분석 결과(장비 목록)를 프롬프트용 문자열로."""
    if not detected_tools:
        return "(인식된 장비 없음)"
    lines: list[str] = []
    for d in detected_tools[:12]:
        if isinstance(d, dict):
            lines.append(f"- {d.get('객체', '—')} (신뢰도 {d.get('신뢰도', '—')})")
        else:
            lines.append(f"- {d}")
    return "\n".join(lines)


def _parse_numbered_lines(text: str, max_items: int = 3) -> list[str]:
    """모델 출력에서 질문 줄만 추출 (번호·불릿 제거)."""
    out: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[\d]+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        if len(line) >= 8:
            out.append(line)
        if len(out) >= max_items:
            break
    return out


def _gemini_text(prompt: str, api_key: str | None, *, temperature: float = 0.35, max_tokens: int = 768) -> str | None:
    key = resolve_google_api_key(api_key)
    if not key or not (prompt or "").strip():
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        return gemini_generate_text(
            genai,
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
    except Exception:
        pass
    return None


def get_ai_scaffolding(
    content: str,
    detected_tools: list[dict] | list[str] | None,
    ncs_unit: str,
    *,
    prior_radar_axes: list[str] | None = None,
    prior_radar_values: list[float] | None = None,
    api_key: str | None = None,
) -> list[str]:
    """
    학생 초안·인식 장비·NCS 단위·(선택) 누적 레이더를 통합 문맥으로 역질문 3개 생성.
    RECOMMENDED_QA 고정 리스트 대신 Gemini 사용, 실패 시 휴리스틱 폴백.
    """
    key = resolve_google_api_key(api_key)
    tools_str = _detected_tools_to_str(detected_tools)
    unit = (ncs_unit or "").strip() or "(미선택)"
    body = (content or "").strip() or "(학생 입력 없음)"

    if (
        prior_radar_axes
        and prior_radar_values
        and len(prior_radar_axes) == len(prior_radar_values)
        and len(prior_radar_values) == len(RADAR_AXES)
    ):
        pairs = ", ".join(f"{a}: {v}점" for a, v in zip(prior_radar_axes, prior_radar_values))
        weak_info = extract_weak_radar_dimensions(list(prior_radar_values))
        weak_str = (
            ", ".join(
                f"{w['axis']}({w['reason']}, {w['value']:.0f}점)"
                for w in weak_info
            )
            if weak_info
            else "누적 기준으로 두드러진 상대 약점 없음 또는 기록 부족"
        )
        radar_block = f"""
[누적 실습 기록 기준 최근 레이더 점수(0~100) — 개인화 참고]
- 축별 점수: {pairs}
- 상대적 약점 후보: {weak_str}
"""
    else:
        radar_block = "[누적 레이더 정보 없음 — 아래 '이전 약점 연계' 질문은 생략 가능]"

    prompt = f"""너는 공업고 전자과 교사다. 아래 **통합 문맥**(초안·사진 인식 장비·누적 역량)을 하나의 실습 상황으로 해석하고, 기술적으로 구체적인 역질문을 정확히 3개만 작성해라.

[학생이 작성한 초안]
{body}

[사진에서 인식된 장비]
{tools_str}

{radar_block}

[선택·매칭된 NCS 능력단위]
{unit}

핵심 지시:
- 학생이 업로드한 사진의 장비와 초안의 현상·측정값·증상을 **서로 연결**해 질문을 만든다.
- 단순히 "무엇을 했는지"를 묻지 말고, **'왜 그런 파형·전압·동작이 나왔는지'**, **트러블슈팅 과정에서 어떤 기술적 판단을 했는지'**, **가설과 검증 순서는 어떻게 짰는지** 등 메타인지·원인 분석을 자극하는 질문을 포함한다.
- **학생의 이전 기록에서 점수가 낮았던 역량 축(예: 안전)이 있다면**, 오늘 실습 내용과 **연결하여 그 부분을 보완할 수 있는 질문을 1개 포함**한다. (예: 지난번엔 안전 점수가 낮았는데, 오늘 회로 시험 전 LOTO 체크는 어떻게 했나요?) — 누적 레이더 정보가 없거나 약점이 없으면 이 항목은 다른 기술 질문으로 채운다.
- 전자과 실습에 맞는 전문 용어(접지, 쇼트, 극성, 파형, 리플, 인터록, 래더, 입출력 등)를 상황에 맞게 사용한다.
- 오실로스코프·파형이 언급되면 전압·주기·노이즈·트리거·왜곡 등과 연결된 질문을 포함할 수 있다.
- 회로 조립·브레드보드·PCB·납땜이 있으면 배선·접지·쇼트·부품 방향·납땜 품질 관련 질문을 포함할 수 있다.
- PLC·제어 관련이면 시퀀스·인터록·입출력 대조·시운전 절차 관련 질문을 포함할 수 있다.
- 각 질문은 한 문장으로 끝낸다.
- 출력은 질문 3줄만. 번호나 기호 없이 한 줄에 질문 하나씩. 다른 설명·인사 금지."""

    raw = _gemini_text(prompt, key, temperature=0.35, max_tokens=512)
    qs = _parse_numbered_lines(raw or "", 3) if raw else []
    weak_hint: list[str] | None = None
    if prior_radar_values and len(prior_radar_values) == len(RADAR_AXES):
        weak_hint = [w["axis"] for w in extract_weak_radar_dimensions(list(prior_radar_values))]
        if not weak_hint:
            weak_hint = None
    if len(qs) >= 3:
        return qs[:3]
    return _fallback_scaffolding_questions(
        body, unit, detected_tools or [], qs, weak_axes_hint=weak_hint
    )


def _fallback_scaffolding_questions(
    content: str,
    ncs_unit: str,
    detected_tools: list,
    partial: list[str] | None = None,
    weak_axes_hint: list[str] | None = None,
) -> list[str]:
    """API 실패 또는 파싱 부족 시 보강."""
    c = content or ""
    lc = c.lower()
    u = (ncs_unit or "").lower()
    pool: list[str] = list(partial or [])

    def add(q: str) -> None:
        if q not in pool:
            pool.append(q)

    if weak_axes_hint:
        for ax in weak_axes_hint[:2]:
            if ax == "안전":
                add(
                    "누적 기록에서 안전 역량이 상대적으로 낮았다. 오늘 실습 전에 전원 차단·LOTO·보호구 확인을 어떤 순서로 수행했는가?"
                )
            elif ax == "제어":
                add(
                    "이전 기록에서 제어 역량이 상대적으로 낮았다. 오늘 시퀀스·인터록·입출력을 어떤 순서로 대조·검증했는가?"
                )
            elif ax == "계측":
                add(
                    "누적 기록에서 계측 역량이 상대적으로 낮았다. 오늘 측정값을 이론·시뮬과 어떻게 대조했고 불일치 시 원인을 어디부터 좁혔는가?"
                )
            elif ax == "설계":
                add(
                    "이전 기록에서 설계 역량이 상대적으로 낮았다. 오늘 회로도·사양과 실제 배선·부품 선정을 어떻게 일치시켰는가?"
                )
            elif ax == "제작":
                add(
                    "누적 기록에서 제작 역량이 상대적으로 낮았다. 오늘 납땜·배선 품질을 어떤 기준으로 점검했는가?"
                )

    if any(k in c for k in ["오실", "oscillo", "파형", "wave"]) or any(
        "오실" in str(d) for d in (detected_tools or [])
    ):
        add(
            "오실로스코프로 관측한 파형의 진폭·주파수·DC 바이어스는 이론값·시뮬값과 어떻게 대조했는가?"
        )
        add("측정 시 노이즈·리플·링잉이 보였다면 원인을 회로의 어느 부분과 연결해 분석했는가?")
    if any(k in c for k in ["브레드", "배선", "회로", "쇼트", "접지", "극성", "납땜", "PCB"]):
        add(
            "회로도와 실제 배선을 대조할 때 오배선·접지·부품 극성 오류를 어떤 순서로 점검했는가?"
        )
        add("쇼트 의심 구간을 좁히기 위해 전원 차단·저항 측정·시각 검사 중 어떤 절차를 우선했는가?")
    if "plc" in lc or "래더" in c or "인터록" in c or "plc" in u:
        add(
            "작성한 래더 논리에서 안전 인터록 조건은 무엇이며, 시운전 시 그 조건이 충족됐는지 어떻게 확인했는가?"
        )
        add("입력·출력 램프 또는 모니터링 값과 현장 동작이 일치하는지 어떻게 대조 검증했는가?")
    if len(pool) < 3:
        add(
            f"[{ncs_unit or '해당 단위'}] 실습 목표 대비 오늘 수행한 핵심 절차와 품질·안전 기준은 무엇이었는가?"
        )
    if len(pool) < 3:
        add("동일 실습을 다시 한다면 측정·점검 순서를 어떻게 바꾸고 싶은가, 그 이유는 무엇인가?")
    if len(pool) < 3:
        add("실습 중 가장 위험했거나 개선이 필요했던 요인 한 가지와, 이를 줄이기 위한 구체적 조치는 무엇인가?")
    out = pool[:3]
    while len(out) < 3:
        out.append("오늘 실습에서 측정·검증한 결과를 근거로, 다음 단계에서 보완할 점은 무엇인가?")
        out = out[:3]
    return out[:3]


def get_reflection_example_sentence(
    content: str,
    detected_tools: list[dict] | list[str] | None,
    ncs_unit: str,
    *,
    api_key: str | None = None,
) -> str:
    """
    전자회로 실습 맥락에 맞는 NCS 수행준거 톤의 성찰 문장 1개(예시) 생성.
    사진 인식 장비를 통합 문맥으로 반영한다.
    """
    key = resolve_google_api_key(api_key)
    tools_str = _detected_tools_to_str(detected_tools)
    unit = (ncs_unit or "").strip() or "(미선택)"
    body = (content or "").strip() or "(학생 입력 없음)"

    prompt = f"""너는 공업고 전자과 교사다. 아래 **통합 문맥**(초안·인식 장비)을 반영해
전자회로 실습에 적합한 **성찰 문장 예시**를 딱 1문장만 작성해라.

형식: NCS 수행준거 스타일로 '~함', '~확인함', '~검토함' 등으로 끝낸다.
내용: 측정·배선·점검 등을 구체적으로 반영하고, 전문 용어를 자연스럽게 쓴다.
인용부호 없이 문장만 출력한다.

[학생 초안]
{body}

[인식 장비]
{tools_str}

[NCS 능력단위]
{unit}"""

    raw = _gemini_text(prompt, key, temperature=0.4, max_tokens=256)
    one = (raw or "").strip().splitlines()
    line = one[0].strip() if one else ""
    line = line.strip().strip('"\'「」')
    if len(line) >= 20:
        return line
    return _fallback_reflection_example(body, unit, detected_tools or [])


def _fallback_reflection_example(content: str, ncs_unit: str, detected_tools: list) -> str:
    if "오실" in content or "파형" in content:
        return (
            "회로도와 실제 브레드보드 배선을 대조하며 오배선 여부를 꼼꼼히 확인하고, 오실로스코프로 파형의 왜곡을 측정하여 회로의 안정성을 검토함."
        )
    if "PLC" in content or "래더" in content:
        return (
            "래더 다이어그램의 인터록 조건을 운전 순서도와 대조하여 검증하고, 시운전 시 입·출력 상태를 단계별로 확인하여 오동작 원인을 점검함."
        )
    return (
        f"{ncs_unit or '전자 실습'} 맥락에서 부품 극성·배선·접지 상태를 순차적으로 점검하고, "
        "측정 결과를 근거로 회로 동작의 적합성을 성찰함."
    )


def generate_teacher_learning_guidance(
    case_records: list[dict],
    *,
    api_key: str | None = None,
) -> str | None:
    """
    레이더 약점 자동 추출 결과를 바탕으로 교사용 지도·비계 문장을 생성.
    case_records: student_label, uid, axis, reason, value, others_avg, scores(선택) 등.
    """
    if not case_records:
        return None
    payload = json.dumps(case_records, ensure_ascii=False, indent=2)
    prompt = f"""당신은 공업고등학교 전기·전자과 실습 지도 교사를 돕는 멘토입니다.
아래는 학생별 실습 성찰 키워드 기반 레이더(설계·제작·계측·제어·안전, 0~100)에서 자동 추출된 '주의 필요' 구간입니다.

데이터:
{payload}

각 사례에 대해 **교사가 다음 실습이나 성찰 활동에서 적용할 수 있는 지도 방안**을 한 문장씩 제안하세요.
- 약점 영역과 상대적으로 강한 영역(others_avg)을 대비하여 구체적으로 서술합니다.
- 예시 형식: "S05 학생은 [제어] 영역 실습은 활발하나 [안전] 영역 점수가 낮습니다. 다음 실습 성찰 시 LOTO(에너지 차단) 절차나 개인보호구(PPE) 확인 여부를 묻는 비계를 설정해 보세요."
- 실무 키워드(LOTO, PPE, 비계, 인터록, 접지, 쇼트, 파형 등)를 상황에 맞게 포함할 수 있습니다.
- 한국어로만 출력합니다. 서론·요약 없이 각 사례별로 한 문단 또는 번호 목록으로 작성합니다."""

    return _gemini_text(prompt, api_key, temperature=0.4, max_tokens=2048)
