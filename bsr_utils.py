"""BSR 구조 시각화 공용 유틸. [배경][해결][성과] 구간별 색상 하이라이트 + NCS 전문 용어 강조."""
import hashlib
import io
import json
import os
import re

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


# Google AI: 404·지역·키 제한 대비 다중 모델 순차 시도 (텍스트·비전 공통).
# list_models() 결과가 우선이며, 여기는 탐색 실패·구버전 SDK 대비 정적 꼬리 목록이다.
GEMINI_MODEL_TRY_ORDER: tuple[str, ...] = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-1.5-flash-002",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-pro-vision",
    "gemini-1.0-pro-vision",
)
GEMINI_UNIFIED_MODEL: str = GEMINI_MODEL_TRY_ORDER[0]
GEMINI_TEXT_MODEL_CANDIDATES: tuple[str, ...] = GEMINI_MODEL_TRY_ORDER
GEMINI_VISION_MODEL_CANDIDATES: tuple[str, ...] = GEMINI_MODEL_TRY_ORDER

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


def resolved_gemini_model_candidates(
    genai,
    static_tail: tuple[str, ...] | None = None,
) -> list[str]:
    """키별 노출 모델을 앞에 두고, 정적 꼬리 목록과 중복 없이 합친다."""
    tail = static_tail or GEMINI_MODEL_TRY_ORDER
    merged: list[str] = []
    seen: set[str] = set()
    for n in discover_gemini_generate_model_names(genai) + list(tail):
        if n and n not in seen:
            seen.add(n)
            merged.append(n)
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
    """``response.text``만 쓰지 않고 candidates/parts를 훑어 본문을 수집한다(차단·빈 후보 대비)."""
    if response is None:
        return ""
    chunks: list[str] = []
    try:
        for cand in getattr(response, "candidates", None) or []:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", None) or []:
                t = getattr(part, "text", None) or ""
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
            t = getattr(part, "text", None) or ""
            if t:
                chunks.append(t)
    except Exception:
        pass
    return "".join(chunks).strip()


def get_gemini_model(genai, model_name: str | None = None):
    """GenerativeModel 생성.

    - ``model_name``이 있으면 해당 이름만 시도하고, 실패 시 ``None``을 반환한다.
    - ``model_name``이 없으면 ``list_models``·정적 꼬리를 합친 순서로 시도하고,
      모두 실패하면 Streamlit이 있을 때 ``st.error`` 후 ``st.stop``한다.
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


def gemini_generate_text(genai, prompt: str, *, generation_config: dict | None = None) -> str | None:
    """generateContent 지원 모델을 순서대로 시도. 전부 실패 시 None."""
    gc = generation_config or {}
    safety = gemini_safety_settings_block_none()
    kwargs: dict = {"generation_config": gc}
    if safety:
        kwargs["safety_settings"] = safety
    for name in resolved_gemini_model_candidates(genai):
        try:
            model = get_gemini_model(genai, name)
            if model is None:
                continue
            response = model.generate_content(prompt, **kwargs)
            text = extract_generate_content_text(response)
            if text:
                return text
        except Exception:
            continue
    return None


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
    text_all = " ".join(str(r.get("bsr", "")) for r in logs)
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
    """[배경] 구간만 추출. 없으면 전체를 사용."""
    m = re.search(r"\[배경\]\s*(.*?)(?=\[해결\]|\[성과\]|\Z)", content or "", re.DOTALL)
    return (m.group(1).strip() if m else (content or "").strip())


def extract_bsr_section(bsr_text: str, section: str) -> str:
    """
    BSR 문자열에서 [배경]|[해결]|[성과] 태그 뒤의 본문만 추출.
    student_view 미리보기·AI 초안 분리 등에 공통 사용.
    """
    if not bsr_text or section not in ("배경", "해결", "성과"):
        return ""
    prefix = f"[{section}]"
    i = bsr_text.find(prefix)
    if i < 0:
        return ""
    segment = bsr_text[i + len(prefix) :]
    boundaries: list[int] = []
    for t in ("[배경]", "[해결]", "[성과]", "[체크리스트:"):
        if t == prefix:
            continue
        pos = segment.find(t)
        if pos >= 0:
            boundaries.append(pos)
    end = min(boundaries) if boundaries else len(segment)
    return segment[:end].strip()


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
    짧은 메모·키워드와 사진 인식 장비 목록으로 Gemini가 BSR 초안을 생성한다.
    반환: {"background": str, "solution": str, "reflection": str}.
    API 키·메모가 비어 있으면 빈 dict를 반환하고, 그 외 Gemini/파싱 오류는 RuntimeError를 발생시킨다.
    """
    key = (api_key or "").strip() or resolve_google_api_key()
    empty = {"background": "", "solution": "", "reflection": ""}
    if not key or not (raw_text or "").strip():
        return dict(empty)

    lines: list[str] = []
    for d in (detected_tools or [])[:12]:
        obj = (d or {}).get("객체", "") or ""
        conf = (d or {}).get("신뢰도", "") or ""
        if obj:
            lines.append(f"- {obj}" + (f" ({conf})" if conf else ""))
    tools_block = "\n".join(lines) if lines else "(사진에서 장비를 특정하지 못했거나 사진이 없습니다.)"

    prompt = f"""당신은 공업고등학교 전기·전자과 실습 지도를 돕는 교사이다.
학생이 남긴 짧은 메모·키워드와 사진에서 인식된 장비 목록을 바탕으로 실습 일지 BSR 초안을 작성한다.

【출력 형식 — 반드시 준수】
- 출력은 JSON 한 덩어리만. 다른 설명·머리말·마크다운 코드펜스 금지.
- 키는 반드시 영어로 다음 세 개만 사용: "background", "solution", "reflection"
- 값은 한국어 순수 텍스트. 별표·밑줄 등 마크다운 강조 기호는 쓰지 말 것.
- background = 실습 목적·상황·환경·장비와의 연관 (2~5문장 수준)
- solution = 문제·시도·절차·측정·안전 (2~5문장 수준)
- reflection = 배운 점·느낀 점·다음 실습에 적용할 점 (2~5문장 수준)
- 없는 사실을 지어내지 말고, 메모·장비 목록에서 합리적으로 추론해 문장을 보강한다.

예시 형식:
{{"background": "...", "solution": "...", "reflection": "..."}}

[학생 메모]
{raw_text.strip()[:8000]}

[사진 인식 장비·기기]
{tools_block}
"""

    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        gc = {"temperature": 0.38, "max_output_tokens": 2048}
        raw = gemini_generate_text(genai, prompt, generation_config=gc)
        if not raw or not str(raw).strip():
            raise RuntimeError(GEMINI_EMPTY_RESPONSE_MESSAGE)
        out = _parse_magic_draft_json_or_tags(raw)
        if not (out.get("background") or out.get("solution") or out.get("reflection")):
            raise RuntimeError(
                "BSR JSON/태그 파싱 후 배경·해결·성과가 모두 비었습니다. "
                f"응답 앞부분: {str(raw)[:900]!r}"
            )
        return out
    except Exception as e:
        raise RuntimeError(f"BSR 초안 생성 중 오류: {e}") from e


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
학생이 제출한 {photo_phrase}과 **[배경] 텍스트**가 서로 **적절한 증거 관계**인지 평가하라.

[배경]에 서술된 활동·장비·상황이 사진에 보이는 내용과 논리적으로 맞는가?
(예: 본문은 PLC 실습인데 사진만 납땜이면 낮은 점수){multi_extra}

[학생 배경 글]
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
        for name in resolved_gemini_model_candidates(genai):
            try:
                model = get_gemini_model(genai, name)
                if model is None:
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
                    break
            except Exception:
                continue
        parsed = _parse_evidence_score_0_100(raw)
        if parsed is not None:
            return parsed
    except Exception:
        pass
    return 75.0


def _compress_bsr_for_school_record_summary(bsr: str, *, max_per_section: int = 180) -> str:
    """토큰 절약을 위해 BSR을 구간별 짧은 한 줄로 압축."""
    parts: list[str] = []
    for sec in ("배경", "해결", "성과"):
        t = extract_bsr_section(bsr, sec)
        if t:
            one = re.sub(r"\s+", " ", t).strip()
            if len(one) > max_per_section:
                one = one[: max_per_section - 1] + "…"
            parts.append(f"{sec}:{one}")
    if parts:
        return " | ".join(parts)
    flat = re.sub(r"\s+", " ", (bsr or "").strip())
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
아래는 한 학생의 NCS 기반 실습 일지(BSR)를 토큰 절약을 위해 요약·샘플링한 자료이다.
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
    날것의 [배경]/[해결]/[성과] 태그 대신 다음 라벨로 치환된다:
      [배경]       -> 실습 배경 및 목표
      [해결]       -> 기술적 문제 해결 및 수행 과정
      [성과]       -> 직무 역량 성장 및 성찰
      [체크리스트:] -> NCS 수행준거 점검
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
        "[배경]": ("실습 배경 및 목표", "#1d4ed8"),
        "[해결]": ("기술적 문제 해결 및 수행 과정", "#b45309"),
        "[성과]": ("직무 역량 성장 및 성찰", "#047857"),
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

    parts = re.split(r"(\[배경\]|\[해결\]|\[성과\]|\[체크리스트:[^\]]*\])", bsr_text)
    result: list[str] = []
    i = 0
    while i < len(parts):
        p = parts[i]
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
아래는 학생별 BSR 키워드 기반 레이더(설계·제작·계측·제어·안전, 0~100)에서 자동 추출된 '주의 필요' 구간입니다.

데이터:
{payload}

각 사례에 대해 **교사가 다음 실습이나 성찰 활동에서 적용할 수 있는 지도 방안**을 한 문장씩 제안하세요.
- 약점 영역과 상대적으로 강한 영역(others_avg)을 대비하여 구체적으로 서술합니다.
- 예시 형식: "S05 학생은 [제어] 영역 실습은 활발하나 [안전] 영역 점수가 낮습니다. 다음 실습 성찰 시 LOTO(에너지 차단) 절차나 개인보호구(PPE) 확인 여부를 묻는 비계를 설정해 보세요."
- 실무 키워드(LOTO, PPE, 비계, 인터록, 접지, 쇼트, 파형 등)를 상황에 맞게 포함할 수 있습니다.
- 한국어로만 출력합니다. 서론·요약 없이 각 사례별로 한 문단 또는 번호 목록으로 작성합니다."""

    return _gemini_text(prompt, api_key, temperature=0.4, max_tokens=2048)
