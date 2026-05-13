import datetime
import html
import io
import json
import re
import textwrap
from collections import Counter, defaultdict
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from student_view import (
    _bsr_preview_snippet,
    _build_project_pages_html,
    _build_resume_page_html,
    _portfolio_css,
)
from bsr_utils import (
    RADAR_AXES,
    extract_weak_radar_dimensions,
    generate_seuteuk_from_bsr_logs,
    generate_teacher_learning_guidance,
    radar_scores_from_logs,
    render_bsr_highlighted,
    resolve_google_api_key,
)
from backup_utils import copy_log_row, logs_to_csv_bytes, profile_to_json_bytes
from constants import DEFAULT_NCS_PROGRESS, format_ncs_unit, GLOSSARY, NCS_DB
from db import (
    STUDENT_COUNT,
    STUDENT_UIDS,
    TEACHER_UID,
    TEST_PERIOD_END,
    TEST_PERIOD_START,
    add_researcher_log,
    app_today,
    clear_logs,
    clear_student_profile,
    delete_log,
    get_portfolio_comment,
    get_student_profile,
    list_logs,
    list_researcher_logs,
    list_user_credentials,
    list_users,
    save_portfolio_comment,
    seed_progress_if_missing,
    student_number,
    test_period_weekdays,
    update_password,
)
from ui_style import P, render_password_change_expander

# 종합 대시보드 Plotly 히트맵: 학생(1~10번) × NCS 핵심 단위 실습 빈도 (전자 능력단위 중심)
CORE_NCS_HEATMAP_UNITS: list[str] = [
    "전자부품장착",
    "전자회로조립",
    "전자회로설계",
    "PCB설계",
    "마이크로컨트롤러",
    "임베디드하드웨어설계",
    "센서응용",
    "산업통신",
    "통신기기하드웨어개발",
    "PLC제어",
    "인버터제어",
    "전기안전",
]


def _heatmap_frequency_matrix(_students: list[dict]) -> tuple[list[str], list[str], list[list[int]]]:
    """행: 1~10번 학생 전원, 열: 핵심 NCS 단위, 값: 해당 단위 일지 건수."""
    col_units = CORE_NCS_HEATMAP_UNITS
    row_labels: list[str] = []
    z: list[list[int]] = []
    for uid in STUDENT_UIDS:
        row_labels.append(_student_label(uid))
        logs = list_logs(uid)
        counts = {u: 0 for u in col_units}
        for r in logs:
            unit = _resolve_ncs_unit(r.get("ncs_unit", "") or "")
            if unit in counts:
                counts[unit] += 1
        z.append([counts[u] for u in col_units])
    col_display = [format_ncs_unit(u) for u in col_units]
    return row_labels, col_display, z


def _student_label(uid: str) -> str:
    """yongsan1 → '1번 학생'"""
    n = student_number(uid)
    return f"{n}번 학생" if n != 999 else str(uid)


def _student_sort_key(uid: str) -> int:
    """1번~10번 순서 정렬용 (yongsan1=1, yongsan10=10)"""
    return student_number(uid)


def _parse_log_date(val) -> datetime.date | None:
    if val is None:
        return None
    s = str(val).strip()[:10]
    try:
        return datetime.datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError:
        return None


_KOR_DAY = ["월", "화", "수", "목", "금", "토", "일"]


def _build_test_period_attendance(students: list[dict]) -> pd.DataFrame:
    """
    테스트 기간(2026-05-11 ~ 2026-05-29) 평일(월~금)을 가로축, 학생을 세로축으로 하는
    제출 현황판 DataFrame. 해당 일에 일지가 있으면 '●', 없으면 '·'.
    오늘 이후 미래 날짜는 '–'로 비워둔다. 우측 끝 '총 제출'은 기간 내 일지 건수 합계.
    """
    date_list = test_period_weekdays()
    today = app_today()
    date_set = set(date_list)
    col_labels = [f"{d.strftime('%m.%d')}({_KOR_DAY[d.weekday()]})" for d in date_list]

    rows: list[dict] = []
    for s in students:
        uid = s["uid"]
        logs = list_logs(uid)
        per_day: dict[datetime.date, int] = defaultdict(int)
        for r in logs:
            d = _parse_log_date(r.get("date"))
            if d is not None and d in date_set:
                per_day[d] += 1
        row: dict[str, object] = {"학생": _student_label(uid)}
        for d, lab in zip(date_list, col_labels, strict=True):
            if d > today:
                row[lab] = "–"
            else:
                row[lab] = "●" if per_day.get(d, 0) > 0 else "·"
        row["총 제출"] = sum(per_day.values())
        rows.append(row)

    out = pd.DataFrame(rows)
    ordered_cols = ["학생"] + col_labels + ["총 제출"]
    if out.empty:
        return pd.DataFrame(columns=ordered_cols)
    return out[ordered_cols]


def _count_submissions_today(students: list[dict]) -> int:
    """오늘(앱 기준) 일지를 1건 이상 제출한 학생 수."""
    today_str = app_today().isoformat()
    n = 0
    for s in students:
        for r in list_logs(s["uid"]):
            if str(r.get("date") or "")[:10] == today_str:
                n += 1
                break
    return n


def _extract_keywords_from_bsr(bsr_text: str) -> tuple[list[str], list[str]]:
    """BSR 원문에서 NCS_DB·GLOSSARY 키워드 추출. (ncs_keywords, glossary_terms) 반환."""
    if not bsr_text:
        return [], []
    text_lower = bsr_text.lower()
    ncs_found: list[str] = []
    glossary_found: list[str] = []
    for unit, meta in NCS_DB.items():
        for kw in meta.get("keywords", []):
            if kw in bsr_text or (len(kw) >= 2 and kw.lower() in text_lower):
                ncs_found.append(kw)
    for term in GLOSSARY:
        if term in bsr_text:
            glossary_found.append(term)
    return ncs_found, glossary_found


def _resolve_ncs_unit(unit_or_code: str) -> str:
    """능력단위명 또는 코드를 정규 단위명으로 변환. 데이터 매칭 정확도 향상."""
    if not unit_or_code:
        return ""
    if unit_or_code in NCS_DB:
        return unit_or_code
    for u, meta in NCS_DB.items():
        if meta.get("code") == unit_or_code:
            return u
    return unit_or_code


def _filter_terms_for_unit(used_terms: list[str], unit_type: str) -> list[str]:
    """단위별 관련 키워드만 필터링."""
    sets = {
        "plc": {"래더", "PLC", "시퀀스", "로직", "접점", "모터", "입출력", "시운전"},
        "solder": {"납땜", "솔더링", "쇼트", "PCB", "극성", "저항", "콘덴서", "인두"},
        "safety": {"접지", "보호구", "LOTO", "ELB", "차단", "MCB", "메거", "절연"},
    }
    s = sets.get(unit_type, set())
    return [t for t in used_terms if t in s][:3]


def _evaluate_seungwa_reflection(bsr_logs: list[dict]) -> tuple[str, str]:
    """
    BSR 로그에서 [성과] 부분을 추출해 성찰 수준(높음/보통/낮음)을 평가.
    반환: (수준, 코멘트).
    """
    high_words = {"깨달음", "성찰", "과정", "이유", "개선", "다음에는", "배운", "어려웠던", "스스로", "이해", "알게", "생각", "판단", "고민"}
    medium_words = {"확인", "점검", "수행", "적용", "이해함", "배웠"}
    low_patterns = ["했다", "됐다", "완료", "끝냄", "했다."]

    scores: list[int] = []
    extracts: list[str] = []
    for row in bsr_logs:
        bsr = (row.get("bsr") or "").strip()
        m = re.search(r"\[성과\]\s*(.*?)(?=\[|$)", bsr, re.DOTALL)
        if m:
            seg = m.group(1).strip()
            extracts.append(seg)
            score = 0
            seg_lower = seg.lower()
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
        level, comment = "높음", "학생이 과정·이유·개선점을 구체적으로 서술하여 메타인지적 성찰 수준이 높습니다."
    elif avg >= 1.5:
        level, comment = "보통", "기본적인 수행 중심 기술에 일부 성찰 요소가 포함되어 있습니다."
    else:
        level, comment = "낮음", "결과 중심의 간단한 서술 위주이며, 성찰 키워드 보완을 권장합니다."
    return level, comment


def _extract_seungwa_from_bsr(bsr_text: str) -> str:
    """BSR 텍스트에서 [성과] 구간만 추출."""
    if not bsr_text:
        return ""
    m = re.search(r"\[성과\]\s*(.*?)(?=\[|$)", str(bsr_text), re.DOTALL)
    return (m.group(1).strip() if m else "").strip()


def _log_competency_scores(bsr_text: str) -> dict[str, float]:
    """BSR 텍스트에서 역량 차원 점수 추출 (구체성, 전문용어, 안전, 성찰)."""
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


def _evaluate_seungwa_level(seungwa_text: str) -> str:
    """[성과] 답변의 성찰 수준을 높음/보통/낮음으로 평가 (휴리스틱 기반)."""
    t = (seungwa_text or "").strip()
    if not t or len(t) < 5:
        return "낮음"
    # 성찰적·메타인지적 표현 (높음)
    high_keywords = ["깨달", "성찰", "과정", "이유", "개선", "다음에는", "배운 점", "어려웠던", "스스로", "생각", "이해", "교훈", "반성", "차후"]
    # 보통 수준 표현
    mid_keywords = ["확인", "점검", "수행", "완료", "작동", "연결", "설정"]
    high_cnt = sum(1 for k in high_keywords if k in t)
    mid_cnt = sum(1 for k in mid_keywords if k in t)
    length_bonus = 1 if len(t) >= 50 else (0.5 if len(t) >= 25 else 0)
    score = high_cnt * 2 + mid_cnt * 0.5 + length_bonus
    if score >= 3:
        return "높음"
    if score >= 1:
        return "보통"
    return "낮음"


def _make_seuteuk_keyword_fallback(uid: str, logs: list[dict]) -> str:
    """Gemini 세특 생성 실패 시 사용하는 키워드 기반 요약."""
    logs_sorted = logs[:10]
    units = [_resolve_ncs_unit(row.get("ncs_unit", "")) for row in logs_sorted]
    unit_counter = Counter(u for u in units if u)
    top_units = ", ".join(f"{format_ncs_unit(u)}({c}회)" for u, c in unit_counter.most_common(3))
    total_cnt = len(logs_sorted)

    # BSR 전체 텍스트에서 키워드 추출 (학생이 실제 사용한 용어 반영)
    all_bsr = " ".join((r.get("bsr") or "") for r in logs_sorted)
    ncs_kw, gl_kw = _extract_keywords_from_bsr(all_bsr)
    used_terms = list(dict.fromkeys(ncs_kw + gl_kw))[:12]

    safety_cnt = sum(_resolve_ncs_unit(r.get("ncs_unit") or "") == "전기안전" for r in logs_sorted)
    plc_cnt = sum("PLC" in _resolve_ncs_unit(r.get("ncs_unit") or "") for r in logs_sorted)
    solder_cnt = sum(_resolve_ncs_unit(r.get("ncs_unit") or "") == "전자부품장착" for r in logs_sorted)

    parts: list[str] = []
    parts.append(
        f"{_student_label(uid)}은(는) 한 학기 동안 전공 실습에 성실히 참여하여 "
        f"{top_units} 영역에서 총 {total_cnt}회 이상의 실습 활동을 수행하였다."
    )

    plc_kw = _filter_terms_for_unit(used_terms, "plc")
    solder_kw = _filter_terms_for_unit(used_terms, "solder")
    safety_kw = _filter_terms_for_unit(used_terms, "safety")

    if plc_cnt:
        if plc_kw:
            kw_str = "·".join(plc_kw[:3])
            base = f"PLC 제어 실습에서는 {kw_str} 기기를 활용하여 래더 로직 작성·입출력 결선·시운전 등 핵심 공정을 수행하며"
        else:
            base = "PLC 제어 실습에서는 입출력 결선 및 시퀀스 제어를 단계적으로 수행하며"
        parts.append(base + " 오동작 원인을 스스로 분석하고 수정하는 경험을 쌓았다.")
    if solder_cnt:
        if solder_kw:
            kw_str = "·".join(solder_kw[:3])
            base = f"전자부품장착 실습에서는 {kw_str} 기기를 활용하여 부품 극성 확인·납땜 품질·쇼트 여부 점검 등 핵심 공정을 수행하며"
        else:
            base = "전자부품장착 관련 실습에서는 회로도에 따라 부품 극성을 확인하고 납땜 품질과 쇼트 여부를 점검하는 등"
        parts.append(base + " 기본기 향상에 노력하였다.")
    if safety_cnt:
        if safety_kw:
            kw_str = "·".join(safety_kw[:3])
            base = f"전기안전 영역에서는 {kw_str} 등 기초 안전수칙(전원 차단·보호구 착용)을 준수하며"
        else:
            base = "전기안전 영역에서는 작업 전 위험요인을 사전에 파악하고 전원 차단·보호구 착용 등"
        parts.append(base + " 위험요인을 사전에 파악하려는 태도를 보였다.")

    text = " ".join(parts)
    return textwrap.shorten(text, width=480, placeholder=" …")


def _make_seuteuk(uid: str, logs: list[dict]) -> str:
    if not logs:
        return "선택한 기간에 해당 학생의 실습 기록이 없습니다."

    api_key = resolve_google_api_key()
    gemini_text = generate_seuteuk_from_bsr_logs(
        logs, _student_label(uid), api_key=api_key
    )
    if gemini_text and len(gemini_text.strip()) >= 40:
        return textwrap.shorten(gemini_text.strip(), width=520, placeholder=" …")

    return _make_seuteuk_keyword_fallback(uid, logs)


def _collect_class_overview(students: list[dict]) -> dict:
    """학급 전체 통계와 학생별 요약 데이터를 한 번에 집계."""
    all_logs_flat: list[dict] = []
    total_logs = 0
    prog_sum = 0
    prog_cnt = 0
    rows: list[dict] = []
    heat_rows: list[dict] = []
    all_units_set: set[str] = set()

    for s in students:
        uid = s["uid"]
        logs = list_logs(uid)
        total_logs += len(logs)
        all_logs_flat.extend(logs)
        prog = seed_progress_if_missing(uid, DEFAULT_NCS_PROGRESS)
        prog_sum += sum(prog.values())
        prog_cnt += len(prog)

        refl_scores = [
            _log_competency_scores(r.get("bsr") or "").get("성찰", 0.0) for r in logs
        ]
        avg_refl = round(sum(refl_scores) / len(refl_scores), 2) if refl_scores else 0.0
        rows.append(
            {
                "학생": _student_label(uid),
                "일지수": len(logs),
                "성찰(평균)": avg_refl,
            }
        )
        for r in logs:
            u = _resolve_ncs_unit(r.get("ncs_unit", ""))
            if u:
                all_units_set.add(u)

    avg_prog = round(prog_sum / max(prog_cnt, 1), 1) if prog_cnt else 0
    all_units = sorted(all_units_set)
    for s in students:
        uid = s["uid"]
        logs = list_logs(uid)
        counter = {u: 0 for u in all_units}
        for r in logs:
            u = _resolve_ncs_unit(r.get("ncs_unit", ""))
            if u in counter:
                counter[u] += 1
        row = {"학생": _student_label(uid)}
        row.update(counter)
        heat_rows.append(row)

    df = pd.DataFrame(rows)

    return {
        "all_logs_flat": all_logs_flat,
        "total_logs": total_logs,
        "avg_prog": avg_prog,
        "df_summary": df,
        "heat_rows": heat_rows,
        "all_units": all_units,
    }


def _style_reflection_low(row: pd.Series) -> list[str]:
    styles: list[str] = []
    for _ in row.index:
        if row.get("성찰(평균)", 99) < 2.0:
            styles.append("background-color: #fff9c4; color: #334155")
        else:
            styles.append("")
    return styles


# ═══════════════════════════════════════════════════════════════════
# 탭1 · 요약: 핵심 지표 + 제출 현황 그리드
# ═══════════════════════════════════════════════════════════════════
def _render_tab_overview(students: list[dict], overview: dict) -> None:
    """[종합 현황] 탭: 진도·제출 흐름을 한눈에 확인합니다."""
    all_logs_flat: list[dict] = overview["all_logs_flat"]
    avg_prog: float = overview["avg_prog"]
    total_logs: int = overview["total_logs"]

    with st.container(border=True):
        st.subheader("핵심 지표", divider="gray")
        st.caption("학급 단위 지표가 조회되었습니다. 세부 내용은 [실습 일지 정밀 점검] 메뉴에서 확인해 주십시오.")
        today_submitters = _count_submissions_today(students)
        ncs_ratios = [r.get("ncs_term_ratio") or 0 for r in all_logs_flat]
        avg_ncs_ratio = round(sum(ncs_ratios) / max(len(ncs_ratios), 1), 1)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "학급 전체 평균 진도율",
                f"{avg_prog}%",
                help=f"학생 {len(students)}명 × NCS 단위별 평균 진행률",
            )
        with m2:
            st.metric(
                "오늘 일지 제출 인원",
                f"{today_submitters} / {len(students)}명",
                help="금일 1건 이상 일지를 저장한 도제생 수",
            )
        with m3:
            st.metric(
                "누적 실습 일지",
                f"{total_logs}건",
                help="전체 학생이 저장한 일지 합계",
            )
        with m4:
            st.metric(
                "NCS 용어 변환률",
                f"{avg_ncs_ratio}%",
                help="구어체 대비 NCS 표준 용어 사용 비율 평균",
            )

    with st.container(border=True):
        st.subheader("제출 현황판", divider="gray")
        st.caption(
            f"실전 테스트 기간 {TEST_PERIOD_START.strftime('%Y-%m-%d')}(월) ~ "
            f"{TEST_PERIOD_END.strftime('%Y-%m-%d')}(금) 평일 기준입니다. "
            "●: 일지 1건 이상 제출, ·: 미제출, –: 아직 도래하지 않은 날짜입니다."
        )
        att_df = _build_test_period_attendance(students)
        st.dataframe(att_df, width="stretch", hide_index=True, height=420)


# ═══════════════════════════════════════════════════════════════════
# 대시보드 심화: 활동 요약 · 히트맵 · AI 가이드 (정밀 점검 메뉴 상단에 배치)
# ═══════════════════════════════════════════════════════════════════
def _render_dashboard_deep_analytics(students: list[dict], overview: dict) -> None:
    """학생별 요약 표·히트맵·AI 교수학습 가이드."""
    df: pd.DataFrame = overview["df_summary"]
    heat_rows: list[dict] = overview["heat_rows"]

    # ─── 3. 학생별 활동 요약 ───
    with st.container(border=True):
        st.subheader("학생별 활동 요약", divider="gray")
        st.caption("성찰(평균) 점수 2.0 미만인 학생은 노란색으로 강조 표시됩니다.")
        try:
            styled_df = df.style.apply(_style_reflection_low, axis=1)
            st.dataframe(styled_df, width="stretch", hide_index=True)
        except Exception:
            st.dataframe(df, width="stretch", hide_index=True)

        if not df.empty:
            df_chart = df.copy()
            df_chart["_ord"] = (
                df_chart["학생"].str.extract(r"(\d+)", expand=False).fillna(999).astype(int)
            )
            df_chart = df_chart.sort_values("_ord", ascending=True).drop(columns=["_ord"])
            fig = px.bar(
                df_chart,
                x="학생",
                y="일지수",
                color_discrete_sequence=[P.get("primary", "#0f766e")],
                category_orders={"학생": df_chart["학생"].tolist()},
            )
            fig.update_layout(
                margin=dict(l=40, r=40, t=20, b=80),
                xaxis_tickangle=-45,
                showlegend=False,
                paper_bgcolor="rgba(255,255,255,0)",
                plot_bgcolor="rgba(255,255,255,0)",
                height=320,
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("작성된 실습일지가 조회되지 않았습니다.", icon=":material/info:")

    # ─── 4. 직무 도달도 히트맵 ───
    with st.container(border=True):
        st.subheader("직무 도달도 히트맵 (핵심 NCS 단위)", divider="gray")
        st.caption(
            f"전체 학생({STUDENT_COUNT}명)과 주요 능력단위별 실습 일지 빈도입니다. "
            "색이 옅은 칸은 해당 단위 실습이 적어 직무 경험이 소외되었을 수 있음을 시사합니다."
        )
        h_rows, h_cols, h_z = _heatmap_frequency_matrix(students)
        fig_hm = go.Figure(
            data=go.Heatmap(
                z=h_z,
                x=h_cols,
                y=h_rows,
                colorscale=[
                    [0.0, "#f1f5f9"],
                    [0.35, "#bae6fd"],
                    [0.65, "#38bdf8"],
                    [1.0, P["primary"]],
                ],
                colorbar=dict(title="일지 수"),
                hovertemplate="학생: %{y}<br>단위: %{x}<br>실습 횟수: %{z}<extra></extra>",
            )
        )
        fig_hm.update_layout(
            margin=dict(l=100, r=40, t=20, b=120),
            xaxis_tickangle=-35,
            height=max(380, 28 * len(h_rows)),
            paper_bgcolor="rgba(255,255,255,0)",
            plot_bgcolor="rgba(255,255,255,0)",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_hm, width="stretch")

        with st.expander("그 외 NCS 단위까지 포함한 상세 표", expanded=False):
            if heat_rows:
                heat_df = pd.DataFrame(heat_rows).set_index("학생")
                heat_df = heat_df.rename(columns={c: format_ncs_unit(c) for c in heat_df.columns})
                try:
                    from matplotlib.colors import LinearSegmentedColormap

                    _cmap = LinearSegmentedColormap.from_list(
                        "ncs_accent",
                        ["#f8fafc", P["accent_soft"], P["accent"], P["primary"]],
                        N=128,
                    )
                    styled = heat_df.style.background_gradient(cmap=_cmap, axis=None)
                    st.dataframe(styled, width="stretch")
                except Exception:
                    st.dataframe(heat_df, width="stretch")
            else:
                st.caption("아직 일지에 기록된 NCS 단위가 없습니다.")

    # ─── 5. 약점 자동 추출 + AI 가이드 ───
    with st.container(border=True):
        st.subheader("AI 기반 교수학습 가이드", divider="gray")
        st.caption(
            "BSR 키워드 기반 레이더(설계·제작·계측·제어·안전)로 전원 점수를 집계하고, "
            "30점 미만이거나 나머지 네 영역 평균 대비 20% 이상 낮은 축을 자동 추출합니다."
        )
        radar_rows: list[dict] = []
        flag_cases: list[dict] = []
        for s in students:
            uid = s["uid"]
            logs = list_logs(uid)
            axes, vals = radar_scores_from_logs(logs)
            row: dict = {"학생": _student_label(uid), "uid": uid}
            for a, v in zip(axes, vals):
                row[a] = v
            radar_rows.append(row)
            for w in extract_weak_radar_dimensions(vals):
                flag_cases.append(
                    {
                        "student_label": _student_label(uid),
                        "uid": uid,
                        "axis": w["axis"],
                        "reason": w["reason"],
                        "value": round(float(w["value"]), 2),
                        "others_avg": round(float(w["others_avg"]), 2),
                        "scores": dict(zip(axes, [round(float(x), 2) for x in vals])),
                    }
                )
        if radar_rows:
            tbl_radar = pd.DataFrame(radar_rows)
            st.markdown("**전체 학생 레이더 점수**")
            st.dataframe(tbl_radar, width="stretch", hide_index=True)
            plot_df = tbl_radar.set_index("학생")[RADAR_AXES]
            fig_radar = px.imshow(
                plot_df,
                labels={"x": "역량 축", "y": "학생", "color": "점수"},
                aspect="auto",
                color_continuous_scale="Blues",
                zmin=0,
                zmax=100,
            )
            fig_radar.update_layout(height=max(240, min(520, 32 * len(plot_df))))
            st.plotly_chart(fig_radar, width="stretch")
        else:
            st.info("등록된 학생이 없어 레이더 데이터를 표시할 수 없습니다.", icon=":material/info:")

        if flag_cases:
            st.markdown("**자동 추출: 지도가 필요한 약점 축**")
            disp_weak = pd.DataFrame(
                [
                    {
                        "학생": c["student_label"],
                        "uid": c["uid"],
                        "약점 축": c["axis"],
                        "사유": c["reason"],
                        "점수": c["value"],
                        "타 영역 평균": c["others_avg"],
                    }
                    for c in flag_cases
                ]
            )
            st.dataframe(disp_weak, width="stretch", hide_index=True)
            api_k = resolve_google_api_key()
            if not api_k:
                st.warning(
                    "Gemini 가이드 생성을 위해 `.streamlit/secrets.toml`에 `GOOGLE_API_KEY`를 설정하시기 바랍니다.",
                    icon=":material/key:",
                )
            if st.button(
                "Gemini 교수학습 가이드 생성",
                key="teacher_radar_guidance_btn",
                icon=":material/auto_awesome:",
            ):
                with st.spinner("교수학습 가이드를 생성하는 중입니다..."):
                    guide = generate_teacher_learning_guidance(flag_cases, api_key=api_k)
                if guide:
                    st.markdown(guide)
                else:
                    st.warning(
                        "가이드 생성에 실패하였습니다. API 키 또는 할당량을 확인해 주십시오.",
                        icon=":material/warning:",
                    )
        else:
            st.success(
                "자동 추출 기준에 해당하는 약점 축이 존재하지 않습니다.",
                icon=":material/check_circle:",
            )


def _log_sort_key_asc(row: dict) -> tuple[datetime.date, int]:
    d = _parse_log_date(row.get("date"))
    if d is None:
        return datetime.date(2099, 12, 31), int(row.get("id") or 0)
    return d, int(row.get("id") or 0)


def _journal_expander_title(row: dict) -> str:
    """날짜 + 핵심 성과(또는 대체 요약) — Expander 라벨."""
    date_s = str(row.get("date") or "—")
    seungwa = _extract_seungwa_from_bsr(str(row.get("bsr") or ""))
    one_line = seungwa.replace("\n", " ").strip()
    if one_line:
        head = textwrap.shorten(one_line, width=48, placeholder="…")
        return f"{date_s} · 핵심 성과: {head}"
    ncs = format_ncs_unit(_resolve_ncs_unit(row.get("ncs_unit", "") or ""))
    snip = _bsr_preview_snippet(row.get("bsr") or "", max_len=36)
    if ncs and snip:
        return f"{date_s} · {ncs} — {snip}"
    if ncs:
        return f"{date_s} · {ncs}"
    return f"{date_s} · 실습 일지"


def _render_tab_student_journals(students: list[dict]) -> None:
    """[학생별 상세보기] 탭: 선택 학생 일지를 날짜순·Expander로 정리."""
    with st.container(border=True):
        st.subheader("학생별 실습일지 목록", divider="gray")
        st.caption(
            "조회할 학생을 선택해 주십시오. 선택된 학생이 작성한 일지만 오래된 날짜부터 순서대로 조회되었습니다. "
            "항목 제목을 누르면 상세 내용이 펼쳐집니다."
        )
        if not students:
            st.info("등록된 학생이 조회되지 않았습니다.", icon=":material/info:")
            return
        opts = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
        sel_uid = st.selectbox(
            "조회할 학생",
            options=list(opts.keys()),
            format_func=lambda u: opts[u],
            key="teacher_tab2_student_logs",
        )
        logs = list_logs(sel_uid)
        sorted_logs = sorted(logs, key=_log_sort_key_asc)
        if not sorted_logs:
            st.info(
                "선택한 학생의 저장된 실습일지가 조회되지 않았습니다.",
                icon=":material/info:",
            )
            return
        st.success(
            f"총 {len(sorted_logs)}건의 일지가 날짜순으로 조회되었습니다.",
            icon=":material/check_circle:",
        )

        with st.container(border=True):
            st.markdown(
                "**일지 개별 삭제** · 연습·오기입 건만 골라 삭제할 수 있습니다. "
                "삭제 후에는 복구되지 않습니다."
            )
            del_opts = [(row.get("id"), _journal_expander_title(row)) for row in sorted_logs]
            jcol1, jcol2 = st.columns([4, 1], gap="small")
            with jcol1:
                del_pick = st.selectbox(
                    "삭제할 일지 선택",
                    options=[o[0] for o in del_opts],
                    format_func=lambda x: next((o[1] for o in del_opts if o[0] == x), str(x)),
                    key=f"teacher_journal_del_sel_{sel_uid}",
                    label_visibility="collapsed",
                )
            with jcol2:
                st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
                if st.button(
                    "삭제 확인…",
                    key=f"teacher_journal_del_btn_{sel_uid}",
                    width="stretch",
                    icon=":material/help_outline:",
                    help="백업 CSV와 최종 확인 창이 열립니다.",
                ):
                    if del_pick is not None:
                        row_del = next((r for r in sorted_logs if r.get("id") == del_pick), None)
                        if row_del:
                            st.session_state["_tch_dlg_journal_one"] = {
                                "sel_uid": sel_uid,
                                "row": copy_log_row(row_del),
                            }
                            st.rerun()

        for row in sorted_logs:
            label = _journal_expander_title(row)
            with st.expander(label, expanded=False):
                ncs_d = format_ncs_unit(_resolve_ncs_unit(row.get("ncs_unit", "") or ""))
                st.caption(f"NCS 능력단위: {ncs_d or '—'}")
                if row.get("image_note"):
                    st.markdown(
                        f"**증거 사진 메모**  \n{html.escape(str(row['image_note']))}"
                    )
                bsr_html = render_bsr_highlighted(str(row.get("bsr") or ""))
                st.markdown(
                    f"<div class='report-card-inner'>{bsr_html}</div>",
                    unsafe_allow_html=True,
                )


def _render_tab_data_administration(students: list[dict]) -> None:
    """[데이터 행정] 탭: 일괄 내보내기, 시스템 초기화 안내, 연구자 로그."""
    with st.container(border=True):
        st.subheader("연구·행정 데이터 일괄 내보내기", divider="gray")
        st.caption(
            "일지별 증거 사진 메모, 학생 성찰(BSR), 휴리스틱 역량 점수, 교사 확정 종합의견을 "
            "CSV 또는 Excel 파일로 내려받으실 수 있습니다. 저장 위치와 개인정보 취급 규정을 확인해 주십시오."
        )
        export_rows: list[dict[str, object]] = []
        for s in students:
            suid = s["uid"]
            pc_row = get_portfolio_comment(suid)
            t_teacher = (pc_row.get("comment_text") or "") if pc_row else ""
            t_conf = "Y" if (pc_row and int(pc_row.get("is_confirmed") or 0)) else "N"
            for row in list_logs(suid):
                bsr_t = str(row.get("bsr") or "")
                scores = _log_competency_scores(bsr_t)
                export_rows.append(
                    {
                        "학생UID": suid,
                        "일지ID": row.get("id", ""),
                        "날짜": row.get("date", ""),
                        "NCS단위": row.get("ncs_unit", ""),
                        "증거사진_메모": row.get("image_note") or "",
                        "학생성찰_BSR": bsr_t,
                        "AI분석_역량점수_JSON": json.dumps(scores, ensure_ascii=False),
                        "교사최종의견": t_teacher,
                        "교사의견_확정여부": t_conf,
                    }
                )
        if export_rows:
            df_exp = pd.DataFrame(export_rows)
            csv_bytes = df_exp.to_csv(index=False).encode("utf-8-sig")
            c_dl1, c_dl2 = st.columns(2)
            with c_dl1:
                st.download_button(
                    "CSV 다운로드 (UTF-8 BOM, Excel 호환)",
                    data=csv_bytes,
                    file_name="research_validity_export.csv",
                    mime="text/csv",
                    key="research_validity_csv_tab3",
                    width="stretch",
                    icon=":material/download:",
                )
            with c_dl2:
                try:
                    buf = io.BytesIO()
                    with pd.ExcelWriter(buf, engine="openpyxl") as wr:
                        df_exp.to_excel(wr, index=False, sheet_name="data")
                    st.download_button(
                        "Excel 다운로드 (.xlsx)",
                        data=buf.getvalue(),
                        file_name="research_validity_export.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key="research_validity_xlsx_tab3",
                        width="stretch",
                        icon=":material/download:",
                    )
                except Exception:
                    st.caption(
                        "Excel 형식으로 저장하시려면 `openpyxl` 패키지 설치 여부를 확인해 주십시오."
                    )
        else:
            st.info(
                "내보낼 실습일지가 조회되지 않았습니다. 데이터 존재 여부를 확인해 주십시오.",
                icon=":material/info:",
            )

    with st.container(border=True):
        st.subheader("시스템 초기화 (실습 일지)", divider="gray")
        st.warning(
            "실습 일지를 삭제하면 복구할 수 없습니다. 실행 전에 반드시 [데이터 일괄 내보내기]로 백업하시기 바랍니다.",
            icon=":material/warning:",
        )
        st.caption(
            "학생 본인은 [실습 이력 관리] 화면 상단에서도 일지를 삭제할 수 있습니다. "
            "교사 화면에서는 선택 학생 또는 전체 학생에 대해 일괄 삭제를 수행하실 수 있습니다."
        )
        if students:
            opts = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
            clear_one = st.selectbox(
                "초기화 대상 학생 (개별)",
                options=list(opts.keys()),
                format_func=lambda u: opts[u],
                key="teacher_clear_logs_one_student",
            )
            agree_one = st.checkbox(
                "선택한 학생의 모든 실습 일지를 삭제함에 동의합니다.",
                key="teacher_clear_one_agree",
            )
            if st.button(
                "삭제 확인 화면 열기 (선택 학생 전체)",
                key="teacher_clear_one_btn",
                disabled=not agree_one,
                icon=":material/warning:",
            ):
                rows_snap = [copy_log_row(r) for r in list_logs(clear_one)]
                st.session_state["_tch_dlg_clear_one"] = {"uid": clear_one, "rows": rows_snap}
                st.rerun()

            st.divider()
            st.markdown("**전체 학생 일괄 삭제**")
            agree_all_a = st.checkbox(
                "전체 학생의 실습 일지를 삭제함에 동의합니다.",
                key="teacher_clear_all_agree_a",
            )
            agree_all_b = st.checkbox(
                "삭제된 데이터는 복구할 수 없음을 확인하였습니다.",
                key="teacher_clear_all_agree_b",
            )
            if st.button(
                "삭제 확인 화면 열기 (전체 학생)",
                key="teacher_clear_all_btn",
                disabled=not (agree_all_a and agree_all_b),
                icon=":material/warning:",
            ):
                snap: dict[str, list[dict[str, Any]]] = {}
                for s in students:
                    su = s["uid"]
                    snap[su] = [copy_log_row(r) for r in list_logs(su)]
                st.session_state["_tch_dlg_clear_all"] = snap
                st.rerun()

    with st.container(border=True):
        st.subheader("학생 이력서·표지 데이터 삭제", divider="gray")
        st.caption(
            "이력서 관리 화면에 저장된 **표지·학력·경력 등**만 삭제합니다. "
            "실습 일지는 위 메뉴 또는 [학생별 실습일지 목록]에서 따로 삭제하세요."
        )
        if students:
            prof_opts = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
            prof_uid = st.selectbox(
                "이력서 데이터를 비울 학생",
                options=list(prof_opts.keys()),
                format_func=lambda u: prof_opts[u],
                key="teacher_clear_profile_student",
            )
            agree_prof = st.checkbox(
                "선택 학생의 저장된 이력서(DB)를 삭제함에 동의합니다.",
                key="teacher_clear_profile_agree",
            )
            if st.button(
                "삭제 확인 화면 열기 (이력서)",
                key="teacher_clear_profile_btn",
                disabled=not agree_prof,
                icon=":material/warning:",
            ):
                st.session_state["_tch_dlg_profile"] = prof_uid
                st.rerun()

    with st.container(border=True):
        st.subheader("연구자 성찰 로그", divider="gray")
        st.caption(
            "지도 경험과 지원 효과를 기록하시면 질적 연구 데이터로 활용하실 수 있습니다."
        )
        with st.form(key="researcher_log_form_tab3", clear_on_submit=True):
            r_date = st.date_input(
                "기록일",
                value=app_today(),
                min_value=TEST_PERIOD_START,
                max_value=TEST_PERIOD_END,
                key="researcher_log_date_tab3",
            )
            r_note = st.text_area(
                "성찰 내용 (지도 경험, 지원 효과, 발견된 패턴 등)",
                placeholder="예: 오늘 해당 학생의 BSR 구조화가 전주보다 구체적이었음. 역질문 답변이 해결 과정을 잘 서술함.",
                height=120,
                key="researcher_log_note_tab3",
            )
            if st.form_submit_button("연구자 로그 저장", icon=":material/save:"):
                if r_note and r_note.strip():
                    add_researcher_log(log_date=str(r_date), note=r_note.strip())
                    st.success(
                        "연구자 성찰 로그가 저장되었습니다.",
                        icon=":material/check_circle:",
                    )
                else:
                    st.warning(
                        "성찰 내용을 입력해 주십시오.",
                        icon=":material/warning:",
                    )
        r_logs = list_researcher_logs()
        if r_logs:
            with st.expander(
                "저장된 연구자 로그 보기",
                expanded=False,
                icon=":material/history:",
            ):
                for r in r_logs[:20]:
                    st.markdown(f"**{r.get('log_date', '')}**")
                    st.write((r.get("note") or "").replace("\n", " "))
                    st.divider()


# ═══════════════════════════════════════════════════════════════════
# 메뉴 2. 실습 일지 정밀 점검
# ═══════════════════════════════════════════════════════════════════
def _render_log_inspection_view(students: list[dict], overview: dict) -> None:
    """좌측 사이드바 「실습 일지 정밀 점검」 본문."""
    all_logs_flat: list[dict] = overview["all_logs_flat"]

    # ─── 역량 성장 비교 ───
    with st.container(border=True):
        st.subheader("역량 성장 비교 (스캐폴딩 효과)", divider="gray")
        st.caption("최초 3개 일지와 최근 3개 일지를 비교하여 성찰의 성장을 시각화합니다.")
        if students:
            radar_uid = st.selectbox(
                "학생 선택",
                options=[s["uid"] for s in students],
                format_func=lambda u: f"{_student_label(u)} ({u})",
                key="radar_student",
            )
            radar_logs = list_logs(radar_uid)
            if len(radar_logs) >= 2:
                reversed_logs = list(reversed(radar_logs))
                first3 = reversed_logs[:3]
                recent3 = radar_logs[:3]
                dims = ["구체성", "전문용어", "안전", "성찰"]

                def avg_scores(log_list):
                    if not log_list:
                        return [0] * 4
                    by_dim = {d: [] for d in dims}
                    for row in log_list:
                        s = _log_competency_scores(row.get("bsr") or "")
                        for d in dims:
                            by_dim[d].append(s.get(d, 0))
                    return [sum(by_dim[d]) / max(len(by_dim[d]), 1) for d in dims]

                first_vals = avg_scores(first3)
                recent_vals = avg_scores(recent3)
                fig_radar = go.Figure()
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=first_vals + [first_vals[0]],
                        theta=dims + [dims[0]],
                        fill="toself",
                        name="최초 3개 일지",
                        line={"color": P.get("accent", "#14b8a6")},
                    )
                )
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=recent_vals + [recent_vals[0]],
                        theta=dims + [dims[0]],
                        fill="toself",
                        name="최근 3개 일지",
                        line={"color": P.get("primary", "#0f766e")},
                    )
                )
                fig_radar.update_layout(
                    polar={"radialaxis": {"visible": True, "range": [0, 5]}},
                    showlegend=True,
                    height=400,
                    margin=dict(l=80, r=80),
                )
                st.plotly_chart(fig_radar, width="stretch")
            else:
                st.info(
                    "역량 성장 비교는 일지가 2건 이상 저장된 경우에 표시됩니다.",
                    icon=":material/info:",
                )

    # ─── BSR 구조화 상세 ───
    with st.container(border=True):
        st.subheader("실습일지 BSR 구조화 상세", divider="gray")
        st.caption("[배경] [해결] [성과] 구간별 시각화로 실무 중심 실체를 확인하실 수 있습니다.")
        if students:
            t_options = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
            t_uid = st.selectbox(
                "학생 선택",
                options=list(t_options.keys()),
                format_func=lambda u: t_options[u],
                key="bsr_student_select",
            )
            t_logs = list_logs(t_uid)
            if t_logs:
                t_detail_opts = [
                    (
                        r.get("id"),
                        f"#{r.get('id')} [{r.get('date','')}] {format_ncs_unit(r.get('ncs_unit',''))}",
                    )
                    for r in t_logs
                ]
                t_sel_id = st.selectbox(
                    "일지 선택",
                    options=[o[0] for o in t_detail_opts],
                    format_func=lambda x: next((o[1] for o in t_detail_opts if o[0] == x), str(x)),
                    key="bsr_log_select",
                )
                t_row = next((r for r in t_logs if r.get("id") == t_sel_id), None)
                if t_row and t_row.get("bsr"):
                    bsr_html = render_bsr_highlighted(str(t_row["bsr"]))
                    st.markdown(
                        f"<div class='report-card-inner'>{bsr_html}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "선택한 학생의 저장된 실습일지가 조회되지 않았습니다.",
                    icon=":material/info:",
                )

    # ─── 성찰 키워드 분석 ───
    with st.container(border=True):
        st.subheader("성찰 키워드 분석", divider="gray")
        REFLECTION_KEYWORDS = [
            "깨달음", "해결", "다음에는", "배운", "이해", "개선",
            "어려웠던", "스스로", "성찰", "과정", "이유", "알게",
        ]
        REFLECTION_TIMELINE_KW = ["깨달음", "해결", "다음에는"]

        st.markdown("##### 성찰 성장 타임라인")
        st.caption("주차별 성찰 키워드(깨달음, 해결, 다음에는) 사용 횟수 추이를 확인하실 수 있습니다.")
        if all_logs_flat:
            week_counts: dict[str, dict[str, int]] = defaultdict(
                lambda: {k: 0 for k in REFLECTION_TIMELINE_KW}
            )
            for row in all_logs_flat:
                bsr = (row.get("bsr") or "").strip()
                d_str = row.get("date", "")
                if not d_str:
                    continue
                try:
                    dt = datetime.datetime.strptime(d_str, "%Y-%m-%d")
                    wk = f"{dt.year}-W{dt.isocalendar()[1]:02d}"
                except (ValueError, TypeError):
                    wk = d_str[:7] if len(d_str) >= 7 else d_str
                for kw in REFLECTION_TIMELINE_KW:
                    if kw in bsr:
                        week_counts[wk][kw] += 1
            if week_counts:
                weeks_sorted = sorted(week_counts.keys())
                tl_data = [{"주차": w, **week_counts[w]} for w in weeks_sorted]
                df_tl = pd.DataFrame(tl_data)
                fig_tl = go.Figure()
                for kw in REFLECTION_TIMELINE_KW:
                    fig_tl.add_trace(
                        go.Scatter(
                            x=df_tl["주차"], y=df_tl[kw],
                            name=kw, mode="lines+markers", line=dict(width=2),
                        )
                    )
                fig_tl.update_layout(
                    height=280, margin=dict(l=50, r=30, t=30, b=80),
                    xaxis_tickangle=-45,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                )
                st.plotly_chart(fig_tl, width="stretch")
            else:
                st.info("주차별 데이터가 조회되지 않았습니다.", icon=":material/info:")
        else:
            st.info("분석 대상 실습일지가 조회되지 않았습니다.", icon=":material/info:")

        st.markdown("##### 성찰 키워드 빈도 (날짜별)")
        st.caption("전체 일지에서 메타인지적 성찰 키워드 사용 빈도를 확인하실 수 있습니다.")
        if all_logs_flat:
            date_counts: dict[str, dict[str, int]] = defaultdict(
                lambda: {k: 0 for k in REFLECTION_KEYWORDS}
            )
            for row in all_logs_flat:
                bsr = (row.get("bsr") or "").strip()
                d = row.get("date", "")
                if not d:
                    continue
                for kw in REFLECTION_KEYWORDS:
                    if kw in bsr:
                        date_counts[d][kw] += 1
            dates_sorted = sorted(date_counts.keys())
            if dates_sorted:
                chart_data = []
                for d in dates_sorted:
                    row_data = {"날짜": d}
                    for kw in REFLECTION_KEYWORDS:
                        row_data[kw] = date_counts[d][kw]
                    chart_data.append(row_data)
                df_kw = pd.DataFrame(chart_data)
                fig_kw = px.bar(
                    df_kw,
                    x="날짜",
                    y=REFLECTION_KEYWORDS,
                    barmode="stack",
                    color_discrete_sequence=[
                        P.get("primary", "#0f766e"), P.get("accent", "#14b8a6"),
                        "#64748b", "#94a3b8", "#cbd5e1", "#e2e8f0",
                        "#475569", "#334155", "#1e293b", "#0f172a",
                        "#f1f5f9", "#f8fafc",
                    ][:12],
                    category_orders={"날짜": dates_sorted},
                )
                fig_kw.update_layout(
                    margin=dict(l=50, r=30, t=40, b=100),
                    xaxis_tickangle=-45,
                    height=360,
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_kw, width="stretch")
            else:
                st.info("날짜별 데이터가 조회되지 않았습니다.", icon=":material/info:")
        else:
            st.info("분석 대상 실습일지가 조회되지 않았습니다.", icon=":material/info:")


# ═══════════════════════════════════════════════════════════════════
# 메뉴 3. 학생별 포트폴리오 조회
# ═══════════════════════════════════════════════════════════════════
def _render_portfolio_review_view(students: list[dict]) -> None:
    """좌측 사이드바 「학생별 포트폴리오 조회」 본문.

    선택한 학생의 베스트 포트폴리오(역량 레이더·NCS 진도·기술 스택·베스트 실습)를
    교사 화면에 그대로 출력하고, 하단에 「지도교사 종합의견」 입력·저장 영역을 배치한다.
    """
    if not students:
        st.info("등록된 학생이 존재하지 않습니다.", icon=":material/info:")
        return

    # ─── 학생 선택 ───
    with st.container(border=True):
        options = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
        selected_uid = st.selectbox(
            "조회할 학생",
            options=list(options.keys()),
            format_func=lambda u: options[u],
            key="portfolio_review_student",
        )

    logs = list_logs(selected_uid)
    prog = seed_progress_if_missing(selected_uid, DEFAULT_NCS_PROGRESS)
    avg_prog = round(sum(prog.values()) / max(len(prog), 1), 1) if prog else 0

    # ─── 포트폴리오 헤더 + KPI ───
    with st.container(border=True):
        st.markdown(
            f"""
            <div style='padding:0.5rem 0 0.75rem 0;'>
              <p style='margin:0 0 0.2rem 0;font-size:0.75rem;color:{P["text_muted"]};
                letter-spacing:0.04em;'>NCS 국가직무능력표준 기반</p>
              <h3 style='margin:0;color:{P["primary"]};font-size:1.25rem;'>
                {options[selected_uid]} · NCS 종합 직무 포트폴리오
              </h3>
              <p style='margin:0.25rem 0 0 0;font-size:0.88rem;color:{P["text_secondary"]};'>
                용산철도고등학교 산학일체형 도제학교 · 교사 검토 화면
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        hm1, hm2, hm3 = st.columns(3)
        hm1.metric("누적 실습", f"{len(logs)}회")
        hm2.metric("평균 NCS 진도", f"{avg_prog}%")
        hm3.metric("추적 단위", f"{len(prog)}개")

    # ─── 역량 레이더 + NCS 진도 ───
    with st.container(border=True):
        st.subheader("NCS 직무 역량 종합 리포트", divider="gray")
        if logs:
            c_l, c_r = st.columns([1, 1])
            with c_l:
                axes, vals = radar_scores_from_logs(logs)
                r_vals = list(vals) + [vals[0]]
                theta_vals = list(axes) + [axes[0]]
                fig = go.Figure()
                fig.add_trace(
                    go.Scatterpolar(
                        r=r_vals,
                        theta=theta_vals,
                        fill="toself",
                        line=dict(color=P["primary"], width=2),
                        fillcolor="rgba(15, 118, 110, 0.15)",
                    )
                )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                    showlegend=False,
                    height=320,
                    margin=dict(l=40, r=40, t=20, b=20),
                    paper_bgcolor="rgba(255,255,255,0)",
                    plot_bgcolor="rgba(255,255,255,0)",
                )
                st.plotly_chart(fig, width="stretch")
            with c_r:
                st.markdown("**NCS 능력단위별 이수 현황**")
                prog_df = pd.DataFrame(
                    [
                        {"능력단위": format_ncs_unit(u), "달성률(%)": v}
                        for u, v in sorted(prog.items(), key=lambda x: -x[1])
                    ]
                )
                st.dataframe(prog_df, width="stretch", hide_index=True, height=320)
        else:
            st.info(
                "저장된 일지가 없어 역량 요약을 표시할 수 없습니다.",
                icon=":material/info:",
            )

    # ─── 일지 목록은 [요약·현황 (탭)] → 학생별 상세보기로 이동 ───
    with st.container(border=True):
        st.subheader("실습일지 목록", divider="gray")
        st.caption(
            "해당 학생이 작성한 모든 실습일지는 좌측 메뉴 [요약·현황 (탭)]에서 "
            "[학생별 상세보기]를 열고 동일한 학생을 선택하시면 날짜순으로 조회하실 수 있습니다."
        )
        if not logs:
            st.info("작성된 실습일지가 조회되지 않았습니다.", icon=":material/info:")

    # ─── AI 세특 초안 도구 ───
    with st.container(border=True):
        st.subheader("AI 세특 초안 도구", divider="gray")
        st.caption("BSR 이력으로부터 자동 생성된 세특 초안을 종합의견 작성 시 참고 자료로 활용하시기 바랍니다.")
        seuteuk_key = f"seuteuk_draft_{selected_uid}"
        if st.button(
            "초안 자동 생성",
            key=f"btn_seuteuk_draft_{selected_uid}",
            width="stretch",
            icon=":material/auto_awesome:",
        ):
            with st.spinner("AI 세특 초안을 생성하는 중입니다..."):
                draft = _make_seuteuk(selected_uid, logs)
            st.session_state[seuteuk_key] = draft
        draft_text = st.session_state.get(seuteuk_key, "")
        if draft_text:
            safe_draft = html.escape(str(draft_text)).replace("\n", "<br/>")
            st.markdown(
                f"<div class='report-card-inner' style='max-height:260px;overflow-y:auto;"
                f"font-size:0.92rem;line-height:1.65;'>{safe_draft}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("[초안 자동 생성] 버튼을 통해 AI 초안을 생성하시기 바랍니다.")

    # ─── 지도교사 종합의견 ───
    with st.container(border=True):
        st.subheader("지도교사 종합의견", divider="gray")
        st.caption(
            "본문은 학생 포트폴리오의 [지도교사 종합의견] 영역에 즉시 반영됩니다. "
            "[확정 저장]으로 저장된 의견만 학생 화면에 노출됩니다."
        )

        teacher_input_key = f"teacher_comment_input_{selected_uid}"
        loaded_marker_key = f"_teacher_loaded_for_{selected_uid}"
        if not st.session_state.get(loaded_marker_key):
            existing = get_portfolio_comment(selected_uid)
            st.session_state[teacher_input_key] = (
                (existing.get("comment_text") or "") if existing else ""
            )
            st.session_state[loaded_marker_key] = True

        st.text_area(
            "교사 코멘트",
            height=220,
            key=teacher_input_key,
            placeholder=(
                "예) S03 학생은 한 학기 동안 PLC 시퀀스 제어 및 전자회로조립 영역에서 "
                "꾸준히 BSR 구조화 일지를 작성하였으며, 특히 안전 점검(LOTO·접지) 절차를 "
                "본인의 언어로 서술한 점이 인상적이었습니다…"
            ),
        )

        existing_row = get_portfolio_comment(selected_uid)
        if existing_row:
            last_at = existing_row.get("updated_at", "")
            confirmed = int(existing_row.get("is_confirmed") or 0)
            status_label = "확정 저장됨 (학생 노출)" if confirmed else "임시 저장 상태"
            st.caption(f"최근 갱신: {last_at} · 상태: {status_label}")

        btn_a, btn_b = st.columns([1, 1])
        with btn_a:
            if st.button(
                "임시 저장",
                key=f"btn_save_draft_{selected_uid}",
                width="stretch",
                icon=":material/save:",
            ):
                body = (st.session_state.get(teacher_input_key) or "").strip()
                if not body:
                    st.warning("저장할 내용을 입력해 주십시오.", icon=":material/warning:")
                else:
                    save_portfolio_comment(selected_uid, body, "", confirmed=False)
                    st.success(
                        "임시 저장이 완료되었습니다. 학생 화면에는 아직 표시되지 않습니다.",
                        icon=":material/check_circle:",
                    )
        with btn_b:
            if st.button(
                "확정 저장 (학생 공개)",
                key=f"btn_save_final_{selected_uid}",
                width="stretch",
                type="primary",
                icon=":material/check_circle:",
            ):
                body = (st.session_state.get(teacher_input_key) or "").strip()
                if not body:
                    st.warning("저장할 내용을 입력해 주십시오.", icon=":material/warning:")
                else:
                    level, _cmt = _evaluate_seungwa_reflection(logs)
                    save_portfolio_comment(selected_uid, body, level, confirmed=True)
                    st.success(
                        "학생 포트폴리오에 지도교사 의견이 확정 반영되었습니다.",
                        icon=":material/check_circle:",
                    )


# ═══════════════════════════════════════════════════════════════════
# 메뉴: 학생별 직무 포트폴리오 (학생 화면과 동일한 HTML/PDF 출력)
# ═══════════════════════════════════════════════════════════════════
def _render_student_job_portfolio_view(students: list[dict]) -> None:
    if not students:
        st.info("등록된 학생이 존재하지 않습니다.", icon=":material/info:")
        return

    options = {s["uid"]: f"{_student_label(s['uid'])} ({s['uid']})" for s in students}
    selected_uid = st.selectbox(
        "조회할 학생",
        options=list(options.keys()),
        format_func=lambda u: options[u],
        key="job_portfolio_student",
    )

    logs = list_logs(selected_uid)
    prog = seed_progress_if_missing(selected_uid, DEFAULT_NCS_PROGRESS)

    # ── 베스트 실습 선택(UI는 교사 화면 세션 기준) ──
    with st.container(border=True):
        st.subheader("베스트 실습 선택", divider="gray")
        st.caption("체크된 항목만 포트폴리오(HTML/PDF)에 포함됩니다.")

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
                    ncs_name = _resolve_ncs_unit(row.get("ncs_unit", "") or "")
                    snippet = _bsr_preview_snippet(row.get("bsr") or "", max_len=30)
                    if ncs_name and snippet:
                        label = f"[{date_short}] {format_ncs_unit(ncs_name)} | {snippet}"
                    elif ncs_name:
                        label = f"[{date_short}] {format_ncs_unit(ncs_name)}"
                    elif snippet:
                        label = f"[{date_short}] {snippet}"
                    else:
                        label = f"[{date_short}]"
                    if st.checkbox(label, key=f"t_port_sel_{selected_uid}_{lid}"):
                        if isinstance(lid, int):
                            selected_ids.append(lid)

        selected_logs = [r for r in logs if r.get("id") in selected_ids]

    # ── HTML 생성/다운로드/미리보기 (student_view 로직 재사용) ──
    resume_html = _build_resume_page_html(selected_uid, get_student_profile(selected_uid), prog, logs)
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

    with st.container(border=True):
        st.subheader("포트폴리오 내보내기 및 미리보기", divider="gray")
        st.download_button(
            label="포트폴리오 HTML 다운로드 (브라우저에서 Ctrl+P → PDF 저장)",
            data=full_html.encode("utf-8"),
            file_name=f"{selected_uid}_portfolio.html",
            mime="text/html",
            key=f"t_portfolio_html_dl_{selected_uid}",
            type="primary",
            width="stretch",
            icon=":material/download:",
        )
        st.caption(
            "다운로드한 HTML 파일을 브라우저에서 열고 Ctrl+P 인쇄 대화상자의 [PDF로 저장]을 선택하시기 바랍니다."
        )
        st.markdown("##### 미리보기")
        st.markdown(
            f"<style>{portfolio_css}</style>{inner_html}",
            unsafe_allow_html=True,
        )

    # ─── 지도교사 종합의견 (확정 저장 시 학생 포트폴리오 HTML/PDF에도 자동 포함) ───
    with st.container(border=True):
        st.subheader("지도교사 종합의견", divider="gray")
        st.caption("확정 저장된 의견은 학생의 최종 포트폴리오(HTML/PDF)에 자동 반영됩니다.")

        teacher_input_key = f"teacher_comment_input_job_{selected_uid}"
        loaded_marker_key = f"_teacher_loaded_job_for_{selected_uid}"
        if not st.session_state.get(loaded_marker_key):
            existing = get_portfolio_comment(selected_uid)
            st.session_state[teacher_input_key] = (existing.get("comment_text") or "") if existing else ""
            st.session_state[loaded_marker_key] = True

        st.text_area(
            "교사 코멘트",
            height=220,
            key=teacher_input_key,
        )

        btn_a, btn_b = st.columns([1, 1])
        with btn_a:
            if st.button(
                "임시 저장",
                key=f"btn_save_draft_job_{selected_uid}",
                width="stretch",
                icon=":material/save:",
            ):
                body = (st.session_state.get(teacher_input_key) or "").strip()
                if not body:
                    st.warning("저장할 내용을 입력해 주십시오.", icon=":material/warning:")
                else:
                    save_portfolio_comment(selected_uid, body, "", confirmed=False)
                    st.success(
                        "임시 저장이 완료되었습니다. 학생 화면에는 아직 표시되지 않습니다.",
                        icon=":material/check_circle:",
                    )
        with btn_b:
            if st.button(
                "확정 저장 (학생 공개)",
                key=f"btn_save_final_job_{selected_uid}",
                width="stretch",
                type="primary",
                icon=":material/check_circle:",
            ):
                body = (st.session_state.get(teacher_input_key) or "").strip()
                if not body:
                    st.warning("저장할 내용을 입력해 주십시오.", icon=":material/warning:")
                else:
                    level, _cmt = _evaluate_seungwa_reflection(logs)
                    save_portfolio_comment(selected_uid, body, level, confirmed=True)
                    st.success(
                        "학생 포트폴리오에 지도교사 의견이 확정 반영되었습니다.",
                        icon=":material/check_circle:",
                    )


# ═══════════════════════════════════════════════════════════════════
# 계정 관리: 학생 ID·비밀번호 일괄 조회 + 개별 비밀번호 재설정
# ═══════════════════════════════════════════════════════════════════
def _render_account_management_view() -> None:
    with st.container(border=True):
        st.subheader("계정 관리", divider="gray")
        st.caption(
            "학생이 비밀번호를 분실하였을 때 즉시 조회 및 재설정할 수 있는 화면입니다. "
            "교내 폐쇄망 운영을 전제로 평문으로 표시되므로, 외부 모니터 및 화면 캡처 노출에 유의하시기 바랍니다."
        )

        creds = list_user_credentials()
        students_creds = sorted(
            [c for c in creds if c.get("role") == "student"],
            key=lambda c: _student_sort_key(c["uid"]),
        )
        teacher_creds = [c for c in creds if c.get("role") == "teacher"]

        # ─── 학생 계정 목록 ───
        st.markdown("##### 학생 계정 목록")
        if not students_creds:
            st.info("등록된 학생이 존재하지 않습니다.", icon=":material/info:")
        else:
            rows = [
                {
                    "번호": _student_sort_key(c["uid"]),
                    "라벨": _student_label(c["uid"]),
                    "아이디": c["uid"],
                    "현재 비밀번호": c.get("password") or c.get("pw") or "",
                }
                for c in students_creds
            ]
            df = pd.DataFrame(rows).sort_values(by="번호").reset_index(drop=True)
            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                column_config={
                    "번호": st.column_config.NumberColumn(width="small"),
                    "라벨": st.column_config.TextColumn(width="small"),
                    "아이디": st.column_config.TextColumn(width="medium"),
                    "현재 비밀번호": st.column_config.TextColumn(width="medium"),
                },
            )

        # ─── 교사 계정 (참고용) ───
        if teacher_creds:
            st.markdown("##### 교사 계정 (참고)")
            t_df = pd.DataFrame(
                [
                    {
                        "역할": "교사",
                        "아이디": c["uid"],
                        "현재 비밀번호": c.get("password") or c.get("pw") or "",
                    }
                    for c in teacher_creds
                ]
            )
            st.dataframe(t_df, width="stretch", hide_index=True)

        st.divider()

        # ─── 학생 비밀번호 재설정 ───
        st.markdown("##### 학생 비밀번호 재설정")
        st.caption(
            "학생이 비밀번호를 분실하거나 변경을 요청한 경우 사용합니다. "
            "재설정 후에는 학생이 본인 사이드바에서 새로운 비밀번호로 다시 변경할 수 있습니다."
        )
        if students_creds:
            col_pick, col_pw, col_btn = st.columns([1.2, 1.2, 1])
            with col_pick:
                target_uid = st.selectbox(
                    "대상 학생",
                    options=[c["uid"] for c in students_creds],
                    format_func=lambda u: f"{_student_label(u)} ({u})",
                    key="account_reset_target",
                )
            with col_pw:
                new_pw = st.text_input(
                    "새 비밀번호",
                    type="password",
                    key="account_reset_new_pw",
                    help="비워 두면 기본값 1234로 초기화됩니다.",
                )
            with col_btn:
                st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)
                if st.button(
                    "비밀번호 재설정",
                    width="stretch",
                    type="primary",
                    icon=":material/lock_reset:",
                ):
                    pw_to_set = (new_pw or "").strip() or "1234"
                    if len(pw_to_set) < 4:
                        st.error(
                            "비밀번호는 4자 이상이어야 합니다.",
                            icon=":material/error:",
                        )
                    elif update_password(target_uid, pw_to_set):
                        st.session_state.pop("account_reset_new_pw", None)
                        st.success(
                            f"{_student_label(target_uid)}({target_uid})의 비밀번호가 "
                            f"'{pw_to_set}'(으)로 재설정되었습니다.",
                            icon=":material/check_circle:",
                        )
                        st.rerun()
                    else:
                        st.error(
                            "비밀번호 재설정에 실패하였습니다.",
                            icon=":material/error:",
                        )


def _teacher_merged_logs_backup_csv(snapshot: dict[str, list[dict[str, Any]]]) -> bytes:
    """여러 학생 일지 스냅샷을 하나의 CSV로 합친다."""
    acc: list[dict[str, Any]] = []
    for suid, rows in snapshot.items():
        for r in rows:
            d = dict(r)
            d["student_uid"] = suid
            acc.append(d)
    if not acc:
        return logs_to_csv_bytes([], owner_uid="")
    return pd.DataFrame(acc).to_csv(index=False).encode("utf-8-sig")


@st.dialog("실습 일지 삭제 확인 (교사)")
def _dlg_teacher_delete_one_log(student_uid: str, row: dict[str, Any]) -> None:
    lid = row.get("id")
    st.markdown(f"**삭제하시겠습니까?**  \n학생 **{student_uid}** · 일지 **#{lid}** · **{row.get('date', '—')}**")
    st.caption("복구할 수 없습니다. 백업 CSV를 받은 뒤 진행하세요.")
    st.download_button(
        "삭제 대상 일지 백업 (CSV)",
        data=logs_to_csv_bytes([row], owner_uid=student_uid),
        file_name=f"backup_log_{student_uid}_{lid}.csv",
        mime="text/csv",
        key=f"tch_dlg_dl_one_{student_uid}_{lid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"tch_dlg_can_one_{student_uid}_{lid}", width="stretch"):
            st.session_state.pop("_tch_dlg_journal_one", None)
            st.rerun()
    with c2:
        if st.button("예, 삭제합니다", type="primary", key=f"tch_dlg_go_one_{student_uid}_{lid}", width="stretch"):
            delete_log(student_uid, int(lid))
            st.session_state.pop("_tch_dlg_journal_one", None)
            st.rerun()


@st.dialog("학생 일지 전체 삭제 확인 (교사)")
def _dlg_teacher_clear_student_logs(student_uid: str, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    st.markdown(f"**정말 이 학생의 일지를 모두 삭제하시겠습니까?**  \n대상: **{student_uid}** · **{n}건**")
    st.caption("복구할 수 없습니다. NCS 진행률도 초기화됩니다. 백업 CSV를 저장하세요.")
    st.download_button(
        "해당 학생 일지 전체 백업 (CSV)",
        data=logs_to_csv_bytes(rows, owner_uid=student_uid),
        file_name=f"backup_all_logs_{student_uid}.csv",
        mime="text/csv",
        key=f"tch_dlg_dl_clr_stu_{student_uid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"tch_dlg_can_clr_stu_{student_uid}", width="stretch"):
            st.session_state.pop("_tch_dlg_clear_one", None)
            st.rerun()
    with c2:
        if st.button("예, 모두 삭제합니다", type="primary", key=f"tch_dlg_go_clr_stu_{student_uid}", width="stretch"):
            clear_logs(student_uid)
            st.session_state.pop("_tch_dlg_clear_one", None)
            st.rerun()


@st.dialog("전체 학생 일지 삭제 확인 (교사)")
def _dlg_teacher_clear_all_logs(snapshot: dict[str, list[dict[str, Any]]]) -> None:
    total = sum(len(v) for v in snapshot.values())
    st.markdown(f"**정말 전체 학생의 일지를 삭제하시겠습니까?**  \n총 **{total}건** (학생 수 {len(snapshot)}명)")
    st.caption("복구할 수 없습니다. 각 학생의 NCS 진행률도 초기화됩니다.")
    st.download_button(
        "전체 학생 일지 백업 (CSV)",
        data=_teacher_merged_logs_backup_csv(snapshot),
        file_name="backup_all_students_logs.csv",
        mime="text/csv",
        key="tch_dlg_dl_clr_all",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key="tch_dlg_can_clr_all", width="stretch"):
            st.session_state.pop("_tch_dlg_clear_all", None)
            st.rerun()
    with c2:
        if st.button("예, 전체 삭제합니다", type="primary", key="tch_dlg_go_clr_all", width="stretch"):
            for suid in snapshot:
                clear_logs(suid)
            st.session_state.pop("_tch_dlg_clear_all", None)
            st.rerun()


@st.dialog("학생 이력서 삭제 확인 (교사)")
def _dlg_teacher_clear_profile(student_uid: str) -> None:
    prof = get_student_profile(student_uid)
    st.markdown(f"**이 학생의 저장된 이력서·표지 정보를 모두 삭제하시겠습니까?**  \n대상: **{student_uid}**")
    st.caption("실습 일지는 삭제되지 않습니다. 복구할 수 없습니다.")
    st.download_button(
        "현재 이력서 백업 (JSON)",
        data=profile_to_json_bytes(prof),
        file_name=f"backup_profile_{student_uid}.json",
        mime="application/json",
        key=f"tch_dlg_dl_prof_{student_uid}",
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("취소", key=f"tch_dlg_can_prof_{student_uid}", width="stretch"):
            st.session_state.pop("_tch_dlg_profile", None)
            st.rerun()
    with c2:
        if st.button("예, 이력서를 삭제합니다", type="primary", key=f"tch_dlg_go_prof_{student_uid}", width="stretch"):
            clear_student_profile(student_uid)
            st.session_state.pop("_tch_dlg_profile", None)
            st.rerun()


def _run_teacher_delete_dialogs() -> None:
    p0 = st.session_state.get("_tch_dlg_journal_one")
    if isinstance(p0, dict) and p0.get("sel_uid") and p0.get("row"):
        _dlg_teacher_delete_one_log(str(p0["sel_uid"]), p0["row"])
    p1 = st.session_state.get("_tch_dlg_clear_one")
    if isinstance(p1, dict) and p1.get("uid") and isinstance(p1.get("rows"), list):
        _dlg_teacher_clear_student_logs(str(p1["uid"]), p1["rows"])
    p2 = st.session_state.get("_tch_dlg_clear_all")
    if isinstance(p2, dict):
        _dlg_teacher_clear_all_logs(p2)
    p3 = st.session_state.get("_tch_dlg_profile")
    if isinstance(p3, str) and p3:
        _dlg_teacher_clear_profile(p3)


# ═══════════════════════════════════════════════════════════════════
# 진입점: 좌측 사이드바 라디오 + 우측 메인 분할 레이아웃
# ═══════════════════════════════════════════════════════════════════
def show_teacher() -> None:
    students = sorted(
        [u for u in list_users() if u["uid"] != TEACHER_UID],
        key=lambda u: _student_sort_key(u["uid"]),
    )
    overview = _collect_class_overview(students)

    NAV_OPTIONS = [
        "요약·현황 (탭)",
        "실습 일지 정밀 점검",
        "학생별 포트폴리오 조회",
        "학생별 직무 포트폴리오",
        "계정 관리",
    ]

    # ─── 좌측 사이드바 ───
    with st.sidebar:
        st.markdown(
            f"""
<div style="padding:0.35rem 0 0.1rem 0;">
  <div style="font-size:0.72rem;color:{P['text_secondary']};letter-spacing:0.08em;
    text-transform:uppercase;font-weight:600;">Teacher Console</div>
  <div style="font-size:1.15rem;font-weight:700;color:{P['text']};
    margin-top:0.15rem;line-height:1.25;">통합 관리 시스템</div>
  <div style="font-size:0.82rem;color:{P['text_secondary']};margin-top:0.1rem;">
    학급 {STUDENT_COUNT}명 · 도제생 진도·성찰 모니터</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            f"<div style='font-size:0.72rem;color:{P['text_secondary']};font-weight:700;"
            "letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.35rem;'>Menu</div>",
            unsafe_allow_html=True,
        )
        nav = st.radio(
            "메뉴",
            options=NAV_OPTIONS,
            key="teacher_nav",
            label_visibility="collapsed",
        )

        # ─── 비밀번호 변경 (사이드바 최하단) ───
        st.divider()
        render_password_change_expander(TEACHER_UID, key_prefix="teacher")

    # ─── 메인 헤더 ───
    st.markdown(
        f"<div style='display:flex;align-items:baseline;gap:0.6rem;"
        f"margin:0 0 0.6rem 0;'>"
        f"<h2 style='margin:0;color:{P['text']};font-weight:800;'>{nav}</h2>"
        f"<span style='color:{P['text_secondary']};font-size:0.88rem;'>"
        f"· 통합 관리 시스템</span></div>",
        unsafe_allow_html=True,
    )

    # ─── 메인 본문 라우팅 ───
    if nav == NAV_OPTIONS[0]:
        tab1, tab2, tab3 = st.tabs(
            [
                ":material/dashboard: 종합 현황",
                ":material/person_search: 학생별 상세보기",
                ":material/database: 데이터 행정",
            ]
        )
        with tab1:
            _render_tab_overview(students, overview)
        with tab2:
            _render_tab_student_journals(students)
        with tab3:
            _render_tab_data_administration(students)
    elif nav == NAV_OPTIONS[1]:
        _render_dashboard_deep_analytics(students, overview)
        _render_log_inspection_view(students, overview)
    elif nav == NAV_OPTIONS[2]:
        _render_portfolio_review_view(students)
    elif nav == NAV_OPTIONS[3]:
        _render_student_job_portfolio_view(students)
    else:
        _render_account_management_view()

    _run_teacher_delete_dialogs()

