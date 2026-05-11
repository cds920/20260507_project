"""
NCS 포트폴리오 - UI 스타일
밝은 화이트 & 틸(Teal) / 민트 톤의 미니멀 플랫 UI
"""

import streamlit as st

# ═══════════════════════════════════════════════════════════════════
# Color Palette — 화이트 베이스 + 틸/민트 (Flat·산뜻한 톤)
# ═══════════════════════════════════════════════════════════════════
P = {
    "primary": "#0f766e",       # Teal-700 — 제목·강조
    "primary_hover": "#115e59",
    "primary_muted": "rgba(15, 118, 110, 0.07)",
    "accent": "#14b8a6",        # Teal-500 — 링크·포인트
    "accent_soft": "#5eead4",   # Teal-300 민트
    "accent_glow": "rgba(20, 184, 166, 0.14)",
    "bg": "#ffffff",
    "bg_deep": "#f8fafc",       # 아주 밝은 쿨그레이
    "bg_elevated": "#ffffff",
    "text": "#1e293b",          # Slate-800
    "text_secondary": "#64748b",
    "text_muted": "#94a3b8",
    "border": "#e2e8f0",
    "border_light": "#e2e8f0",
    "success": "#0d9488",
    "shadow_sm": "0 1px 2px rgba(0, 0, 0, 0.04)",
    "shadow": "0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03)",
    "shadow_lg": "0 10px 15px -3px rgba(0, 0, 0, 0.06), 0 4px 6px -2px rgba(0, 0, 0, 0.04)",
    "card_hover": "0 6px 10px -2px rgba(15, 118, 110, 0.08), 0 2px 4px -1px rgba(0, 0, 0, 0.04)",
}

# Spacing Scale (px) - 타이트한 밀도 (불필요한 빈 박스/여백 최소화)
S = {
    "xs": "4px",
    "sm": "8px",
    "md": "14px",
    "lg": "20px",
    "xl": "28px",
    "2xl": "44px",
}


def render_app_footer() -> None:
    """모든 화면 하단 전문 워터마크 (회색 톤)."""
    st.markdown(
        """
        <div class="app-footer-watermark">용산철도고 NCS 직무 포트폴리오 시스템</div>
        """,
        unsafe_allow_html=True,
    )


def render_password_change_expander(uid: str, *, key_prefix: str) -> None:
    """학생/교사 사이드바 최하단의 [비밀번호 변경] 공통 위젯.

    - 자기 자신(현재 로그인 UID)의 비밀번호만 변경할 수 있다.
    - 신규/확인 비밀번호가 일치하고 4자 이상일 때 저장.
    - 비밀번호 평문은 화면에 표시하지 않는다.
    """
    # 지연 import — db 모듈이 ui_style을 다시 import하지 않도록
    from db import update_password

    with st.expander("비밀번호 변경", expanded=False):
        st.caption(
            "현재 로그인 계정의 비밀번호를 변경합니다. "
            "비밀번호는 4자 이상이며, 외부에 노출되지 않도록 주의해 주세요."
        )
        new_pw = st.text_input(
            "새 비밀번호",
            type="password",
            key=f"{key_prefix}_new_pw",
        )
        confirm_pw = st.text_input(
            "새 비밀번호 확인",
            type="password",
            key=f"{key_prefix}_confirm_pw",
        )
        if st.button("비밀번호 저장", key=f"{key_prefix}_save_pw", width="stretch"):
            new_pw_s = (new_pw or "").strip()
            confirm_s = (confirm_pw or "").strip()
            if not new_pw_s:
                st.error("새 비밀번호를 입력해 주세요.")
            elif len(new_pw_s) < 4:
                st.error("비밀번호는 4자 이상이어야 합니다.")
            elif new_pw_s != confirm_s:
                st.error("두 비밀번호가 서로 일치하지 않습니다.")
            elif update_password(uid, new_pw_s):
                # 입력값 흔적 제거
                st.session_state.pop(f"{key_prefix}_new_pw", None)
                st.session_state.pop(f"{key_prefix}_confirm_pw", None)
                st.success("비밀번호가 변경되었습니다. 다음 로그인부터 새 비밀번호를 사용해 주세요.")
            else:
                st.error("비밀번호 저장에 실패했습니다. 관리자에게 문의해 주세요.")


def _disable_browser_translate() -> None:
    """브라우저 자동 번역 차단.

    Chrome/Edge가 한국어 페이지를 자동 번역하면 React가 관리하는 DOM 노드를
    임의로 교체해서 `Failed to execute 'removeChild' on 'Node'` 류의 에러가
    Streamlit 화면 상단에 빨갛게 뜬다. 백엔드 동작에는 지장 없지만 시각적으로
    거슬리므로 부모 문서(top-level)에 `translate="no"`와 `<meta name="google"
    content="notranslate">`를 강제로 주입한다.

    streamlit.components.v1.html 은 iframe 안에서 실행되지만, 동일 오리진이라
    `window.parent.document`로 부모 문서에 접근할 수 있다. height=0 으로 화면에는
    안 보이게 한다.
    """
    try:
        import streamlit.components.v1 as components

        components.html(
            """
            <script>
            (function() {
                try {
                    var doc = (window.parent && window.parent.document) || document;
                    doc.documentElement.setAttribute('translate', 'no');
                    if (doc.documentElement.classList) {
                        doc.documentElement.classList.add('notranslate');
                    }
                    if (doc.head && !doc.head.querySelector('meta[name="google"][content="notranslate"]')) {
                        var m = doc.createElement('meta');
                        m.name = 'google';
                        m.content = 'notranslate';
                        doc.head.appendChild(m);
                    }
                } catch (e) {}
            })();
            </script>
            """,
            height=0,
        )
    except Exception:
        # 컴포넌트 주입이 실패해도 메인 UI 동작에는 영향 없음.
        pass


def apply_advanced_ui() -> None:
    # 자동 번역으로 인한 React DOM 충돌 방지 (가장 먼저 주입)
    _disable_browser_translate()

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=DM+Sans:ital,opsz,wght@0,9..40,500;0,9..40,600;0,9..40,700&display=swap');

        .app-footer-watermark {{
            text-align: center;
            font-size: 0.8rem;
            color: {P["text_muted"]};
            padding: {S["xl"]} 0 {S["md"]};
            margin-top: {S["2xl"]};
            border-top: 1px solid {P["border_light"]};
            letter-spacing: 0.02em;
        }}

        /* ─── Base — 솔리드·밝은 배경 (무거운 방사형 그라데이션 없음) ─── */
        .stApp {{
            background: linear-gradient(180deg, {P["bg"]} 0%, {P["bg_deep"]} 100%);
            color: {P["text"]};
            font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            -webkit-font-smoothing: antialiased;
        }}
        [data-testid="stMarkdown"] p {{
            line-height: 1.65;
        }}
        /* IMPORTANT:
           Streamlit 기본 UI(헤더/사이드바/메뉴)는 숨기지 않는다.
           (header, #MainMenu, 사이드바 토글 등은 Streamlit이 관리) */

        /* ─── Layout & Spacing (상단 UI가 잘리지 않도록 2.5rem의 숨통 확보) ─── */
        div.block-container {{
            padding: 2.5rem 1rem 1rem 1rem !important;
            max-width: 1180px;
        }}

        /* ─── Login (로그인 전용 셸) ─── */
        .login-page-outer {{
            max-width: 480px;
            margin: 0 auto 1.5rem;
            padding: {S["md"]} {S["sm"]};
        }}
        .login-page-card {{
            background: #ffffff;
            border: 1px solid {P["border_light"]};
            border-radius: 16px;
            box-shadow: {P["shadow_sm"]};
            padding: {S["xl"]} {S["xl"]} {S["lg"]};
        }}
        /* Streamlit 컬럼이 카드 안에서 자연스럽게 보이도록 */
        .login-page-card [data-testid="column"] {{
            min-width: 0;
        }}
        h1.login-title {{
            font-family: 'Noto Sans KR', sans-serif !important;
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            color: {P["primary"]} !important;
            margin: 0 0 0.5rem !important;
            text-align: center;
            line-height: 1.3 !important;
        }}
        p.login-subtitle {{
            font-family: 'Noto Sans KR', sans-serif !important;
            font-size: 0.95rem !important;
            color: {P["text_secondary"]} !important;
            letter-spacing: -0.01em;
            margin: 0 auto !important;
            max-width: 28em;
            text-align: center;
            line-height: 1.55 !important;
        }}

        /* ─── Sidebar ─── */
        section[data-testid="stSidebar"],
        section.stSidebar {{
            background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
        }}
        section[data-testid="stSidebar"] > div,
        section.stSidebar > div {{
            background: transparent !important;
            border-right: 1px solid #e2e8f0;
            box-shadow: {P["shadow_sm"]};
            padding: 2.5rem 1rem 1rem 1rem !important;
        }}
        section[data-testid="stSidebar"] h3,
        section.stSidebar h3 {{
            font-size: 1.02rem !important;
            letter-spacing: -0.02em;
            padding-bottom: 0.35rem;
            border-bottom: 1px solid {P["border_light"]};
            margin-bottom: 0.65rem !important;
        }}
        section[data-testid="stSidebar"] .stMarkdown,
        section.stSidebar .stMarkdown {{
            margin-bottom: {S["sm"]};
        }}
        section[data-testid="stSidebar"] [data-testid="stMetric"],
        section.stSidebar [data-testid="stMetric"] {{
            padding: 0.65rem 0.85rem !important;
        }}
        /* ─── Sidebar 세로형 네비 메뉴 (radio → 버튼 스타일) ─── */
        section[data-testid="stSidebar"] div[role="radiogroup"],
        section.stSidebar div[role="radiogroup"] {{
            flex-direction: column !important;
            gap: 0.3rem !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label,
        section.stSidebar div[role="radiogroup"] > label {{
            display: flex !important;
            align-items: center;
            width: 100%;
            padding: 0.55rem 0.8rem !important;
            margin: 0 !important;
            border-radius: 10px;
            border: 1px solid transparent;
            background: transparent;
            cursor: pointer;
            transition: background 0.15s ease, border-color 0.15s ease;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover,
        section.stSidebar div[role="radiogroup"] > label:hover {{
            background: #f1f5f9;
            border-color: {P["border_light"]};
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
        section.stSidebar div[role="radiogroup"] > label:has(input:checked) {{
            background: #eff6ff;
            border-color: #bfdbfe;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child,
        section.stSidebar div[role="radiogroup"] > label > div:first-child {{
            display: none !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label p,
        section.stSidebar div[role="radiogroup"] > label p {{
            font-size: 0.92rem !important;
            font-weight: 500 !important;
            color: #334155 !important;
            margin: 0 !important;
        }}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p,
        section.stSidebar div[role="radiogroup"] > label:has(input:checked) p {{
            color: {P["primary"]} !important;
            font-weight: 700 !important;
        }}

        /* ─── Cards — 또렷한 경계·옅은 섀도우 (배경과 확실히 구분) ─── */
        .report-card {{
            background: #ffffff;
            padding: 1.1rem 1.25rem;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            margin-bottom: {S["md"]};
        }}
        .report-card-tab {{
            margin-bottom: {S["md"]};
            padding: 1.1rem 1.25rem;
            transition: box-shadow 0.2s ease, border-color 0.2s ease;
            border-radius: 14px;
            border: 1px solid #e2e8f0;
            background: {P["bg_elevated"]};
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        }}
        /* 카드 내부 Streamlit 기본 마진 축소 — 빈 공간 최소화 */
        .report-card > div:first-child,
        .report-card-tab > div:first-child {{
            margin-top: 0;
        }}
        .report-card [data-testid="stMarkdown"]:empty,
        .report-card-tab [data-testid="stMarkdown"]:empty {{
            display: none;
        }}

        /* ─── Tab1: 입력(Input) vs AI 피드백(Output) 구역 ─── */
        .tab1-main-heading {{
            color: {P["primary"]};
            font-size: 1.22rem;
            font-weight: 700;
            margin: 0 0 0.65rem 0;
            letter-spacing: -0.02em;
        }}
        .zone-label {{
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            margin: 0 0 0.45rem 0;
        }}
        .zone-label--input {{
            color: {P["primary"]};
            padding: 0.4rem 0 0.15rem 0;
        }}
        .zone-label--feedback {{
            color: {P["primary"]};
            border-left: 4px solid {P["accent"]};
            padding: 0.35rem 0 0.15rem 0.65rem;
            margin-bottom: 0.5rem;
        }}
        .input-zone {{
            background: #ffffff;
            box-shadow: {P["shadow_sm"]};
            border: 1px solid #e2e8f0;
            border-radius: 14px;
            padding: {S["md"]} {S["lg"]};
            margin-bottom: {S["sm"]};
        }}
        .input-zone--wide {{
            margin-top: {S["xs"]};
        }}
        .ai-feedback-zone {{
            background: linear-gradient(180deg, #ffffff 0%, rgba(204, 251, 241, 0.35) 100%);
            border-left: 4px solid {P["accent"]};
            border-radius: 14px;
            padding: {S["md"]} {S["lg"]};
            margin-bottom: {S["sm"]};
            box-shadow: {P["shadow_sm"]};
        }}
        .ai-feedback-panel {{
            background: rgba(255, 255, 255, 0.78);
            border-radius: 10px;
            padding: {S["sm"]} {S["md"]};
            margin-bottom: {S["sm"]};
            border: 1px solid #e2e8f0;
        }}
        .ai-feedback-panel:last-child {{
            margin-bottom: 0;
        }}
        .ai-feedback-panel__title {{
            font-size: 1rem;
            font-weight: 700;
            color: {P["primary"]};
            margin: 0 0 0.65rem 0;
        }}
        .ai-feedback-panel__label {{
            font-size: 0.75rem;
            font-weight: 700;
            color: {P["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 0.65rem 0 0.3rem 0;
        }}
        .ai-feedback-panel__label:first-of-type {{
            margin-top: 0;
        }}
        .ai-feedback-panel__body {{
            font-size: 0.95rem;
            line-height: 1.65;
            color: {P["text"]};
            margin: 0;
        }}
        .ai-feedback-panel__ncs {{
            font-size: 1rem;
            font-weight: 600;
            color: {P["primary"]};
            margin: 0;
        }}
        .ai-feedback-panel__safety {{
            margin: 0;
        }}
        .ai-feedback-panel__score {{
            color: #059669;
            font-weight: 700;
            font-size: 1.05rem;
        }}
        .ai-feedback-panel__hint {{
            font-size: 0.88rem;
            color: {P["text_secondary"]};
            line-height: 1.55;
            margin: 0.35rem 0 0 0;
        }}
        .ai-feedback-panel--merged {{
            padding: 1.1rem 1.25rem;
        }}
        .ai-feedback-panel__merged-block {{
            margin: 0.35rem 0 0.5rem 0;
        }}
        .ai-feedback-panel__merged-k {{
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            color: {P["text_secondary"]};
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin: 0 0 0.35rem 0;
        }}
        .ai-feedback-panel__rule {{
            height: 1px;
            background: linear-gradient(90deg, {P["border"]}, transparent);
            margin: 0.65rem 0;
            border: none;
        }}
        .ai-feedback-fallback {{
            border-left: 3px solid {P["accent"]};
            padding: 0.75rem 1rem;
            background: rgba(248, 250, 252, 0.95);
            border-radius: 8px;
            font-size: 0.92rem;
            line-height: 1.65;
            margin-top: 0.5rem;
        }}
        .report-card-tab:hover {{
            box-shadow: {P["card_hover"]};
            border-color: rgba(20, 184, 166, 0.22);
        }}
        .report-card-tab + .report-card-tab {{
            margin-top: 0;
        }}
        div[data-baseweb="tab-panel"] .report-card-tab {{
            margin-bottom: {S["md"]};
        }}
        .report-card-inner {{
            padding: 1.25rem 1.35rem;
            font-size: 0.95rem;
            line-height: 1.7;
            background: rgba(248, 250, 252, 0.6);
            border-radius: 6px;
            margin-top: {S["sm"]};
        }}
        /* 섹션 구분용 (박스 없이 여백만) */
        .section-spacer {{
            margin-bottom: {S["lg"]};
        }}

        /* ─── Typography (가독성·학술 톤: 과장 없는 크기·행간) ─── */
        h1, h2, h3, h4 {{
            color: {P["primary"]};
            font-weight: 600;
            letter-spacing: -0.02em;
            line-height: 1.38;
        }}
        h1 {{ font-size: 1.65rem; margin-bottom: 1rem; margin-top: 0.25rem; }}
        h2 {{ font-size: 1.35rem; margin-bottom: 0.65rem; }}
        h3 {{ font-size: 1.12rem; margin-bottom: 0.5rem; }}
        h4 {{ font-size: 1rem; margin-bottom: 0.4rem; }}
        /* Streamlit 기본 헤딩 (st.title / st.header / st.subheader) */
        [data-testid="stHeader"] h1, div[data-testid="stDecoration"] + div h1 {{
            font-size: 1.55rem !important;
            font-weight: 600 !important;
            color: {P["primary"]} !important;
            letter-spacing: -0.02em !important;
        }}
        .main h1 {{ font-size: 1.55rem !important; }}
        .main h2 {{ font-size: 1.28rem !important; }}
        .main h3 {{ font-size: 1.08rem !important; }}
        .stMarkdown, .stText, .stCaption, .stTextInput label, .stTextArea label {{
            color: {P["text"]} !important;
            line-height: 1.65 !important;
        }}
        p, .stMarkdown p {{ color: {P["text"]}; line-height: 1.72; font-size: 1rem; }}
        .main a {{ color: {P["accent"]} !important; text-decoration: none; font-weight: 500; }}
        .main a:hover {{ text-decoration: underline; color: #0d9488 !important; }}

        /* ─── Buttons — 기본은 깔끔한 Ghost 버튼 / Primary는 Teal 그라데이션 ─── */
        .stButton > button {{
            border-radius: 10px;
            padding: 0.45rem 1rem;
            font-weight: 600;
            font-size: 0.9rem;
            letter-spacing: 0.01em;
            transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease, border-color 0.15s ease;
            /* 기본(Secondary) = Ghost 버튼: 흰색 배경 + 옅은 회색 테두리 */
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: {P["text"]} !important;
            box-shadow: none !important;
        }}
        .stButton > button:hover:not(:disabled) {{
            border-color: #cbd5e1 !important;
            background: #f8fafc !important;
            color: {P["primary"]} !important;
        }}
        .stButton > button:active {{
            transform: scale(0.98);
        }}
        .stButton > button:disabled {{
            opacity: 0.55;
            cursor: not-allowed;
        }}
        .stButton > button[kind="secondary"] {{
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: {P["text"]} !important;
        }}
        .stButton > button[kind="secondary"]:hover:not(:disabled) {{
            border-color: #cbd5e1 !important;
            background: #f8fafc !important;
            color: {P["primary"]} !important;
        }}
        .stButton > button[kind="primary"] {{
            background: linear-gradient(135deg, {P["primary"]} 0%, {P["accent"]} 100%) !important;
            border: 1px solid transparent !important;
            color: #ffffff !important;
            box-shadow: 0 2px 6px rgba(15, 118, 110, 0.18) !important;
        }}
        .stButton > button[kind="primary"]:hover:not(:disabled) {{
            background: linear-gradient(135deg, {P["primary_hover"]} 0%, #0d9488 100%) !important;
            box-shadow: 0 4px 10px rgba(15, 118, 110, 0.22) !important;
            color: #ffffff !important;
        }}
        /* Download 버튼도 Ghost 스타일 통일 */
        .stDownloadButton > button {{
            border-radius: 10px;
            padding: 0.45rem 1rem;
            font-weight: 600;
            font-size: 0.9rem;
            background: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            color: {P["text"]} !important;
        }}
        .stDownloadButton > button:hover:not(:disabled) {{
            border-color: #cbd5e1 !important;
            background: #f8fafc !important;
            color: {P["primary"]} !important;
        }}

        /* ─── Inputs ─── */
        .stTextInput input, .stTextArea textarea {{
            border-radius: 12px;
            border: 1px solid {P["border"]};
            padding: 0.65rem 1rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .stTextInput input:focus, .stTextArea textarea:focus {{
            border-color: {P["accent"]};
            box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.18);
            outline: none;
        }}

        /* ─── Select / Multiselect / File uploader ─── */
        [data-baseweb="select"] > div {{
            border-radius: 12px !important;
            border-color: {P["border"]} !important;
        }}
        [data-testid="stFileUploader"] section {{
            border-radius: 16px !important;
            border: 2px dashed rgba(15, 118, 110, 0.2) !important;
            background: {P["primary_muted"]} !important;
            padding: 1rem !important;
            transition: border-color 0.2s ease, background 0.2s ease;
        }}
        [data-testid="stFileUploader"]:hover section {{
            border-color: rgba(20, 184, 166, 0.45) !important;
            background: {P["accent_glow"]} !important;
        }}
        [data-testid="stAudioInput"] {{
            border-radius: 12px;
        }}

        /* ─── Checkbox / Radio ─── */
        .stCheckbox label, .stRadio label {{
            font-weight: 500;
        }}

        /* ─── NCS Tag ─── */
        .ncs-tag {{
            background: linear-gradient(135deg, {P["primary"]} 0%, #2dd4bf 100%);
            color: white;
            padding: {S["xs"]} {S["md"]};
            border-radius: 9999px;
            font-size: 0.8125rem;
            font-weight: 600;
        }}

        /* ─── Progress Bar ─── */
        .stProgress > div > div > div > div {{
            background: linear-gradient(90deg, {P["primary"]} 0%, {P["accent"]} 100%);
        }}

        /* ─── Glossary Box ─── */
        .glossary-box {{
            background: linear-gradient(90deg, rgba(20, 184, 166, 0.08) 0%, transparent 100%);
            border-left: 4px solid {P["accent"]};
            padding: {S["md"]} {S["lg"]};
            margin: {S["md"]} 0;
            border-radius: 0 8px 8px 0;
            color: {P["text"]};
        }}

        /* ─── 역량 레이더·AI 정합성 경고 (시각적 정돈) ─── */
        div[data-testid="stAlert"] {{
            border-radius: 10px;
            border: 1px solid {P["border_light"]};
        }}
        .radar-chart-caption {{
            font-size: 0.8rem;
            color: {P["text_secondary"]};
            margin-top: 0.35rem;
        }}

        /* ─── 실습 맞춤형 AI 코칭 (역질문 강조·설명은 help/캡션으로 축소) ─── */
        .ai-coaching-panel {{
            background: #ffffff;
            border: 1px solid {P["border_light"]};
            border-radius: 12px;
            padding: {S["md"]} {S["lg"]};
            margin: {S["sm"]} 0 {S["md"]} 0;
            box-shadow: {P["shadow_sm"]};
        }}
        .ai-coaching-q-title {{
            font-size: 1.2rem;
            font-weight: 800;
            color: {P["primary"]};
            margin: 0.35rem 0 0.55rem 0;
            letter-spacing: -0.03em;
        }}
        ol.ai-coaching-qlist {{
            margin: 0.25rem 0 0.85rem 1.15rem;
            padding: 0;
            line-height: 1.95;
            color: {P["text"]};
            font-size: 1.12rem;
            font-weight: 600;
        }}
        ol.ai-coaching-qlist li {{
            margin-bottom: 0.65rem;
        }}
        .ai-coaching-reflection-label {{
            margin: 0.85rem 0 0.3rem 0;
            font-size: 0.8rem;
            color: {P["text_secondary"]};
            letter-spacing: 0.02em;
        }}
        .ai-coaching-reflection-box {{
            border-left: 3px solid {P["primary"]};
            background: #f8fafc;
            padding: 0.85rem 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.94rem;
            line-height: 1.72;
            color: {P["text"]};
        }}

        /* ─── 메타인지 역질문 (따뜻한 교육 코치 톤) ─── */
        .meta-cognition-coach {{
            background: linear-gradient(145deg, rgba(255, 251, 235, 0.95) 0%, rgba(254, 243, 199, 0.35) 100%);
            border: 1px solid rgba(251, 191, 36, 0.35);
            border-radius: 16px;
            padding: 1.15rem 1.35rem 1.25rem 1.35rem;
            margin: 0.65rem 0 1rem 0;
            box-shadow: 0 2px 12px rgba(245, 158, 11, 0.08);
        }}
        .meta-cognition-title {{
            font-size: 1.14rem;
            font-weight: 700;
            color: #92400e;
            margin: 0 0 0.75rem 0;
            letter-spacing: -0.02em;
        }}
        ol.meta-cognition-qlist {{
            margin: 0;
            padding-left: 1.35rem;
            line-height: 1.85;
        }}
        ol.meta-cognition-qlist li.meta-cognition-qitem {{
            font-size: 1.07rem;
            font-weight: 600;
            color: #334155;
            margin-bottom: 0.6rem;
        }}
        p.reflection-example-heading {{
            font-size: 0.88rem;
            font-weight: 600;
            color: {P["text_secondary"]};
            margin: 1rem 0 0.45rem 0;
        }}
        div.reflection-example-box {{
            border-left: 3px solid rgba(20, 184, 166, 0.55);
            background: rgba(240, 253, 250, 0.75);
            padding: 0.9rem 1.05rem;
            border-radius: 0 12px 12px 0;
            font-size: 0.95rem;
            line-height: 1.75;
            color: #334155;
        }}

        /* ─── BSR 프로젝트 보고서 뷰 (render_bsr_highlighted 출력) ─── */
        .bsr-report {{
            font-family: 'Noto Sans KR', -apple-system, 'Segoe UI', sans-serif;
            color: {P["text"]};
        }}
        .bsr-report section:first-child h4 {{
            margin-top: 0.1rem !important;
        }}
        .bsr-report h4 {{
            /* 인라인 스타일이 우선이지만, 앱 내 기본 h4 마진을 차단 */
            margin-top: 1.05rem !important;
            margin-bottom: 0.45rem !important;
        }}
        .bsr-report ul {{
            margin: 0;
            padding-left: 1.1rem;
        }}
        .bsr-report li {{
            margin: 0.15rem 0;
        }}

        /* ─── 포트폴리오 표지 — 기술 스택(Tech Stack) 배지 ─── */
        .portfolio-tech-stack {{
            margin: 0.35rem 0 0.5rem 0;
        }}
        .portfolio-tech-stack h3 {{
            font-size: 1.08rem;
            color: {P["primary"]};
            margin: 0 0 0.65rem 0;
            border-bottom: 2px solid {P["primary"]};
            padding-bottom: 0.25rem;
        }}
        .portfolio-tech-stack__hint {{
            font-size: 0.82rem;
            color: {P["text_secondary"]};
            margin: 0 0 0.65rem 0;
        }}
        .portfolio-tech-stack__grid {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem 0.45rem;
            margin: 0.2rem 0 0.4rem 0;
        }}
        .portfolio-tech-badge {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.3rem 0.7rem;
            background: linear-gradient(135deg, rgba(15,118,110,0.08) 0%, rgba(20,184,166,0.08) 100%);
            border: 1px solid rgba(15,118,110,0.22);
            color: {P["primary"]};
            border-radius: 999px;
            font-size: 0.85rem;
            font-weight: 600;
            letter-spacing: 0.01em;
            line-height: 1.3;
        }}
        .portfolio-tech-badge::before {{
            content: "";
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: {P["accent"]};
        }}
        .portfolio-tech-empty {{
            color: {P["text_muted"]};
            font-style: italic;
            font-size: 0.9rem;
            padding: 0.5rem 0;
        }}

        /* ─── BSR 성찰 카드 (원문 vs AI 다듬기) ─── */
        .bsr-reflection-card {{
            border: 1px solid {P["border"]};
            border-radius: 16px;
            padding: 1.2rem 1.35rem 1.35rem 1.35rem;
            margin: 0.5rem 0 1rem 0;
            background: #ffffff;
            box-shadow: {P["shadow_sm"]};
        }}
        .bsr-reflection-h4 {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {P["primary"]};
            margin: 0.35rem 0 0.65rem 0;
            letter-spacing: -0.02em;
        }}
        .bsr-pair-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.85rem 1rem;
            align-items: stretch;
        }}
        @media (max-width: 720px) {{
            .bsr-pair-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        .bsr-col {{
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            min-height: 2.5rem;
        }}
        .bsr-col--original {{
            background: rgba(248, 250, 252, 0.85);
            border: 1px solid rgba(226, 232, 240, 0.9);
        }}
        .bsr-col--refined {{
            background: rgba(240, 253, 250, 0.45);
            border: 1px solid rgba(153, 246, 228, 0.55);
        }}
        .bsr-col-label {{
            display: block;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {P["text_secondary"]};
            margin-bottom: 0.45rem;
        }}
        .bsr-col-body {{
            font-size: 0.92rem;
            line-height: 1.65;
            color: #334155;
        }}
        .bsr-col-body--empty {{
            min-height: 3rem;
            display: flex;
            align-items: center;
        }}
        span.bsr-placeholder {{
            color: {P["text_muted"]};
            font-size: 0.88rem;
            font-style: italic;
        }}
        .bsr-flow-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent 0%, {P["border"]} 50%, transparent 100%);
            margin: 0.6rem 0;
        }}

        /* ─── Logo ─── */
        .school-logo-container {{
            display: flex;
            justify-content: center;
            align-items: center;
            margin-bottom: {S["lg"]};
        }}
        .header-title {{
            font-size: 1.75rem;
            font-weight: 700;
            color: {P["primary"]};
            letter-spacing: -0.03em;
            margin-bottom: {S["sm"]};
        }}

        /* ─── Dividers & Metrics ─── */
        hr {{
            border: none;
            border-top: 1px solid {P["border_light"]};
            margin: {S["lg"]} 0;
            opacity: 0.85;
        }}

        /* ─── Tabs — 4칸 균등 분할 (전체 너비) ─── */
        .stTabs [data-baseweb="tab-list"] {{
            display: flex !important;
            width: 100% !important;
            gap: 6px;
            margin-bottom: 1.25rem;
            padding: 6px;
            background: linear-gradient(180deg, rgba(240, 253, 250, 0.65) 0%, {P["primary_muted"]} 100%);
            border-radius: 14px;
            border: 1px solid {P["border_light"]} !important;
            border-bottom: 1px solid {P["border_light"]} !important;
        }}
        .stTabs [data-baseweb="tab"] {{
            flex: 1 1 0 !important;
            min-width: 0 !important;
            padding: 0.6rem 0.5rem !important;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.82rem;
            color: {P["text_secondary"]};
            border: none !important;
            margin: 0 !important;
            text-align: center !important;
            justify-content: center !important;
            white-space: normal !important;
            line-height: 1.35 !important;
        }}
        .stTabs [data-baseweb="tab"]:hover {{
            color: {P["primary"]};
            background: rgba(255, 255, 255, 0.65);
        }}
        .stTabs [aria-selected="true"] {{
            color: {P["primary"]} !important;
            background: #ffffff !important;
            box-shadow: {P["shadow_sm"]};
        }}
        div[data-baseweb="tab-panel"] {{
            padding-top: {S["sm"]};
            min-height: 120px;
        }}
        /* 탭 안에서 빈 p/markdown 블록 자동 축소 */
        div[data-baseweb="tab-panel"] [data-testid="stMarkdown"] p:empty {{
            display: none;
        }}
        [data-testid="stVerticalBlock"] > [data-testid="element-container"]:empty {{
            display: none;
        }}

        /* ─── Expanders ─── */
        .streamlit-expanderHeader {{
            border-radius: 12px;
            font-weight: 600;
            background: {P["primary_muted"]};
            border: 1px solid {P["border_light"]};
        }}
        details[open] > summary .streamlit-expanderHeader {{
            border-bottom-left-radius: 0;
            border-bottom-right-radius: 0;
        }}

        /* ─── Caption ─── */
        .stCaption {{
            color: {P["text_secondary"]} !important;
            font-size: 0.8rem !important;
            line-height: 1.55 !important;
        }}

        /* ─── 스크롤바 (WebKit) ─── */
        ::-webkit-scrollbar {{
            width: 10px;
            height: 10px;
        }}
        ::-webkit-scrollbar-track {{
            background: {P["border_light"]};
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #cbd5e1;
            border-radius: 8px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #94a3b8;
        }}

        /* ─── Dataframes (메인 컬러 조화) ─── */
        .stDataFrame {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(15, 118, 110, 0.12);
            box-shadow: {P["shadow_sm"]};
        }}
        .stDataFrame thead th {{
            background: linear-gradient(180deg, {P["primary"]} 0%, #14b8a6 100%) !important;
            color: white !important;
            font-weight: 600 !important;
            padding: 0.65rem 0.85rem !important;
        }}
        .stDataFrame tbody td {{
            padding: 0.5rem 0.75rem !important;
        }}

        /* Plotly 차트 영역 */
        [data-testid="stPlotlyChart"] {{
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid {P["border_light"]};
            background: #ffffff;
            box-shadow: {P["shadow_sm"]};
        }}

        /* ─── Metrics (st.metric) ─── */
        [data-testid="stMetric"] {{
            padding: 1rem 1.15rem;
            border-radius: 16px;
            border: 1px solid {P["border_light"]};
            border-left: 4px solid {P["accent"]};
            background: linear-gradient(135deg, #ffffff 0%, rgba(240, 253, 250, 0.5) 100%);
            box-shadow: {P["shadow_sm"]};
        }}
        [data-testid="stMetric"] label {{
            color: {P["text_secondary"]} !important;
            font-weight: 600 !important;
            font-size: 0.8rem !important;
            text-transform: none;
            letter-spacing: 0.02em;
        }}
        [data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: {P["primary"]} !important;
            font-weight: 700 !important;
            font-family: 'DM Sans', 'Noto Sans KR', sans-serif !important;
            font-variant-numeric: tabular-nums;
        }}

        /* ─── Info/Warning Boxes ─── */
        [data-testid="stAlert"] {{
            border-radius: 10px;
        }}

        /* ─── 포트폴리오 A4 인쇄용 ─── */
        .portfolio-print-wrapper {{
            margin: 0;
            padding: 0;
        }}
        .portfolio-a4-card {{
            background: #ffffff;
            padding: 2rem;
            border-radius: 16px;
            box-shadow: {P["shadow_sm"]};
            margin: 1rem 0;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
        }}
        .portfolio-watermark {{
            color: {P["text_muted"]};
            font-size: 0.85rem;
            margin: 0.25rem 0;
        }}
        .portfolio-title {{
            font-size: 1.5rem;
            color: {P["primary"]};
            margin: 0.5rem 0;
        }}
        .portfolio-subtitle {{
            color: {P["text_secondary"]};
            font-size: 0.9rem;
            margin: 0;
        }}
        .portfolio-section h3 {{
            font-size: 1.1rem;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            border-bottom: 2px solid {P["primary"]};
            padding-bottom: 0.25rem;
        }}
        .portfolio-hint {{
            font-size: 0.85rem;
            color: {P["text_muted"]};
            margin: 0;
        }}
        .portfolio-cover-page {{
            page-break-after: always;
        }}
        .portfolio-cover-stats {{
            display: grid;
            grid-template-columns: minmax(260px, 1fr) minmax(260px, 1fr);
            gap: 1.5rem;
            margin-bottom: 1.5rem;
            align-items: start;
        }}
        @media (max-width: 720px) {{
            .portfolio-cover-stats {{
                grid-template-columns: 1fr;
            }}
        }}
        .portfolio-radar {{
            width: 100%;
            min-height: 260px;
            padding: 0.75rem;
            background: rgba(248, 250, 252, 0.9);
            border-radius: 12px;
            border: 1px solid {P["border_light"]};
        }}
        .portfolio-stats-table {{
            width: 100%;
            padding: 1rem 1.1rem;
            background: #ffffff;
            border-radius: 12px;
            border: 1px solid {P["border_light"]};
            box-shadow: {P["shadow_sm"]};
        }}
        .portfolio-stat-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-bottom: 1rem;
        }}
        .portfolio-stat-chips .portfolio-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.88rem;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(15, 118, 110, 0.08);
            color: {P["text"]};
            border: 1px solid rgba(15, 118, 110, 0.12);
        }}
        .portfolio-stat-chips .portfolio-chip strong {{
            color: {P["primary"]};
            font-size: 1.05rem;
        }}
        .portfolio-stat-label {{
            font-size: 0.85rem;
            color: {P["text_secondary"]};
            margin: 0.25rem 0;
        }}
        .portfolio-stat-value {{
            font-size: 1.25rem;
            font-weight: 700;
            color: {P["primary"]};
            margin: 0 0 0.5rem 0;
        }}
        /* 포트폴리오 탭 상단 히어로 (Streamlit markdown 블록) */
        .portfolio-tab-hero {{
            margin-bottom: 1rem;
            padding: 1.15rem 1.35rem;
            border-radius: 16px;
            border: 1px solid {P["border_light"]};
            background: linear-gradient(165deg, #ffffff 0%, rgba(240, 253, 250, 0.45) 100%);
            box-shadow: {P["shadow_sm"]};
        }}
        /* 실습 일지 탭 — NCS 대시보드 카드 */
        .ncs-block {{
            margin-bottom: 1.35rem;
            padding: 1.15rem 1.35rem;
            border-radius: 16px;
            border: 1px solid {P["border_light"]};
            background: linear-gradient(165deg, #ffffff 0%, rgba(240, 253, 250, 0.45) 100%);
            box-shadow: {P["shadow_sm"]};
        }}
        .ncs-block h4 {{
            margin: 0 0 0.85rem 0;
            font-size: 1.05rem;
            font-weight: 700;
            color: {P["primary"]};
            letter-spacing: -0.02em;
        }}
        .portfolio-ncs-table {{
            margin-top: 0.5rem;
            font-size: 0.9rem;
            width: 100%;
            border-collapse: collapse;
        }}
        .portfolio-ncs-table th, .portfolio-ncs-table td {{
            padding: 0.35rem 0.5rem;
            text-align: left;
            border-bottom: 1px solid {P["border_light"]};
        }}
        .portfolio-ncs-table th {{
            color: {P["primary"]};
        }}
        .portfolio-log-entry {{
            margin-bottom: 2rem;
            padding-bottom: 1.5rem;
            border-bottom: 1px dashed {P["border_light"]};
        }}
        .portfolio-log-entry:last-child {{
            border-bottom: none;
        }}
        .portfolio-log-item {{
            margin: 0 0 0.5rem;
            font-size: 0.95rem;
        }}
        .portfolio-evidence-badge {{
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            background: rgba(15, 118, 110, 0.1);
            border-radius: 4px;
            color: {P["primary"]};
            margin-left: 0.5rem;
        }}
        .portfolio-bsr {{
            padding: 1rem;
            background: #f8fafc;
            border-radius: 6px;
            margin-bottom: 0;
            border-left: 4px solid {P["accent"]};
        }}
        /* 지도교사 종합의견 */
        .portfolio-comment {{
            padding: 1.35rem 1.5rem;
            background: linear-gradient(135deg, #f0fdfa 0%, #ecfdf5 50%, #f8fafc 100%);
            border-radius: 12px;
            border: 1px solid rgba(15, 118, 110, 0.15);
            border-left: 5px solid {P["primary"]};
            line-height: 1.75;
            box-shadow: {P["shadow_sm"]};
        }}
        .portfolio-empty {{
            color: {P["text_muted"]};
            font-style: italic;
            padding: 1rem;
        }}
        .portfolio-footer {{
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid {P["border_light"]};
            text-align: center;
            color: {P["text_muted"]};
            font-size: 0.85rem;
        }}

        /* ═══════════════════════════════════════════════════════════
           학생 화면 가독성 개선용 컴포넌트
           - student-page-header : 화면 상단 친근한 안내 헤더
           - stepper : 실습 일지 작성의 Step 1·2·3 진행 인디케이터
           - step-card__head : 각 Step 카드 헤더(번호 원형 + 제목 + 상태)
           - dash-chips / dash-chip : 상단 핵심 지표 칩
           - history-card-grid / history-card : 실습 이력 카드 그리드
           - empty-state / action-strip : 빈 상태 & AI 액션 영역
           ═══════════════════════════════════════════════════════════ */

        .student-page-header {{
            margin: 0 0 1.25rem 0;
            padding: 1.15rem 1.4rem;
            border-radius: 18px;
            border: 1px solid {P["border_light"]};
            background: linear-gradient(135deg, #ffffff 0%, rgba(204, 251, 241, 0.45) 100%);
            box-shadow: {P["shadow_sm"]};
        }}
        .student-page-header__eyebrow {{
            font-size: 0.74rem;
            font-weight: 700;
            color: {P["primary"]};
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin: 0 0 0.35rem 0;
        }}
        .student-page-header__title {{
            font-size: 1.45rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0 0 0.4rem 0;
            letter-spacing: -0.025em;
            line-height: 1.3;
        }}
        .student-page-header__desc {{
            margin: 0;
            color: {P["text_secondary"]};
            font-size: 0.94rem;
            line-height: 1.65;
        }}

        /* ─── Stepper (Step 1·2·3 진행 표시) ─── */
        .stepper {{
            display: flex;
            align-items: center;
            gap: 0;
            margin: 0 0 1.1rem 0;
            padding: 0.85rem 1.1rem;
            border-radius: 14px;
            border: 1px solid {P["border_light"]};
            background: #ffffff;
            box-shadow: {P["shadow_sm"]};
        }}
        .stepper__item {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            flex: 0 0 auto;
        }}
        .stepper__bullet {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            background: #f1f5f9;
            color: #94a3b8;
            font-weight: 700;
            font-size: 0.85rem;
            border: 1.5px solid #e2e8f0;
            transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
            font-family: 'DM Sans', sans-serif;
        }}
        .stepper__label {{
            font-size: 0.88rem;
            font-weight: 600;
            color: #94a3b8;
            letter-spacing: -0.01em;
            transition: color 0.2s ease;
            white-space: nowrap;
        }}
        .stepper__connector {{
            flex: 1 1 auto;
            height: 2px;
            background: #e2e8f0;
            margin: 0 0.85rem;
            border-radius: 999px;
            min-width: 24px;
        }}
        .stepper__connector--done {{
            background: linear-gradient(90deg, {P["primary"]} 0%, {P["accent"]} 100%);
        }}
        .stepper__item--active .stepper__bullet {{
            background: linear-gradient(135deg, {P["primary"]} 0%, {P["accent"]} 100%);
            border-color: {P["primary"]};
            color: #ffffff;
            box-shadow: 0 2px 6px rgba(15, 118, 110, 0.25);
        }}
        .stepper__item--active .stepper__label {{
            color: {P["primary"]};
            font-weight: 700;
        }}
        .stepper__item--done .stepper__bullet {{
            background: rgba(15, 118, 110, 0.14);
            border-color: rgba(15, 118, 110, 0.4);
            color: {P["primary"]};
        }}
        .stepper__item--done .stepper__label {{
            color: {P["primary"]};
        }}

        /* ─── Step 카드 헤더 (st.container(border=True) 내부에서 사용) ─── */
        .step-card__head {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            padding: 0 0 0.75rem 0;
            margin: 0 0 0.85rem 0;
            border-bottom: 1px solid #eef2f7;
        }}
        .step-card__num {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 32px;
            height: 32px;
            border-radius: 10px;
            background: linear-gradient(135deg, {P["primary"]} 0%, {P["accent"]} 100%);
            color: #ffffff;
            font-weight: 700;
            font-size: 0.95rem;
            box-shadow: 0 1px 3px rgba(15, 118, 110, 0.25);
            flex: 0 0 auto;
            font-family: 'DM Sans', sans-serif;
        }}
        .step-card__head-text {{
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
            flex: 1 1 auto;
            min-width: 0;
        }}
        .step-card__title {{
            font-size: 1.02rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            letter-spacing: -0.02em;
            line-height: 1.35;
        }}
        .step-card__sub {{
            margin: 0;
            font-size: 0.82rem;
            color: {P["text_secondary"]};
            line-height: 1.55;
        }}
        .step-card__meta {{
            margin-left: auto;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: #f1f5f9;
            color: #475569;
            font-size: 0.78rem;
            font-weight: 600;
            white-space: nowrap;
            flex: 0 0 auto;
        }}
        .step-card__meta--ok {{
            background: rgba(13, 148, 136, 0.14);
            color: #0f766e;
        }}
        .step-card__meta--warn {{
            background: rgba(251, 191, 36, 0.20);
            color: #92400e;
        }}

        /* ─── 상단 핵심 지표 칩 (대시보드) ─── */
        .dash-chips {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
            margin: 0 0 1.1rem 0;
        }}
        .dash-chip {{
            display: flex;
            flex-direction: column;
            gap: 0.2rem;
            padding: 0.95rem 1.1rem;
            border-radius: 14px;
            border: 1px solid {P["border_light"]};
            background: linear-gradient(165deg, #ffffff 0%, rgba(240, 253, 250, 0.5) 100%);
            box-shadow: {P["shadow_sm"]};
            border-left: 4px solid {P["accent"]};
        }}
        .dash-chip__label {{
            font-size: 0.72rem;
            font-weight: 700;
            color: {P["text_secondary"]};
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}
        .dash-chip__value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: {P["primary"]};
            font-family: 'DM Sans', 'Noto Sans KR', sans-serif;
            letter-spacing: -0.02em;
            line-height: 1.15;
        }}
        .dash-chip__trend {{
            margin-top: 0.15rem;
            font-size: 0.78rem;
            color: {P["text_secondary"]};
            line-height: 1.45;
        }}
        .dash-chip__trend--up {{ color: #0f766e; font-weight: 600; }}
        .dash-chip__trend--down {{ color: #b45309; font-weight: 600; }}

        /* ─── 실습 이력 카드 그리드 ─── */
        .history-card-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 0.85rem;
            margin: 0.5rem 0 1.25rem 0;
        }}
        .history-card {{
            border: 1px solid {P["border_light"]};
            border-radius: 14px;
            background: #ffffff;
            padding: 0.95rem 1.05rem 1rem;
            box-shadow: {P["shadow_sm"]};
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: box-shadow 0.18s ease, transform 0.18s ease, border-color 0.18s ease;
            position: relative;
        }}
        .history-card:hover {{
            box-shadow: 0 6px 14px -4px rgba(15, 118, 110, 0.16);
            border-color: rgba(15, 118, 110, 0.28);
            transform: translateY(-1px);
        }}
        .history-card__top {{
            display: flex;
            align-items: center;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        .history-card__date {{
            display: inline-flex;
            align-items: center;
            gap: 0.3rem;
            font-size: 0.78rem;
            font-weight: 700;
            color: {P["primary"]};
            background: rgba(15, 118, 110, 0.1);
            padding: 0.2rem 0.55rem;
            border-radius: 6px;
            font-variant-numeric: tabular-nums;
        }}
        .history-card__id {{
            font-size: 0.72rem;
            color: {P["text_muted"]};
            font-family: 'DM Sans', monospace;
            margin-left: auto;
        }}
        .history-card__ncs {{
            font-size: 0.94rem;
            font-weight: 700;
            color: #0f172a;
            line-height: 1.35;
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .history-card__excerpt {{
            font-size: 0.85rem;
            color: #475569;
            line-height: 1.55;
            margin: 0;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }}
        .history-card__foot {{
            display: flex;
            gap: 0.4rem;
            flex-wrap: wrap;
            margin-top: auto;
            padding-top: 0.25rem;
        }}
        .history-card__chip {{
            font-size: 0.72rem;
            color: #475569;
            background: #f1f5f9;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
        }}
        .history-card__chip--evidence {{
            color: {P["primary"]};
            background: rgba(15, 118, 110, 0.08);
        }}

        /* ─── 빈 상태 & 액션 스트립 (AI 성장 진단 등에서 활용) ─── */
        .empty-state {{
            text-align: center;
            padding: 1.6rem 1.25rem;
            border-radius: 14px;
            border: 1px dashed #cbd5e1;
            background: rgba(248, 250, 252, 0.7);
            color: {P["text_secondary"]};
        }}
        .empty-state__icon {{
            font-size: 1.85rem;
            margin: 0 0 0.4rem 0;
            line-height: 1;
        }}
        .empty-state__title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #334155;
            margin: 0 0 0.3rem 0;
        }}
        .empty-state__desc {{
            margin: 0;
            font-size: 0.85rem;
            line-height: 1.6;
            color: {P["text_secondary"]};
        }}
        .action-strip {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.85rem;
            padding: 1rem 1.15rem;
            margin: 0 0 0.75rem 0;
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(204,251,241,0.45) 0%, #ffffff 100%);
            border: 1px solid rgba(15,118,110,0.18);
        }}
        .action-strip__text {{
            min-width: 0;
        }}
        .action-strip__title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: #0f172a;
            margin: 0;
            letter-spacing: -0.01em;
        }}
        .action-strip__sub {{
            margin: 0.2rem 0 0;
            font-size: 0.83rem;
            color: {P["text_secondary"]};
            line-height: 1.55;
        }}

        /* AI 성장 진단의 결과 박스 (Gemini 응답을 카드로 감쌈) */
        .growth-result {{
            padding: 1rem 1.15rem;
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(240, 253, 250, 0.55) 0%, #ffffff 100%);
            border: 1px solid rgba(15, 118, 110, 0.16);
            color: #1f2937;
            line-height: 1.7;
            font-size: 0.94rem;
            white-space: pre-wrap;
        }}

        /* ═══════════════════════════════════════════════════════════
           📱 모바일 반응형 (max-width: 768px)
           - 모든 st.columns 자동 세로 스택
           - 버튼 폰 화면 가득, 터치 타겟 확대 (min-height 48px)
           - 인풋 글자 16px 이상 (iOS 줌 방지)
           - 카드/대시보드 칩/그리드 1~2열로 축소
           - Stepper 라벨 숨김(번호 원형만 노출)
           ═══════════════════════════════════════════════════════════ */
        @media (max-width: 768px) {{
            /* ── 페이지 패딩 축소 (작은 화면에서 양옆 여백 낭비 방지) ── */
            div.block-container {{
                padding: 1.25rem 0.75rem 1rem 0.75rem !important;
            }}

            /* ── st.columns 자동 세로 스택: 모든 가로 블록을 위→아래로 ── */
            [data-testid="stHorizontalBlock"] {{
                flex-direction: column !important;
                gap: 0.6rem !important;
            }}
            [data-testid="stHorizontalBlock"] > [data-testid="column"],
            [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {{
                width: 100% !important;
                min-width: 100% !important;
                flex: 1 1 100% !important;
            }}

            /* ── 버튼: 폰 화면 가득 + 손가락 터치 타겟(48px 이상) ── */
            .stButton > button,
            .stDownloadButton > button {{
                width: 100% !important;
                min-height: 48px !important;
                font-size: 1rem !important;
                padding: 0.75rem 1rem !important;
            }}

            /* ── 인풋: iOS Safari 자동 줌 방지를 위해 16px 이상 + 큼직한 패딩 ── */
            .stTextInput input,
            .stTextArea textarea,
            [data-baseweb="select"] > div,
            .stNumberInput input {{
                font-size: 1rem !important;
                padding: 0.75rem 0.9rem !important;
            }}
            .stTextArea textarea {{
                line-height: 1.6 !important;
            }}
            /* 라벨도 살짝 키움 */
            .stTextInput label,
            .stTextArea label,
            .stSelectbox label,
            .stFileUploader label,
            .stCheckbox label,
            .stRadio label {{
                font-size: 0.95rem !important;
            }}

            /* ── 페이지 헤더 ── */
            .student-page-header {{
                padding: 1rem 1.05rem;
                border-radius: 14px;
            }}
            .student-page-header__title {{
                font-size: 1.22rem;
            }}
            .student-page-header__desc {{
                font-size: 0.9rem;
            }}

            /* ── Stepper: 라벨 숨기고 번호 원형만 (좁은 폭에서 깔끔) ── */
            .stepper {{
                padding: 0.65rem 0.75rem;
            }}
            .stepper__label {{
                display: none;
            }}
            .stepper__bullet {{
                width: 32px;
                height: 32px;
                font-size: 0.95rem;
            }}
            .stepper__connector {{
                margin: 0 0.4rem;
                min-width: 16px;
            }}

            /* ── Step 카드 헤더: 상태 배지를 새 줄로 (좁아지면 줄바꿈) ── */
            .step-card__head {{
                flex-wrap: wrap;
                gap: 0.65rem;
            }}
            .step-card__meta {{
                margin-left: 0;
                width: 100%;
                justify-content: center;
            }}
            .step-card__title {{
                font-size: 0.98rem;
            }}

            /* ── 대시보드 칩: 2열로 축소 (4개일 때 2x2 그리드) ── */
            .dash-chips {{
                grid-template-columns: repeat(2, 1fr);
                gap: 0.5rem;
            }}
            .dash-chip {{
                padding: 0.7rem 0.8rem;
            }}
            .dash-chip__value {{
                font-size: 1.2rem;
            }}
            .dash-chip__label {{
                font-size: 0.66rem;
            }}

            /* ── 실습 이력 카드 그리드: 1열로 축소 ── */
            .history-card-grid {{
                grid-template-columns: 1fr;
                gap: 0.6rem;
            }}

            /* ── Action Strip: 세로 정렬 ── */
            .action-strip {{
                flex-direction: column;
                align-items: stretch;
                gap: 0.5rem;
            }}

            /* ── 사이드바: 너무 넓지 않게 + 자동으로 화면을 가리지 않도록 ── */
            section[data-testid="stSidebar"],
            section.stSidebar {{
                min-width: 80vw !important;
                max-width: 88vw !important;
            }}
            section[data-testid="stSidebar"] > div,
            section.stSidebar > div {{
                padding: 1.5rem 0.85rem 1rem 0.85rem !important;
            }}

            /* ── 탭(Tabs) 컴팩트 모드 ── */
            .stTabs [data-baseweb="tab"] {{
                font-size: 0.78rem;
                padding: 0.5rem 0.35rem !important;
            }}

            /* ── 메트릭 / 알림 박스도 패딩 살짝 축소 ── */
            [data-testid="stMetric"] {{
                padding: 0.75rem 0.85rem;
            }}
            [data-testid="stAlert"] {{
                padding: 0.85rem 1rem;
                font-size: 0.92rem;
            }}

            /* ── 데이터프레임/플롯리 차트가 가로로 넘치지 않게 ── */
            .stDataFrame, [data-testid="stPlotlyChart"] {{
                max-width: 100%;
                overflow-x: auto;
            }}

            /* ── 로그인 카드도 컴팩트 ── */
            .login-page-card {{
                padding: 1.15rem 0.95rem 0.85rem;
                border-radius: 12px;
            }}

            /* ── 로고는 모바일에서 80px ── */
            .school-logo-container img {{
                max-width: 80px;
            }}
        }}

        /* ── 더 작은 화면(폰 세로 모드, ~ 480px): 더 빡빡하게 ── */
        @media (max-width: 480px) {{
            .dash-chips {{
                grid-template-columns: 1fr 1fr;
            }}
            .dash-chip__value {{
                font-size: 1.1rem;
            }}
            div.block-container {{
                padding: 1rem 0.6rem 0.85rem 0.6rem !important;
            }}
            .student-page-header__title {{
                font-size: 1.12rem;
            }}
            .stepper__bullet {{
                width: 28px;
                height: 28px;
                font-size: 0.85rem;
            }}
        }}

        /* ─── @media print: 인쇄(PDF 저장) 시 페이지 깔끔하게 넘어가기 ─── */
        @media print {{
            .stApp, body {{
                background: #ffffff !important;
            }}
            .stSidebar {{ display: none !important; }}
            div.block-container {{
                padding: 0 !important;
                max-width: 100% !important;
            }}
            .portfolio-a4-card {{
                box-shadow: none !important;
                border: 1px solid #e2e8f0 !important;
                page-break-inside: auto;
            }}
            .portfolio-print-wrapper {{
                -webkit-print-color-adjust: exact;
                print-color-adjust: exact;
            }}
            .portfolio-cover-page {{
                page-break-after: always;
                page-break-inside: avoid;
            }}
            .portfolio-cover-stats {{
                page-break-inside: avoid;
            }}
            .portfolio-best-practices {{
                page-break-before: auto;
            }}
            .portfolio-log-entry {{
                page-break-inside: avoid;
                break-inside: avoid;
            }}
            .portfolio-log-entry[data-print-break="avoid"] {{
                break-inside: avoid;
            }}
            .portfolio-bsr {{
                break-inside: avoid;
            }}
            .portfolio-section {{
                page-break-after: avoid;
            }}
            .portfolio-section h3 {{
                page-break-after: avoid;
            }}
            [data-testid="stHorizontalBlock"] {{
                break-inside: avoid;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
