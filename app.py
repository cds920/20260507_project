import streamlit as st
from pathlib import Path

from ui_style import apply_advanced_ui, inject_primary_button_text_overrides, render_app_footer
from student_view import show_student
from teacher_view import show_teacher
from constants import DEFAULT_NCS_PROGRESS
from db import (
    TEACHER_UID,
    authenticate,
    ensure_default_users,
    seed_progress_if_missing,
)

# 학교 로고 경로 (assets/school_logo.png 있으면 사용, 없으면 placeholder)
_APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = _APP_DIR / "assets" / "school_logo.png"
LOGO_PLACEHOLDER = _APP_DIR / "assets" / "school_logo_placeholder.svg"


# 1. 시스템 초기화
def init_data():
    ensure_default_users()
    if "user" not in st.session_state:
        st.session_state.user = None
    if "ncs_progress" not in st.session_state:
        st.session_state.ncs_progress = {}
    if "skills" not in st.session_state:
        st.session_state.skills = {d: 3.0 for d in ["회로", "PLC", "설계", "센서", "안전"]}

def _logo_path():
    """학교 로고 경로: school_logo.png 있으면 사용, 없으면 placeholder SVG 사용"""
    return str(LOGO_PATH) if LOGO_PATH.exists() else str(LOGO_PLACEHOLDER)


# --- [페이지] 로그인 ---
def show_login():
    logo_path = _logo_path()
    st.markdown(
        '<div class="login-page-outer">'
        '<div class="login-page-card">',
        unsafe_allow_html=True,
    )
    _, col_center, _ = st.columns([1, 2.2, 1])
    with col_center:
        st.image(logo_path, width=128)
        st.markdown(
            "<h1 class='login-title'>NCS 직무 포트폴리오</h1>"
            "<p class='login-subtitle'>용산철도고등학교 · 산학일체형 도제학교 · 전기·전자 실습 기록</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    col1, col2, col3 = st.columns([1, 1.25, 1])
    with col2:
        uid_raw = st.text_input("아이디 (yongsan1~yongsan10 / teacher)")
        upw = st.text_input("비밀번호", type="password")
        if st.button("통합인증 로그인", width="stretch"):
            # 대소문자 구분 없이 인증 (예: Yongsan1, YONGSAN1, TEACHER 모두 허용)
            uid_norm = (uid_raw or "").strip().lower()
            user = authenticate(uid_norm, (upw or "").strip())
            if user:
                st.session_state.user = user["uid"]  # DB 표준 UID로 세션 저장
                if user["uid"] != TEACHER_UID:
                    # 최초 로그인 시에는 모든 진도를 0으로 시작하고,
                    # 실습일지가 저장될 때마다 증가시키도록 설정.
                    st.session_state.ncs_progress = seed_progress_if_missing(
                        user["uid"],
                        DEFAULT_NCS_PROGRESS,
                    )
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 일치하지 않습니다.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    render_app_footer()


# 실행부
st.set_page_config(
    page_title="NCS 직무 포트폴리오",
    page_icon="📘",
    layout="wide",
    # "auto"는 모바일에서 기본 접힘 → 열기 버튼이 가려지면 메뉴를 못 씀. 항상 펼친 상태로 시작.
    initial_sidebar_state="expanded",
)

apply_advanced_ui()
init_data()

if st.session_state.user is None:
    show_login()
else:
    uid = st.session_state.user

    # --- 사이드바 상단: 로고 ---
    with st.sidebar:
        st.image(_logo_path(), width=120)

    # --- 메인 영역 + 사이드바 중단(view가 직접 주입: 프로필 + 메뉴) ---
    if uid == TEACHER_UID:
        show_teacher()
    else:
        show_student(uid)

    # --- 사이드바 하단: 로그아웃 (NCS 이수 현황 중복 제거, 메트릭 블록 모두 삭제) ---
    with st.sidebar:
        st.divider()
        if st.button("로그아웃", width="stretch"):
            st.session_state.user = None
            st.rerun()
    render_app_footer()

# 테마 CSS보다 늦게 primary 버튼 글자색 고정 (폼 제출·일반 primary 공통)
inject_primary_button_text_overrides()

