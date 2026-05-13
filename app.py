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

# 학교 로고 경로
_APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = _APP_DIR / "assets" / "school_logo.png"
LOGO_PLACEHOLDER = _APP_DIR / "assets" / "school_logo_placeholder.svg"

# 1. 시스템 초기화 및 세션 안전장치
def init_data():
    ensure_default_users()
    # 앱 시작 시 세션 정보가 없으면 None으로 초기화 (에러 방지 핵심)
    if "user" not in st.session_state:
        st.session_state.user = None
    if "ncs_progress" not in st.session_state:
        st.session_state.ncs_progress = {}
    if "skills" not in st.session_state:
        st.session_state.skills = {d: 3.0 for d in ["회로", "PLC", "설계", "센서", "안전"]}

def _logo_path():
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
            "<p class='login-subtitle'>용산철도고등학교 · 산학일체형 도제학교</p>",
            unsafe_allow_html=True,
        )
    st.divider()

    col1, col2, col3 = st.columns([1, 1.25, 1])
    with col2:
        uid_raw = st.text_input("아이디 (yongsan1~yongsan10 / teacher)")
        upw = st.text_input("비밀번호", type="password")
        # 버튼 속성 수정 (use_container_width 사용)
        if st.button("통합인증 로그인", use_container_width=True):
            uid_norm = (uid_raw or "").strip().lower()
            user = authenticate(uid_norm, (upw or "").strip())
            if user:
                st.session_state.user = user["uid"]
                if user["uid"] != TEACHER_UID:
                    st.session_state.ncs_progress = seed_progress_if_missing(
                        user["uid"],
                        DEFAULT_NCS_PROGRESS,
                    )
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 일치하지 않습니다.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    render_app_footer()

# --- 실행부 ---
st.set_page_config(
    page_title="NCS 직무 포트폴리오",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# UI 스타일 적용 및 데이터 초기화
apply_advanced_ui()
init_data()

# [중요] 로그인 상태 체크 (이곳이 안전장치의 핵심 위치입니다!)
if st.session_state.get('user') is None:
    show_login()
    st.stop()  # 로그인이 안 되어 있으면 여기서 실행 중단! (뒤쪽 에러 원천 차단)

# --- 로그인 성공 시 실행되는 구역 ---
uid = st.session_state.user

# 사이드바 상단 로고
with st.sidebar:
    st.image(_logo_path(), width=120)

# 권한별 화면 분구
if uid == TEACHER_UID:
    show_teacher()
else:
    show_student(uid)

# 사이드바 하단 로그아웃
with st.sidebar:
    st.divider()
    if st.button("로그아웃", use_container_width=True):
        st.session_state.user = None
        st.rerun()

render_app_footer()
inject_primary_button_text_overrides()
