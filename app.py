import streamlit as st

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

# 1. 시스템 초기화 및 세션 안전장치
def init_data():
    # 앱 시작 시 세션 정보가 없으면 None으로 초기화 (에러 방지)
    if "user" not in st.session_state:
        st.session_state.user = None
    if "ncs_progress" not in st.session_state:
        st.session_state.ncs_progress = {}
    if "skills" not in st.session_state:
        st.session_state.skills = {d: 3.0 for d in ["회로", "PLC", "설계", "센서", "안전"]}
    try:
        ensure_default_users()
    except Exception as e:
        detail = str(e)
        lowered = detail.lower()
        transient = any(
            token in lowered
            for token in (
                "503",
                "429",
                "502",
                "504",
                "currently unavailable",
                "rate limit",
                "quota",
            )
        )
        if transient:
            st.error(
                "구글 스프레드시트가 잠시 응답하지 않습니다. "
                "설정 파일이나 권한이 잘못된 것이 아닙니다. "
                "10초 정도 기다렸다가 새로고침해 주세요.\n\n"
                f"상세: {e}"
            )
        else:
            st.error(
                "구글 스프레드시트에 연결하지 못했습니다. `.streamlit/secrets.toml`에 "
                "`GOOGLE_CREDENTIALS`(서비스 계정 JSON)를 설정하고, 해당 서비스 계정 이메일에 "
                "스프레드시트 편집 권한을 공유했는지 확인해 주세요.\n\n"
                f"상세: {e}"
            )
        st.stop()

# --- [페이지] 로그인 ---
def show_login():
    st.markdown(
        """
        <style>
        [data-testid="stMainBlockContainer"] {
            padding-top: 3.5rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _, col_center, _ = st.columns([1, 1.2, 1])
    with col_center:
        st.markdown(
            """
            <div style="text-align: center; margin: 0 0 1.1rem 0;">
                <h2 style="
                    font-family: 'Noto Sans KR', sans-serif;
                    font-size: 1.75rem;
                    font-weight: 700;
                    letter-spacing: -0.03em;
                    color: #1e3a5f;
                    margin: 0 0 0.55rem 0;
                    line-height: 1.35;
                ">🚄 NCS 직무 포트폴리오 시스템</h2>
                <p style="
                    color: #64748b;
                    font-size: 0.95rem;
                    margin: 0;
                    line-height: 1.55;
                ">용산철도고등학교 · 산학일체형 도제학교</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p style="
                text-align: center;
                color: #94a3b8;
                font-size: 0.82rem;
                margin: 0 0 2rem 0;
                line-height: 1.75;
                letter-spacing: 0.01em;
            ">
                AI 실습 일지 분석 &nbsp;·&nbsp;
                직무 역량 추적 &nbsp;·&nbsp;
                생기부 초안 자동 생성
            </p>
            """,
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown(
                "<p style='font-size:1.05rem;font-weight:600;color:#334155;"
                "margin:0 0 1rem 0;letter-spacing:-0.02em;'>로그인</p>",
                unsafe_allow_html=True,
            )
            uid_raw = st.text_input(
                "아이디",
                placeholder="yongsan1",
                key="login_uid",
            )
            upw = st.text_input(
                "비밀번호",
                type="password",
                placeholder="비밀번호를 입력하세요",
                key="login_password",
            )
            if st.button(
                "통합인증 로그인",
                key="login_submit",
                type="primary",
                width="stretch",
                icon=":material/login:",
            ):
                uid_norm = (uid_raw or "").strip().lower()
                try:
                    with st.spinner("로그인 정보를 확인하고 있습니다..."):
                        user = authenticate(uid_norm, (upw or "").strip())
                        if user and user["uid"] != TEACHER_UID:
                            st.session_state.ncs_progress = seed_progress_if_missing(
                                user["uid"],
                                DEFAULT_NCS_PROGRESS,
                            )
                except Exception:
                    st.error(
                        "사용자 정보를 불러오는 중 문제가 발생했습니다.\n"
                        "잠시 후 다시 시도해주세요."
                    )
                else:
                    if user:
                        st.session_state.user = user["uid"]
                        st.rerun()
                    else:
                        st.error("아이디 또는 비밀번호가 일치하지 않습니다.")

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
