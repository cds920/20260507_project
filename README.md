# NCS 포트폴리오 (Streamlit)

## 실행 방법 (Windows / PowerShell)

```powershell
cd "c:\Users\최동수\Desktop\2026\ai_final"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

처음 실행 시 `Email:` 입력이 나오면 **그냥 Enter**를 누르면 됩니다.

## AI 사진 분석 (Google Gemini API)

실습 증거 사진의 AI 분석 기능을 사용하려면 Google AI Studio API 키가 필요합니다.

1. [Google AI Studio](https://aistudio.google.com/apikey)에서 API 키 발급
2. `.streamlit/secrets.toml` 파일 생성 후 아래 내용 추가:
   ```toml
   GOOGLE_API_KEY = "발급받은-API-키"
   ```
3. 또는 환경 변수 `GOOGLE_API_KEY` 설정

예시 파일: `.streamlit/secrets.toml.example`

## 노트북(다른 PC)로 옮길 때

- 이 폴더 전체(`ai_final`)를 그대로 복사합니다. (`.venv`는 굳이 옮기지 않아도 됩니다)
- 새 PC에서 위 실행 방법대로 다시 설치/실행하면 됩니다.

## 접속

- 같은 PC: `http://localhost:8501`
- 같은 와이파이의 휴대폰/다른 기기: `http://<이 PC의 IP>:8501`

## 배포

Streamlit Cloud/ngrok 배포는 `DEPLOY.md`를 참고하세요.

## NCS 능력단위 파일 등록 (선택)

NCS 직무표준을 더 정확히 반영하려면 `constants.py`의 `NCS_DB`를 수정하면 됩니다.
공공 NCS 포털에서 능력단위를 다운받아 아래 형식으로 `constants.py`에 추가할 수 있습니다.

```python
# NCS_DB 형식 예시
"능력단위명": {
    "elements": ["수행요소1", "수행요소2", ...],
    "keywords": ["키워드1", "키워드2", ...],  # 매칭에 사용
},
```

- **구어체 매핑** (`COLLOQUIAL_TO_NCS`)에도 학생들이 자주 쓰는 표현을 추가하면 NCS 용어 변환이 더 잘 됩니다.

