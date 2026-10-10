"""
01_dh_caffe_crawling_with_auto_login.py

네이버 자동로그인 전략 (3단계):
  1. 저장된 쿠키로 세션 복원 (가장 빠름 - 최초 1회 이후)
  2. pyautogui OS 레벨 키보드 입력으로 로그인 (Selenium 감지 우회) 
  2. 클립보드+JS 방식 자동 로그인 → 이미지 퀴즈는 Gemini Vision으로 자동 풀기
  3. 위 두 방법 모두 실패 시, 수동 로그인 대기 (fallback)

드라이버: webdriver_manager로 현재 Chrome 버전에 맞는 chromedriver 자동 설치
쿠키 파일: ./naver_cookies.pkl (최초 로그인 성공 시 자동 저장)
"""

import os
import io
import sys
import base64
import json
import re
import time
import pickle
import random
from datetime import date, timedelta

import numpy as np
import pyautogui
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from webdriver_manager.chrome import ChromeDriverManager

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from web.fetcher import fetch_draw  # noqa: E402  (공홈 API — 등수별 상세 조회용)

# 계정/키는 코드에 두지 않고 프로젝트 루트 .env 에서 읽는다 (NAVER_ID, NAVER_PW, GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    print("[경고] python-dotenv 미설치 → .env 를 읽지 못함 (pip install python-dotenv). "
          "쿠키 만료 시 자동로그인 불가")

# ─────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────
NAVER_ID = os.getenv("NAVER_ID", "")
NAVER_PW = os.getenv("NAVER_PW", "")
LOGIN_URL  = "https://nid.naver.com/nidlogin.login"
COOKIE_PATH = os.path.join(os.path.dirname(__file__), "naver_cookies.pkl")
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "../data/history_from_cafe.csv")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # https://aistudio.google.com/apikey

# 공홈 API(등수별 상세) 컬럼 — scripts/sync_cafe_history.py의 DETAIL_FIELDS와 순서 동일해야 함
# 뒤에 추가할 땐 여기 끝에 이어붙일 것 (마이그레이션이 컬럼 수 차이만큼 뒤에서부터 채움)
DETAIL_COLUMNS = [
    "winners_1", "winners_2", "winners_3", "winners_4", "winners_5",
    "total_prize_1", "total_prize_2", "total_prize_3", "total_prize_4", "total_prize_5",
    "total_sales",
    "prize_1", "prize_2", "prize_3", "prize_4", "prize_5",
]

# ─────────────────────────────────────────────
# 추첨일 유틸
# ─────────────────────────────────────────────
ROUND1_DATE = date(2002, 12, 7)   # 1회 추첨일 (토요일)

def round_to_date(round_num: int) -> str:
    """회차 번호 → 추첨일 문자열 (YYYY-MM-DD)"""
    return (ROUND1_DATE + timedelta(weeks=round_num - 1)).strftime("%Y-%m-%d")

def check_needs_update(history_path: str) -> bool:
    """
    CSV를 읽어 마지막 추첨일을 확인하고, 새 회차 데이터가 있으면 True 반환.
    브라우저 없이 파일만으로 판단.

    갱신 필요 조건:
      오늘 날짜 >= 마지막 추첨일 + 7일 (다음 토요일 추첨이 이미 지남)
    """
    if not os.path.exists(history_path):
        print("[갱신 체크] 히스토리 파일 없음 → 크롤링 필요")
        return True

    try:
        with open(history_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        if not lines:
            print("[갱신 체크] 파일이 비어 있음 → 크롤링 필요")
            return True

        # 컬럼 수로 draw_date 포함 여부 판단 (10=구버전, 21=현재. draw_date 위치는 동일)
        cols = lines[-1].split(",")
        if len(cols) >= 10:          # ball_set, round, draw_date, n1~n6, bonus, [등수별 상세...]
            last_date_str = cols[2].strip()
            last_draw = date.fromisoformat(last_date_str)
        elif len(cols) == 9:         # 구버전 (draw_date 없음) → 회차로 역산
            last_round = int(cols[1].strip())
            last_draw  = ROUND1_DATE + timedelta(weeks=last_round - 1)
        else:
            print("[갱신 체크] 알 수 없는 CSV 형식 → 크롤링 필요")
            return True

        next_draw = last_draw + timedelta(weeks=1)
        today     = date.today()
        needs     = today >= next_draw

        print(f"[갱신 체크] 마지막 추첨일: {last_draw}  |  다음 추첨일: {next_draw}  |  오늘: {today}")
        if needs:
            print("[갱신 체크] 새 회차 데이터 있음 → 크롤링 필요")
        else:
            print("[갱신 체크] 최신 데이터 보유 중 → 크롤링 불필요")
        return needs

    except Exception as e:
        print(f"[갱신 체크] 오류 ({e}) → 안전하게 크롤링 진행")
        return True

def migrate_csv_add_date(history_path: str):
    """
    구버전 CSV (9컬럼)를 draw_date 포함 10컬럼으로 변환.
    이미 10컬럼이면 아무것도 하지 않음.
    """
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return
    if len(lines[0].split(",")) >= 10:
        return   # 이미 draw_date 있음 (10=구버전, 26=등수별 상세 포함 현재 형식)

    print("[마이그레이션] draw_date 컬럼 추가 중...")
    new_lines = []
    for line in lines:
        cols = line.split(",")
        if len(cols) != 9:
            new_lines.append(line)
            continue
        round_num = int(cols[1].strip())
        draw_date = round_to_date(round_num)
        new_cols  = [cols[0], cols[1], draw_date] + cols[2:]
        new_lines.append(",".join(new_cols))

    with open(history_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"[마이그레이션] 완료: {len(new_lines)}행 변환")

def migrate_csv_add_prize_details(history_path: str):
    """
    구버전 CSV(DETAIL_COLUMNS 전체 컬럼 수보다 적으면)에 부족한 만큼만 빈 칸 추가.
    DETAIL_COLUMNS 뒤에 새 항목이 추가되어 목표 폭이 늘어난 경우(예: winners_*/
    total_prize_*/total_sales만 있던 21컬럼 파일에 prize_1~5가 새로 추가된 경우)에도
    이미 채워진 값은 그대로 두고 모자란 컬럼 수만큼만 이어붙인다.
    """
    if not os.path.exists(history_path):
        return
    with open(history_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return

    current_cols = len(lines[0].split(","))
    target_cols = 10 + len(DETAIL_COLUMNS)
    if current_cols >= target_cols:
        return   # 이미 마이그레이션됨

    missing = target_cols - current_cols
    print(f"[마이그레이션] 등수별 상세 컬럼 {missing}개 추가 중...")
    blank = "," * missing
    new_lines = [line + blank for line in lines]
    with open(history_path, "w") as f:
        f.write("\n".join(new_lines) + "\n")
    print(f"[마이그레이션] 완료: {len(new_lines)}행 변환")

# ─────────────────────────────────────────────
# 드라이버 생성 (webdriver_manager로 Chrome 버전 자동 매칭)
# ─────────────────────────────────────────────
def create_driver():
    options = Options()
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1280,900")
    # 봇 탐지 우회를 위한 옵션
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    )
    # webdriver_manager가 현재 Chrome 버전(145)에 맞는 driver를 자동 설치
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    # navigator.webdriver 속성 숨기기 (봇 감지 우회)
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"}
    )
    return driver


# ─────────────────────────────────────────────
# 전략 1: 쿠키로 세션 복원
# ─────────────────────────────────────────────
def login_with_cookies(driver) -> bool:
    """저장된 쿠키 파일이 있으면 로드하고 로그인 상태를 반환."""
    if not os.path.exists(COOKIE_PATH):
        print("[쿠키] 쿠키 파일 없음 → 건너뜀")
        return False

    print("[쿠키] 쿠키 로드 시도...")
    # 쿠키는 해당 도메인에 접속한 상태에서만 추가 가능
    driver.get("https://www.naver.com")
    time.sleep(1)

    with open(COOKIE_PATH, "rb") as f:
        cookies = pickle.load(f)
    for cookie in cookies:
        # expiry 필드가 float이면 int로 변환 (selenium 요구사항)
        if "expiry" in cookie:
            cookie["expiry"] = int(cookie["expiry"])
        try:
            driver.add_cookie(cookie)
        except Exception:
            pass

    driver.refresh()
    time.sleep(2)

    # 현재 naver.com에 있으므로 _is_logged_in이 재로드하지 않음
    if _is_logged_in(driver):
        print("[쿠키] 쿠키 로그인 성공!")
        return True
    else:
        print("[쿠키] 쿠키 만료 → 재로그인 필요")
        return False


# ─────────────────────────────────────────────
# 전략 2: pyautogui OS 레벨 키보드 입력
# ─────────────────────────────────────────────
def _set_input_value(driver, element, value: str):
    """
    React 기반 네이버 로그인 폼에 값을 입력하는 강화된 방식.
    1) 필드를 먼저 비운 뒤 JavaScript nativeInputValueSetter로 값 설정
    2) input/change 이벤트 dispatch로 React synthetic event 발생
    """
    driver.execute_script("""
        var nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value').set;
        nativeInputValueSetter.call(arguments[0], '');
        arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
        nativeInputValueSetter.call(arguments[0], arguments[1]);
        arguments[0].dispatchEvent(new Event('input', { bubbles: true }));
        arguments[0].dispatchEvent(new Event('change', { bubbles: true }));
    """, element, value)
    time.sleep(0.2)


def login_with_pyautogui(driver) -> bool:
    """
    클립보드 붙여넣기 + JavaScript React input setter 방식으로 로그인.
    Selenium send_keys() 이벤트 패턴을 완전히 우회합니다.
    """
    if not NAVER_ID or not NAVER_PW:
        print("[자동로그인] .env 에 NAVER_ID / NAVER_PW 가 없음 → 자동로그인 건너뜀")
        return False
    print("[자동로그인] 클립보드+JS 방식으로 로그인 시도...")
    driver.get(LOGIN_URL)
    time.sleep(2)

    # 로그인 유지 체크박스 클릭
    try:
        keep_checkbox = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "keep"))
        )
        driver.execute_script("arguments[0].click();", keep_checkbox)
        time.sleep(0.3)
    except Exception:
        pass

    # ── ID 입력 ──
    try:
        id_field = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "id"))
        )
        id_field.click()
        time.sleep(0.3)
        _set_input_value(driver, id_field, NAVER_ID)
        print(f"  ID 입력 확인: '{id_field.get_attribute('value')}'")
        time.sleep(random.uniform(0.3, 0.7))
    except Exception as e:
        print(f"[자동로그인] ID 입력 실패: {e}")
        return False

    # ── PW 입력 ──
    try:
        pw_field = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "pw"))
        )
        pw_field.click()
        time.sleep(0.3)
        _set_input_value(driver, pw_field, NAVER_PW)
        print(f"  PW 입력 확인: {'*' * len(pw_field.get_attribute('value'))}")
        time.sleep(random.uniform(0.3, 0.7))
    except Exception as e:
        print(f"[자동로그인] PW 입력 실패: {e}")
        return False

    # ── 로그인 버튼 클릭 ──
    try:
        login_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "log.login"))
        )
        driver.execute_script("arguments[0].click();", login_btn)
        print("  로그인 버튼 클릭 완료")
    except Exception:
        pyautogui.press("enter")

    time.sleep(4)

    # 디버그: 현재 화면 스크린샷 저장 (퀴즈 화면 여부 확인용)
    try:
        cur_url = driver.current_url or ""
        page_src = driver.page_source
        driver.save_screenshot("login_after_click.png")
        print(f"  현재 URL: {cur_url}")
        print(f"  디버그 스크린샷 저장: login_after_click.png")
    except Exception:
        cur_url = ""
        page_src = ""

    # ── 브라우저 등록 팝업 자동 처리 ──
    # 네이버: 새 브라우저 첫 로그인 시 "이 브라우저를 등록하시겠습니까?" 팝업 표시
    _click_browser_register(driver)
    # 등록 후 잠시 대기 및 상태 갱신
    time.sleep(1)
    try:
        cur_url = driver.current_url or ""
        page_src = driver.page_source
    except Exception:
        pass

    # 이미지 퀴즈 감지 및 자동 풀기 (URL + 소스 이중 감지)
    if _detect_quiz(page_src, url=cur_url):
        print("[자동로그인] 이미지 퀴즈 화면 감지 → Gemini Vision으로 자동 풀기 시도...")
        if not solve_quiz_with_gemini(driver):
            # 자동 풀기 실패: 퀴즈 화면을 그대로 유지하고 수동 대기
            print("[자동로그인] 퀴즈 자동 풀기 실패 → 현재 퀴즈 화면에서 수동 풀기 대기 (60초)")
            deadline = time.time() + 60
            while time.time() < deadline:
                time.sleep(2)
                if _is_logged_in(driver):
                    print("[자동로그인] 수동 퀴즈 풀기 감지! 쿠키 저장 중...")
                    _save_cookies(driver)
                    return True
                if not _detect_quiz(driver.page_source, url=driver.current_url or ""):
                    break  # 퀴즈 화면이 사라졌으면 루프 탈출
                print(f"  퀴즈 대기 중... ({int(deadline - time.time())}초 남음)", end="\r")
            print()
        time.sleep(2)

    # 캡챠 / 추가인증 화면 감지
    cur_url = driver.current_url
    page_src = driver.page_source
    if "captcha" in cur_url or "보안문자" in page_src:
        print("[자동로그인] 캡챠 감지됨 → 수동로그인 fallback으로 전환")
        return False
    if "ndynamic" in cur_url or "nid.naver.com/login/ext" in cur_url:
        print("[자동로그인] 추가 인증(OTP/SMS) 요청됨 → 수동로그인 fallback으로 전환")
        return False

    if _is_logged_in(driver):
        print("[자동로그인] 로그인 성공! 쿠키 저장 중...")
        _save_cookies(driver)
        return True

    print("[자동로그인] 로그인 실패 (봇 탐지 또는 잘못된 계정 정보)")
    return False


# ─────────────────────────────────────────────
# 브라우저 등록 팝업 자동 처리
# ─────────────────────────────────────────────
def _click_browser_register(driver):
    """
    네이버 첫 로그인 시 나타나는 '이 브라우저를 등록하시겠습니까?' 팝업 처리.
    '등록' 버튼을 자동으로 클릭합니다. 팝업이 없으면 조용히 넘어갑니다.
    """
    # 등록 버튼 셀렉터 목록 (네이버 버전에 따라 다를 수 있음)
    register_selectors = [
        # 텍스트 기반
        "//button[contains(text(), '등록')]",
        "//a[contains(text(), '등록')]",
        "//input[@value='등록']",
        # ID/클래스 기반 (실제 소스 확인 후 필요 시 추가)
        "#btnRegister",
        ".btn_register",
        "button.btn_confirm",
    ]

    for selector in register_selectors:
        try:
            if selector.startswith("//"):
                # XPath
                btn = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.XPATH, selector))
                )
            else:
                # CSS
                btn = WebDriverWait(driver, 2).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
            driver.execute_script("arguments[0].click();", btn)
            print(f"[브라우저등록] '등록' 버튼 클릭 완료 (selector: {selector})")
            time.sleep(1)
            return
        except Exception:
            continue

    # 버튼을 못 찾았을 때: 페이지 소스에 '등록' 관련 텍스트가 있으면 로그 출력
    try:
        src = driver.page_source
        if "등록" in src and "브라우저" in src:
            print("[브라우저등록] 등록 팝업이 감지되었으나 버튼을 클릭하지 못했습니다.")
            print("              login_after_click.png 스크린샷을 확인하세요.")
    except Exception:
        pass


# ─────────────────────────────────────────────
# 이미지 퀴즈 자동 풀기 (Gemini Vision)
# ─────────────────────────────────────────────
def _detect_quiz(page_source: str, url: str = "") -> bool:
    """네이버 보안 퀴즈 화면 여부 감지.
    실제 네이버 퀴즈 페이지 URL/코드를 소스에서 분석해서 만든 키워드.
    """
    # URL 기반 감지 (가장 신뢰도 높음)
    quiz_urls = [
        "nid.naver.com/user2/help",
        "nid.naver.com/login/protect",
        "nid.naver.com/nidlogin.loginSso",
        "/login/quiz", "/quiz", "/protect",
    ]
    if any(q in url for q in quiz_urls):
        return True

    # 페이지 소스 기반 감지
    keywords = [
        # 한국어 키워드
        "보안 퀴즈", "부가인증", "추가 인증",
        "내가 선택한 정보", "회원정보 확인",
        "로그인 확인", "보안 확인",
        # HTML 클래스/ID 패턴
        "quiz_wrap", "quizWrap", "quiz-wrap",
        "protect_wrap", "protectWrap",
        "btn_confirm_quiz", "quiz_answer",
        # 자바스크립트 패턴
        "loginProtect", "loginQuiz",
    ]
    return any(k in page_source for k in keywords)


def solve_quiz_with_gemini(driver) -> bool:
    """
    Gemini Vision API로 네이버 로그인 퀴즈를 자동으로 풀기.

    동작 방식:
      1. 퀴즈 영역 스크린샷 캡처
      2. Gemini에 이미지 + 질문 전달: '정답 번호(1/2/3)만 반환해줘'
      3. 반환된 번호에 해당하는 라디오버튼 클릭 후 확인 버튼 누름
    """
    if not GEMINI_API_KEY:
        print("[퀴즈] GEMINI_API_KEY 미설정 → 퀴즈 자동 풀기 불가")
        print("       https://aistudio.google.com/apikey 에서 발급 후")
        print("       파일 상단 GEMINI_API_KEY 변수에 입력하세요.")
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
    except ImportError:
        print("[퀴즈] google-generativeai 패키지 없음. pip install google-generativeai")
        return False
    except Exception as e:
        print(f"[퀴즈] Gemini 초기화 실패: {e}")
        return False

    try:
        # ── 1. 퀴즈 영역 스크린샷 캡처 ──
        # 전체 페이지 스크린샷 후 PIL로 처리
        screenshot_bytes = driver.get_screenshot_as_png()
        img = Image.open(io.BytesIO(screenshot_bytes))

        # base64로 변환하여 Gemini에 전달
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode()

        # ── 2. Gemini Vision으로 정답 분석 ──
        prompt = (
            "이 이미지는 네이버 로그인 보안 퀴즈 화면입니다.\n"
            "화면에 표시된 질문과 보기를 읽고, 정답에 해당하는 보기의 번호(1, 2, 3 중 하나)만 "
            "숫자 하나로만 답해주세요. 다른 설명은 하지 마세요.\n"
            "예시 답변: 2"
        )

        response = model.generate_content([
            prompt,
            {"mime_type": "image/png", "data": img_b64}
        ])
        answer_text = response.text.strip()
        print(f"[퀴즈] Gemini 분석 결과: '{answer_text}'")

        # 숫자만 추출
        import re
        digits = re.findall(r"\d", answer_text)
        if not digits:
            print("[퀴즈] Gemini가 유효한 번호를 반환하지 않음")
            return False
        answer_num = int(digits[0])  # 첫 번째 숫자 사용
        print(f"[퀴즈] 선택할 보기 번호: {answer_num}")

        # ── 3. 해당 번호 라디오버튼 클릭 ──
        # 네이버 퀴즈: <input type="radio" name="answer" value="1|2|3">
        radio_selectors = [
            f"input[type='radio'][value='{answer_num}']",
            f"input[type='radio']:nth-of-type({answer_num})",
            f"label:nth-of-type({answer_num})",
        ]
        clicked = False
        for selector in radio_selectors:
            try:
                radio = WebDriverWait(driver, 3).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                driver.execute_script("arguments[0].click();", radio)
                clicked = True
                print(f"[퀴즈] 보기 {answer_num} 클릭 완료")
                break
            except Exception:
                continue

        if not clicked:
            print("[퀴즈] 라디오버튼을 찾지 못함 → 확인 버튼만 클릭 시도")

        # 확인/제출 버튼 클릭
        time.sleep(0.5)
        submit_selectors = [
            "button[type='submit']",
            "input[type='submit']",
            ".btn_confirm", ".btn_next", "#btnNext",
        ]
        for selector in submit_selectors:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, selector)
                driver.execute_script("arguments[0].click();", btn)
                print("[퀴즈] 제출 버튼 클릭 완료")
                break
            except Exception:
                continue

        time.sleep(2)
        return True

    except Exception as e:
        print(f"[퀴즈] 자동 풀기 중 오류: {e}")
        # 디버그용 스크린샷 저장
        try:
            driver.save_screenshot("quiz_debug.png")
            print("[퀴즈] 디버그 스크린샷 저장: quiz_debug.png")
        except Exception:
            pass
        return False


# ─────────────────────────────────────────────
# Fallback: 수동 로그인 대기
# ─────────────────────────────────────────────
def login_with_manual_wait(driver, timeout=120) -> bool:
    """
    자동 로그인이 모두 실패한 경우 사용자가 직접 로그인할 때까지 대기.
    현재 페이지(퀴즈/로그인 화면)를 그대로 유지하므로 사용자가 이어서 처리 가능.
    성공 시 쿠키를 저장하여 다음 실행부터는 쿠키 로그인으로 처리.
    """
    # 현재 페이지가 로그인 관련 페이지가 아닌 경우에만 로그인 페이지로 이동
    if "nid.naver.com" not in driver.current_url:
        driver.get(LOGIN_URL)
    print(f"\n[수동] {timeout}초 안에 브라우저에서 직접 로그인해 주세요...")

    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(2)
        if _is_logged_in(driver):
            print("[수동] 수동 로그인 감지! 쿠키 저장 중...")
            _save_cookies(driver)
            return True
        remaining = int(deadline - time.time())
        print(f"  로그인 대기 중... ({remaining}초 남음)", end="\r")

    print("\n[수동] 타임아웃 → 로그인 실패")
    return False


# ─────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────
def _is_logged_in(driver) -> bool:
    """쿠키 딕셔너리에서 NID_SES(로그인 세션 쿠키) 존재 여부로 빠르게 확인."""
    try:
        current_url = driver.current_url or ""
    except Exception:
        return False
    # naver.com 도메인 쿠키가 없으면 일단 이동
    if "naver.com" not in current_url:
        driver.get("https://www.naver.com")
        time.sleep(1)
    try:
        cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
    except Exception:
        return False
    # NID_SES: 로그인 세션 / NID_AUT: 인증 토큰 — 둑 다 로그인 할 때만 존재
    return "NID_SES" in cookies or "NID_AUT" in cookies


def _save_cookies(driver):
    """현재 세션의 쿠키를 파일로 저장. 이미 naver.com이면 재로드 생략."""
    try:
        current_url = driver.current_url or ""
    except Exception:
        current_url = ""
    if "naver.com" not in current_url:
        driver.get("https://www.naver.com")
        time.sleep(1)
    with open(COOKIE_PATH, "wb") as f:
        pickle.dump(driver.get_cookies(), f)
    print(f"[쿠키] 쿠키 저장 완료: {COOKIE_PATH}")


def naver_login(driver) -> bool:
    """3단계 로그인 전략을 순서대로 시도."""
    if login_with_cookies(driver):
        return True
    if login_with_pyautogui(driver):
        return True
    if login_with_manual_wait(driver):
        return True
    return False


# ─────────────────────────────────────────────
# 공홈 API — 등수별 상세(당첨게임수/총당첨금/총판매금액) 조회
# ─────────────────────────────────────────────
def _fetch_official_details(round_no: int) -> np.ndarray:
    """실제 당첨 행에만 채울 등수별 상세 값. 조회 실패 시 빈 값(11개)."""
    try:
        data = fetch_draw(round_no)
    except Exception as e:
        print(f"[공홈] {round_no}회차 조회 실패: {e}")
        data = None

    if not data:
        return np.array([""] * len(DETAIL_COLUMNS))

    return np.array([
        "" if data.get(col) is None else str(data[col])
        for col in DETAIL_COLUMNS
    ])


# ─────────────────────────────────────────────
# 메인 크롤링 로직 (원본과 동일)
# ─────────────────────────────────────────────
CAFE_ID = 29572332          # 동행복권 커뮤니티
MENU_ID = 22                # [로또복권] 추첨리허설
# 네이버 카페 새 UI(/f-e/...)로 바뀌며 목록 페이지의 cafe_main iframe 이 사라짐 → JSON API 로 목록 수집
LIST_API = (
    "https://apis.naver.com/cafe-web/cafe2/ArticleListV2dot1.json"
    "?search.clubid={cafe}&search.menuid={menu}&search.page={page}&search.perPage=15"
)
# 본문: 새 UI 주소 우선, 안 되면 예전 iframe 주소
ARTICLE_URLS = (
    "https://cafe.naver.com/f-e/cafes/{cafe}/articles/{aid}?menuid={menu}&referrerAllArticles=false",
    "https://cafe.naver.com/dhlottery?iframe_url_utf8=%2FArticleRead.nhn"
    "%253Fclubid%3D{cafe}%2526page%3D1%2526menuid%3D{menu}%2526boardtype%3DL"
    "%2526articleid%3D{aid}%2526referrerAllArticles%3Dfalse",
)
CONTAINER_CLASSES = ("se-main-container", "ContentRenderer")
ROUND_IN_SUBJECT = re.compile(r"로또6/45\s*제\s*(\d+)\s*회")


def _get_json(driver, url: str) -> dict:
    """로그인된 브라우저로 JSON API 호출 (쿠키 재사용). 로그인 페이지 등 JSON 이 아니면 RuntimeError."""
    driver.get(url)
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    pre = driver.find_elements(By.TAG_NAME, "pre")       # Chrome JSON 뷰어는 <pre> 안에 원문을 둔다
    text = pre[0].text if pre else driver.find_element(By.TAG_NAME, "body").text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise RuntimeError(f"JSON 이 아닌 응답 (로그인 만료?) — {url}\n  응답 앞부분: {text[:200]!r}")


MAX_LIST_PAGES = 200   # 15개 × 200 = 3,000글. 무한 순회 방지용 상한


def list_articles(driver) -> np.ndarray:
    """
    리허설 게시판 글 목록 → [[articleId, 회차], ...] (최신 글 먼저).
    같은 회차 글이 여러 개면 최신 글 하나만 남긴다.
    """
    rows, seen_rounds, seen_ids, page = [], set(), set(), 1
    while page <= MAX_LIST_PAGES:
        data = _get_json(driver, LIST_API.format(cafe=CAFE_ID, menu=MENU_ID, page=page))
        msg = data.get("message", {})
        if str(msg.get("status")) != "200" or "result" not in msg:
            raise RuntimeError(f"카페 목록 API 오류 (page={page}): {str(msg.get('error') or msg)[:200]}")
        result = msg["result"]
        articles = result.get("articleList") or []
        new_ids = [a["articleId"] for a in articles if a["articleId"] not in seen_ids]
        if not new_ids:                     # 빈 페이지 / 같은 페이지 반복 → 끝
            break
        seen_ids.update(new_ids)
        for a in articles:
            m = ROUND_IN_SUBJECT.search(a.get("subject", ""))
            if not m or m.group(1) in seen_rounds:
                continue
            seen_rounds.add(m.group(1))
            rows.append([str(a["articleId"]), m.group(1)])
        if not result.get("hasNext"):
            break
        page += 1
        time.sleep(random.uniform(0.5, 1.2))
    else:
        print(f"[목록] 페이지 상한({MAX_LIST_PAGES}) 도달 — 여기까지만 사용")
    print(f"[목록] 회차 글 {len(rows)}개 ({page}페이지)")
    return np.array(rows, dtype=str).reshape(-1, 2)


def open_article(driver, article_id: str, timeout: float = 15):
    """
    글 본문 컨테이너를 찾아 반환. 새 UI(iframe 없음)와 예전 UI(cafe_main iframe) 모두 지원.
    못 찾으면 TimeoutException.
    """
    for tmpl in ARTICLE_URLS:
        driver.get(tmpl.format(cafe=CAFE_ID, menu=MENU_ID, aid=article_id))
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                driver.switch_to.default_content()
                frames = driver.find_elements(By.ID, "cafe_main")
                if frames:
                    driver.switch_to.frame(frames[0])
                for cls in CONTAINER_CLASSES:
                    found = driver.find_elements(By.CLASS_NAME, cls)
                    if found and found[0].text.strip():
                        return found[0]
            except WebDriverException:      # 렌더링 중 stale element / frame 전환 실패 → 다시 시도
                pass
            time.sleep(0.5)
    raise TimeoutException(f"게시글 {article_id} 본문을 찾지 못함")


def merge_history(crawled: list, history: np.ndarray) -> list:
    """
    카페에서 얻은 행 + 카페에 글이 없는 기존 행(733~834회 공홈 백필 등)을 회차 순으로 합친다.
    (예전에는 카페 행만으로 CSV 를 덮어써서 카페 개설 전 회차가 매번 지워졌다)
    같은 회차 안에서는 원래 순서(리허설 → 본게임)를 유지한다.
    """
    crawled_rounds = {int(r[1]) for r in crawled}
    kept = [row for row in history if int(row[1]) not in crawled_rounds]
    merged = list(crawled) + kept
    return sorted(merged, key=lambda r: int(r[1]))   # sorted 는 안정 정렬


def crawl_cafe(driver, history):
    history_numbers = history[:, 1].astype(int) if len(history) else np.array([], dtype=int)

    # ── 게시글 목록 수집 ──
    text_list = list_articles(driver)

    # ── 각 게시글 상세 크롤링 ──
    result_history = []
    failed: list[int] = []

    for name_tag, tap in text_list[::-1]:
        # 이미 수집된 회차는 건너뜀
        if len(history[history_numbers == int(tap)]) == 2:
            for row in history[history_numbers == int(tap)]:
                result_history.append(row)
            continue

        try:
            container = open_article(driver, name_tag)
        except TimeoutException as e:
            print(f"[{tap}회차] {e} → 건너뜀")
            failed.append(int(tap))
            continue

        content = container.text
        content2 = np.array(content.split("\n"))
        content3 = np.array([c.replace(" ", "") for c in content2])

        try:
            try:
                name_ballset = np.argwhere(content3 == "1.볼세트")[0][0]
                ballset = content3[name_ballset + 1]
            except Exception:
                ballset = int(content2[0][-1])

            name_rehearsal = np.argwhere(content3 == "2.모의추첨번호(리허설)")[0][0]
            name_winning   = np.argwhere(content3 == "3.당첨번호(추첨순)")[0][0]
            name_nextwin   = np.argwhere(content3 == "4.당첨번호(오름차순)")[0][0]
        except (IndexError, ValueError):
            print(f"[{tap}회차] 본문 형식이 예상과 다름 → 건너뜀\n  본문 앞부분: {content[:200]!r}")
            failed.append(int(tap))
            continue

        # 리허설 번호 파싱
        gap_r = name_winning - name_rehearsal
        if gap_r == 2:
            numbers_rehearsal = content2[name_rehearsal + 1].split()
        elif gap_r == 8:
            numbers_rehearsal = list(content3[name_rehearsal + 1: name_rehearsal + 8])
        else:
            numbers_rehearsal = None

        # 당첨 번호 파싱
        gap_w = name_nextwin - name_winning
        if gap_w == 2:
            numbers_winning = content2[name_winning + 1].split()
        elif gap_w in (3, 4):
            numbers_winning = [n for c in content2[name_winning + 1: name_nextwin] for n in c.split() if n]
        elif gap_w == 8:
            numbers_winning = list(content3[name_winning + 1: name_winning + 8])
        else:
            numbers_winning = None

        # 숫자 토큰만 남기고(예: "+" 구분자 제거) 본번호 6 + 보너스 1 = 7개인지 확인 — 폭이 다른 행이
        # 섞이면 main() 의 np.array/savetxt 가 실패해 그 실행의 수집분 전체를 잃는다
        def _clean(nums):
            return None if nums is None else [n for n in nums if n.isdigit() and 1 <= int(n) <= 45]
        numbers_rehearsal, numbers_winning = _clean(numbers_rehearsal), _clean(numbers_winning)
        if not numbers_rehearsal or not numbers_winning or len(numbers_rehearsal) != 7 or len(numbers_winning) != 7:
            print(f"[{tap}회차] 번호 파싱 오류 (gap 리허설={gap_r}, 당첨={gap_w}, "
                  f"리허설={numbers_rehearsal}, 당첨={numbers_winning}) → 건너뜀")
            failed.append(int(tap))
            continue
        numbers_rehearsal, numbers_winning = np.array(numbers_rehearsal), np.array(numbers_winning)

        draw_date_str = round_to_date(int(tap))
        blank_details = np.array([""] * len(DETAIL_COLUMNS))
        official_details = _fetch_official_details(int(tap))
        old_rows = history[history_numbers == int(tap)]
        if not any(str(v) for v in official_details) and len(old_rows):
            # 공홈 조회 실패 시 기존(본게임) 행의 등수별 상세를 유지 — 재수집이 상세를 지우지 않도록
            official_details = old_rows[-1][10:]

        res_rehearsal = np.concatenate(
            [np.array([ballset, tap, draw_date_str]), numbers_rehearsal, blank_details]
        )
        res_winning = np.concatenate(
            [np.array([ballset, tap, draw_date_str]), numbers_winning, official_details]
        )

        result_history.append(res_rehearsal)
        result_history.append(res_winning)

        print(f"{tap}회차")
        print(res_rehearsal)
        print(res_winning)
        print()

    if failed:
        print(f"[경고] 본문을 못 읽은 회차: {sorted(failed)} — 다음 실행 때 다시 시도됨")
    return merge_history(result_history, history)


# ─────────────────────────────────────────────
# 엔트리포인트
# ─────────────────────────────────────────────
def main():
    # ── 구버전 CSV 마이그레이션 (draw_date 없으면 추가, 등수별 상세 없으면 추가) ──
    migrate_csv_add_date(HISTORY_PATH)
    migrate_csv_add_prize_details(HISTORY_PATH)

    # ── 갱신 필요 여부 사전 체크 (브라우저 없이) ─────────────────────────────
    if not check_needs_update(HISTORY_PATH):
        print("최신 데이터 보유 중. 크롤링을 건너뜁니다.")
        print("done")
        return

    # ── 기존 히스토리 로드 ────────────────────────────────────────────────────
    if os.path.exists(HISTORY_PATH):
        # draw_date(문자열) 컬럼이 있으므로 dtype=str로 로드
        raw = np.loadtxt(HISTORY_PATH, delimiter=",", dtype=str)
    else:
        print(f"[경고] 히스토리 파일 없음: {HISTORY_PATH}")
        raw = np.array([]).reshape(0, 10 + len(DETAIL_COLUMNS))

    # crawl_cafe 에 전체 컬럼(10 + 등수별 상세)을 넘겨야 기존 행 재사용 시 shape 일치
    history_for_crawl = raw

    driver = create_driver()

    try:
        if not naver_login(driver):
            print("로그인 실패. 프로그램을 종료합니다.")
            return

        result_history = crawl_cafe(driver, history_for_crawl)

    finally:
        driver.quit()

    if result_history:
        result_array = np.array(result_history)
        np.savetxt(HISTORY_PATH, result_array, delimiter=",", fmt="%s")
        print(f"\n저장 완료: {HISTORY_PATH}")

        # ── Supabase 동기화 ───────────────────────────────────────────
        print("\n[동기화] Supabase 업로드 중...")
        try:
            import subprocess
            sync_script = os.path.join(os.path.dirname(__file__), "../scripts/sync_cafe_history.py")
            result = subprocess.run(
                [sys.executable, sync_script],
                check=True, capture_output=True, text=True
            )
            print(result.stdout.strip())
        except Exception as e:
            print(f"[동기화] 실패 — 수동으로 실행하세요: python scripts/sync_cafe_history.py\n  오류: {e}")
    else:
        print("수집된 데이터 없음.")

    print("done")


if __name__ == "__main__":
    main()
