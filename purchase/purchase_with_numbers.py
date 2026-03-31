"""
purchase_with_numbers.py
========================
지정된 번호 세트를 동행복권 사이트에서 자동 구매합니다.

기존 auto_purchase.py 와 다른 점:
  - 자동번호(랜덤) 대신 수동번호(지정 6개)를 선택
  - tickets: list[list[int]] 형태로 최대 5장까지 지정
  - 각 장마다 6개 번호를 직접 클릭하여 구매

사용법 (단독 실행):
  LOTTO_USER_ID=xxx LOTTO_USER_PW=yyy python purchase_with_numbers.py \
    --numbers "1,14,16,17,19,21" "3,9,12,22,26,36" "3,4,12,16,26,42"
"""

import os
import sys
import time
import argparse
from datetime import datetime

import pytz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

USER_ID = os.getenv("LOTTO_USER_ID", "")
USER_PW = os.getenv("LOTTO_USER_PW", "")


# ── 구매 가능 시간 확인 ────────────────────────────────────────────────────────
def check_purchase_time() -> bool:
    kst = pytz.timezone("Asia/Seoul")
    now = datetime.now(kst)
    weekday = now.weekday()   # 0=월, 5=토, 6=일
    hour    = now.hour
    print(f"현재 KST: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    if weekday == 5:          # 토요일
        ok = 6 <= hour < 20
    else:
        ok = 6 <= hour < 24
    if not ok:
        print("구매 가능 시간이 아닙니다.")
    return ok


# ── Chrome 드라이버 초기화 ────────────────────────────────────────────────────
def setup_driver() -> webdriver.Chrome:
    opts = Options()
    if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("HEADLESS") == "true":
        opts.add_argument("--headless")
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    try:
        driver = webdriver.Chrome(options=opts)
    except Exception as e:
        print(f"ChromeDriver 초기화 실패: {e}")
        sys.exit(1)
    return driver


# ── 로그인 ────────────────────────────────────────────────────────────────────
def login(driver) -> bool:
    print("로그인 시도...")
    driver.get("https://www.dhlottery.co.kr/")
    try:
        btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "loginBtn"))
        )
        try:
            btn.click()
        except Exception:
            driver.execute_script("arguments[0].click();", btn)

        id_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "inpUserId"))
        )
        pw_input = driver.find_element(By.ID, "inpUserPswdEncn")
        id_input.clear(); id_input.send_keys(USER_ID)
        pw_input.clear(); pw_input.send_keys(USER_PW)

        login_btn = driver.find_element(By.ID, "btnLogin")
        driver.execute_script("arguments[0].click();", login_btn)
        time.sleep(5)

        curr_url = driver.current_url
        if "login" not in curr_url:
            print("로그인 성공")
            return True

        try:
            alert = driver.switch_to.alert
            print(f"로그인 알림: {alert.text}")
            alert.accept()
        except Exception:
            pass
        print("로그인 실패")
        return False

    except Exception as e:
        print(f"로그인 예외: {e}")
        return False


# ── 구매 창 열기 및 iframe 진입 ───────────────────────────────────────────────
def open_purchase_window(driver) -> tuple[str, str]:
    """(main_window, game_window) 핸들 반환. iframe은 호출 후 직접 전환."""
    main_window  = driver.current_window_handle
    old_handles  = set(driver.window_handles)

    driver.execute_script(
        "try { goGame('LO40'); } catch(e) {"
        "  window.open('https://el.dhlottery.co.kr/game/TotalGame.jsp?LottoId=LO40',"
        "  'pop01', 'width=830,height=660'); }"
    )
    WebDriverWait(driver, 10).until(EC.new_window_is_opened(old_handles))
    game_window = [h for h in driver.window_handles if h not in old_handles][0]

    driver.switch_to.window(game_window)
    print("구매 창 열림")

    try:
        iframe = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.ID, "ifrm_tab"))
        )
        driver.switch_to.frame(iframe)
        print("iframe 진입 완료")
    except TimeoutException:
        driver.save_screenshot("purchase_error.png")
        raise RuntimeError("ifrm_tab iframe을 찾지 못했습니다. purchase_error.png 확인")

    return main_window, game_window


# ── 수동 번호 선택 (1장) ──────────────────────────────────────────────────────
def select_manual_numbers(driver, numbers: list[int]) -> bool:
    """
    수동 탭에서 6개 번호를 선택하고 '확인' 버튼을 누릅니다.
    성공 시 True 반환.

    동행복권 게임 페이지 번호 버튼 구조 (일반적):
      <span id="number645_1" class="ball_645 ball_645_1">1</span>  (숫자별)
    또는 input type=checkbox / label 구조.
    여러 셀렉터를 시도하여 범용성 확보.
    """
    assert len(numbers) == 6, "번호는 반드시 6개여야 합니다."

    # ── 수동 탭 클릭 ─────────────────────────────────────────────────────────
    manual_selectors = [
        (By.ID,    "num1"),
        (By.XPATH, "//a[contains(text(),'수동번호발급') or contains(text(),'수동')]"),
        (By.XPATH, "//li[contains(@class,'manual')]//a"),
    ]
    for by, sel in manual_selectors:
        try:
            tab = WebDriverWait(driver, 5).until(EC.element_to_be_clickable((by, sel)))
            tab.click()
            print(f"  수동 탭 클릭 ({sel})")
            time.sleep(0.5)
            break
        except Exception:
            continue

    # ── 번호 클릭 ─────────────────────────────────────────────────────────────
    for num in numbers:
        clicked = False
        candidates = [
            # 패턴 1: id="number645_N"
            (By.ID,    f"number645_{num}"),
            # 패턴 2: span/label text = num
            (By.XPATH, f"//span[@class and contains(@class,'ball') and text()='{num}']"),
            (By.XPATH, f"//label[text()='{num}']"),
            # 패턴 3: data-value 속성
            (By.XPATH, f"//*[@data-value='{num}']"),
            # 패턴 4: input checkbox value=num
            (By.XPATH, f"//input[@type='checkbox' and @value='{num}']"),
        ]
        for by, sel in candidates:
            try:
                el = WebDriverWait(driver, 3).until(EC.element_to_be_clickable((by, sel)))
                driver.execute_script("arguments[0].click();", el)
                clicked = True
                print(f"    번호 {num} 선택 완료")
                time.sleep(0.2)
                break
            except Exception:
                continue

        if not clicked:
            print(f"    [경고] 번호 {num} 클릭 실패 (UI 구조 확인 필요)")

    # ── 확인 버튼 ────────────────────────────────────────────────────────────
    try:
        confirm = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.ID, "btnSelectNum"))
        )
        confirm.click()
        print("  확인 버튼 클릭")
        time.sleep(0.8)
    except Exception:
        # 다른 선택자 시도
        try:
            driver.find_element(
                By.XPATH, "//input[@type='button' and contains(@value,'확인')]"
            ).click()
        except Exception as e:
            print(f"  [경고] 확인 버튼 클릭 실패: {e}")
            return False

    # ── 팝업 닫기 ─────────────────────────────────────────────────────────────
    try:
        WebDriverWait(driver, 3).until(
            EC.visibility_of_element_located((By.ID, "popupLayerAlert"))
        )
        driver.execute_script(
            "if(typeof closepopupLayerAlert==='function') closepopupLayerAlert();"
            "else document.getElementById('popupLayerAlert').style.display='none';"
        )
        time.sleep(0.5)
    except TimeoutException:
        pass  # 팝업 없으면 OK

    return True


# ── 장바구니 구매 ─────────────────────────────────────────────────────────────
def checkout(driver) -> bool:
    """구매하기 버튼 클릭 및 최종 확인."""
    try:
        btn_buy = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.ID, "btnBuy"))
        )
        driver.execute_script("arguments[0].click();", btn_buy)
        print("구매하기 버튼 클릭")
        time.sleep(1)

        # 구매 확인 팝업
        driver.execute_script(
            "if(typeof closepopupLayerConfirm==='function') closepopupLayerConfirm(true);"
        )
        print("구매 확인 완료")
        time.sleep(2)

        # 결과 alert
        try:
            alert = driver.switch_to.alert
            print(f"결과: {alert.text}")
            alert.accept()
        except Exception:
            pass

        return True

    except Exception as e:
        print(f"구매 실패: {e}")
        driver.save_screenshot("purchase_failure.png")
        return False


# ── 메인: 번호 목록 구매 ─────────────────────────────────────────────────────
def buy_with_numbers(tickets: list[list[int]]) -> bool:
    """
    tickets: 최대 5개의 [n1, n2, n3, n4, n5, n6] 리스트.
    공세트별 최고 조합 5장을 한 번에 구매합니다.
    """
    assert 1 <= len(tickets) <= 5, "tickets는 1~5장"
    for t in tickets:
        assert len(t) == 6, "각 티켓은 6개 번호"

    if not check_purchase_time():
        return False

    if not USER_ID or not USER_PW:
        print("[오류] LOTTO_USER_ID / LOTTO_USER_PW 환경변수를 설정하세요.")
        return False

    driver = setup_driver()
    try:
        if not login(driver):
            return False

        main_window, game_window = open_purchase_window(driver)

        print(f"\n총 {len(tickets)}장 수동 번호 구매 시작")
        for i, nums in enumerate(tickets, 1):
            print(f"\n[{i}번째 티켓] 번호: {sorted(nums)}")
            ok = select_manual_numbers(driver, nums)
            if not ok:
                print(f"  [{i}번째 티켓] 번호 선택 실패 → 건너뜀")

        success = checkout(driver)

        driver.switch_to.window(main_window)
        return success

    except Exception as e:
        print(f"예외 발생: {e}")
        try:
            driver.save_screenshot("purchase_error.png")
        except Exception:
            pass
        return False

    finally:
        time.sleep(3)
        driver.quit()


# ── 단독 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--numbers", nargs="+", required=True,
        metavar="N1,N2,N3,N4,N5,N6",
        help='예: "1,14,16,17,19,21" "3,9,12,22,26,36"'
    )
    args = parser.parse_args()

    tickets = []
    for s in args.numbers:
        nums = [int(x.strip()) for x in s.split(",")]
        tickets.append(nums)

    result = buy_with_numbers(tickets)
    print("\n구매 결과:", "성공" if result else "실패")
    sys.exit(0 if result else 1)
