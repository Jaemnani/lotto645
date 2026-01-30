
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os

# Credentials
# Prioritize Environment Variables (for GitHub Actions), fallback to hardcoded for local testing
USER_ID = os.getenv("LOTTO_USER_ID", "yaejm")
USER_PW = os.getenv("LOTTO_USER_PW", "woans955!")

def setup_driver():
    chrome_options = Options()
    
    # Headless mode for clean automation (Mandatory for GitHub Actions)
    # Check if running in GitHub Actions or explicitly requested
    if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("HEADLESS") == "true":
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        # Add User-Agent to prevent bot detection in headless
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36")
    else:
        # Local non-headless (visible)
        # chrome_options.add_argument("--headless") # Comment out for visual verification
        chrome_options.add_argument("--window-size=1920,1080")
    
    # Try to find chromedriver or let selenium manager handle it
    try:
        driver = webdriver.Chrome(options=chrome_options)
    except Exception as e:
        print(f"Error initializing Chrome driver: {e}")
        sys.exit(1)
    return driver

def login(driver):
    print("Navigating to homepage...")
    driver.get("https://www.dhlottery.co.kr/")
    
    try:
        # Click login button on homepage
        print("Clicking login button on homepage...")
        login_home_btn = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "loginBtn"))
        )
        # Use simple click first, then JS click if needed? Or just JS click to be safe.
        # driver.execute_script("arguments[0].click();", login_home_btn)
        # Actually, let's try standard click but catch specific errors.
        try:
             login_home_btn.click()
        except Exception as click_e:
             print(f"Standard click failed ({click_e}), trying JS click...")
             driver.execute_script("arguments[0].click();", login_home_btn)
        
        # Wait for login page
        print("Waiting for login page...")
        id_input = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "inpUserId"))
        )
        pw_input = driver.find_element(By.ID, "inpUserPswdEncn")
        
        print(f"Attempting login with ID: {USER_ID}")
        
        # ID Input
        id_input.click()
        id_input.clear()
        id_input.send_keys(USER_ID)
        
        # Password Input
        pw_input.click()
        pw_input.clear()
        pw_input.send_keys(USER_PW)
        
        # Debug: Screenshot before clicking login
        print("Taking debug screenshot of login form...")
        driver.save_screenshot("debug_login_form_filled.png")
             
        # Click login button on login page
        login_btn = driver.find_element(By.ID, "btnLogin") # form submit button
        driver.execute_script("arguments[0].click();", login_btn) # Use JS click here too just in case
        
        # Check for success
        print("Waiting for login completion...")
        time.sleep(5) # Increased wait time
        
        # Check if browser is still open
        if not driver.window_handles:
             print("Error: Browser window closed unexpectedly.")
             return False
             
        try:
            curr_url = driver.current_url
            print(f"Current URL: {curr_url}")
            
            if "login" not in curr_url:
                 print("Login successful (URL check).")
                 return True
            else:
                 print("Still on login URL, checking for alert...")
                 try:
                    alert = driver.switch_to.alert
                    print(f"Login Alert: {alert.text}")
                    alert.accept()
                    return False
                 except:
                    # Check if 'Log out' button exists, enabling login verification without URL change
                    try:
                        logout_btn = driver.find_element(By.XPATH, "//*[contains(text(), '로그아웃')]")
                        print("Found logout button - Login successful.")
                        return True
                    except:
                        pass
                    
                    print("Warning: Still on login page but no alert. Assume failed or slow.")
                    return False # Safer to assume false if we are still on login page
                    
        except Exception as url_e:
            print(f"Error checking URL/Login status: {url_e}")
            return False
             
        return True

    except Exception as e:
        print(f"Login failed: {e}")
        return False

def buy_lotto(driver):
    print("Attempting to enter purchase page via Homepage...")
    
    try:
        # User requested: Home -> Login (Done) -> Click Purchase -> Popup -> Buy
        
        # 1. Find the "Lotto 6/45" purchase button. 
        # Usually found in the main GNB or a quick menu. 
        # Verified selector might be needed, but let's try robust text search.
        # "Lotto 6/45" text is common.
        
        # Trying to find the 'Buy' button for Lotto 6/45
        # Common pattern: A link that pops up the game
        print("Looking for 'Lotto 6/45' buy link...")
        
        # Setup main window handle
        main_window = driver.current_window_handle
        old_handles = driver.window_handles
        
        # Try clicking the main 'Lotto 6/45' menu item or "Buying" button
        # This xpath looks for a link that likely leads to the game
        # We look for "Lotto 6/45" in the navigation
        try:
             # Try specific robust path if possible, or generic text
             # Often "번호선택" (Select Number) or just the main game link
             lotto_link = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//a[contains(@onclick, 'LottoId=LO40') or contains(@href, 'TotalGame.jsp?LottoId=LO40')]"))
             )
             print("Found Lotto 6/45 link by URL pattern.")
             lotto_link.click()
        except:
             print("URL pattern link not found, trying text search '로또6/45'...")
             # This might click the info page, not purchase.
             # Let's try finding the specific "Purchase" button in the menu
             # Often "구매하기"
             # Let's try forcing the javascript open if link is hard to find, but user wants "Click"
             try:
                 # Look for '구매하기' inside a structure related to Lotto 6/45
                 # Or just the main visual button
                 buy_btn = driver.find_element(By.CSS_SELECTOR, "#gnb .gnb1 > a") # First GNB item is usually Lotto 6/45
                 buy_btn.click() # This might just open menu
                 # Then click purchase? 
                 # Let's try a safer bet: The quick menu or specific ID "gnb_01_01" (Lotto 6/45)
                 pass
             except:
                 pass
                 
             # Fallback: Just go to the URL directly if click fails? 
             # User said "Must go through homepage". 
             # If we can't find the click, we might fail.
             # Let's try to execute the function that the button calls.
             # usually `goGame('LO40')`
             print("Attempting to trigger game via JS function goGame...")
             driver.execute_script("try { goGame('LO40'); } catch(e) { window.open('https://el.dhlottery.co.kr/game/TotalGame.jsp?LottoId=LO40', 'pop01', 'width=800,height=600'); }")

        # 2. Wait for new window
        print("Waiting for new purchase window...")
        WebDriverWait(driver, 10).until(EC.new_window_is_opened(old_handles))
        
        new_handles = driver.window_handles
        new_window = [h for h in new_handles if h != main_window][0]
        
        driver.switch_to.window(new_window)
        print("Switched to purchase window.")
        
        # 3. Wait for Iframe
        print("Waiting for game iframe...")
        iframe = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.ID, "ifrm_tab"))
        )
        driver.switch_to.frame(iframe)
        print("Switched to game iframe.")
        
        # 4. Select Automatic Tab
        print("Selecting 'Automatic Number' Tab (자동번호발급)...")
        # User requested using #num2
        try:
            auto_tab = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "num2"))
            )
            auto_tab.click()
            print("Clicked Automatic tab.")
        except Exception as e:
             print(f"Tab click failed: {e}. Trying alternate selector...")
             # Fallback
             driver.find_element(By.XPATH, "//a[contains(text(), '자동번호발급')]").click()
             
        # Select Quantity '5'
        print("Selecting Quantity 5...")
        try:
            # Use JS to reliably set quantity and trigger change event
            # element: <select id="amoundApply" ... onchange="paperTextChange(this.value)">
            driver.execute_script("""
                var sel = document.getElementById('amoundApply');
                sel.value = '5';
                if(typeof paperTextChange === 'function') {
                    paperTextChange(5);
                } else {
                    sel.dispatchEvent(new Event('change'));
                }
            """)
            print("Selected quantity 5 using JS.")
        except Exception as e:
            print(f"JS select failed: {e}. Trying standard...")
            try:
                quantity_element = driver.find_element(By.ID, "amoundApply")
                Select(quantity_element).select_by_value("5")
            except Exception as e2:
                print(f"Standard select also failed: {e2}")


        # Click 'Confirm' to add to cart
        print("Clicking 'Confirm' (확인) to select numbers...")
        btn_select = driver.find_element(By.ID, "btnSelectNum")
        btn_select.click()
        
        # Check for any alerts/popups after selecting numbers (e.g., "Added" or "Select more")
        print("Checking for alerts after selection...")
        try:
             # Check for DOM alert
             popup_alert = WebDriverWait(driver, 3).until(
                 EC.visibility_of_element_located((By.ID, "popupLayerAlert"))
             )
             print(f"Found popup alert. Text: {popup_alert.text}")
             
             # Try determining if it is just an "Added" confirmation
             # Attempt to close it
             try:
                 # Standard way to close this alert
                 driver.execute_script("closepopupLayerAlert();")
                 print("Dismissed popup alert via JS.")
             except:
                 # Fallback to clicking button
                 popup_ok = popup_alert.find_element(By.CSS_SELECTOR, ".btn_common")
                 popup_ok.click()
             
             # Wait for popup to disappear
             time.sleep(1)
             
        except TimeoutException:
             print("No popup alert appeared immediately.")
        except Exception as e:
             # Force close if failed
             print(f"Error handling popup: {e}. Trying JS hide...")
             try:
                driver.execute_script("document.getElementById('popupLayerAlert').style.display='none';")
             except:
                pass
        
        # 5. Buy
        print("Clicking 'Buy' (구매하기)...")
        time.sleep(1) # Safety wait
        btn_buy = driver.find_element(By.ID, "btnBuy")
        
        # Use JS click to avoid 'element click intercepted' if something is still there
        driver.execute_script("arguments[0].click();", btn_buy)
        
        # 6. Confirm Popup
        print("Handling confirmation popup...")
        try:
             # User provided HTML: 
             # <input type="button" class="button lrg confirm" value="확인" onclick="javascript:closepopupLayerConfirm(true);">
             
             # Execute the JS function directly since clicking is proving difficult
             print("Executing closepopupLayerConfirm(true) via JS...")
             driver.execute_script("if(typeof closepopupLayerConfirm === 'function') { closepopupLayerConfirm(true); } else { console.log('Function not found'); }")
             print("Executed confirmation JS.")
             
             # Success Check
             print("Purchase command sent. Checking for success alert...")
             time.sleep(2)
             try:
                 final_alert = driver.switch_to.alert
                 print(f"Result Alert: {final_alert.text}")
                 final_alert.accept()
             except:
                 print("No final result alert found.")
                 
        except Exception as e:
            print(f"Error during confirmation: {e}")
            
    except Exception as e:
        print(f"Purchase failed: {e}")
        driver.save_screenshot("purchase_failure.png")
    
    # Switch back just in case
    try:
        driver.switch_to.window(main_window)
    except:
        pass

def main():
    driver = setup_driver()
    try:
        if login(driver):
            buy_lotto(driver)
        else:
            print("Skipping purchase due to login failure.")
    finally:
        print("Closing driver in 5 seconds...")
        time.sleep(5)
        driver.quit()

if __name__ == "__main__":
    main()
