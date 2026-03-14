from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import numpy as np
import time
import os

history_path = "./history_from_cafe.csv"
is_exists_history_file = os.path.exists(history_path)
if is_exists_history_file:
    history = np.loadtxt(history_path, delimiter=',').astype(int)

AllClassFlag = True
# chrome headless mode
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

# open chrome driver
driver = webdriver.Chrome(options=chrome_options)

#네이버 로그인하기
naver_id = "dpwoans"
naver_pw = "Woans8643!"
login_url = "https://nid.naver.com/nidlogin.login"
driver.get(login_url)

# id_input = driver.find_element(By.ID, "id")
# id_input.clear()
# id_input.send_keys(naver_id)

# pw_input = driver.find_element(By.ID, "pw")
# pw_input.clear()
# pw_input.send_keys(naver_pw)

# login_button = driver.find_element(By.ID, "log.login")

checkbox = driver.find_element(By.ID, 'keep')
driver.execute_script("arguments[0].click();", checkbox)


# login_button.click()


# 수동으로 로그인을 해야 할듯...
time.sleep(1)


page_url = "https://cafe.naver.com/dhlottery?iframe_url=/ArticleList.nhn%3Fsearch.clubid=29572332%26search.menuid=22%26search.boardtype=L%26search.totalCount=151%26search.cafeId=29572332%26search.page=" 
page_idx = 1

text_list = np.array([]).reshape(-1, 2)
while True:
    cur_url = page_url + str(page_idx)
    driver.get(cur_url)
    iframe = driver.find_element(By.ID, 'cafe_main')
    driver.switch_to.frame(iframe)
    content = driver.find_element(By.TAG_NAME, 'body').text
    if "등록된 게시글이 없습니다." in content:
        break
    content2 = np.array(content.split("\n"))
    list_idx = np.argwhere(content2=='게시물 목록')
    if len(list_idx) == 2:
        idx_notice = list_idx[0][0]
        idx_content = list_idx[1][0]
    elif len(list_idx) == 1:
        idx_content = list_idx[0][0]
    else:
        print('게시물 목록이 한 개 혹은 두 개가 나오지 않는 경우가 있네요? 확인하세요')
        exit()
    content3 = content2[idx_content+1:]
    # # 로그인이 안 되었을 시,
    # list_idx = np.argwhere(content3=='페이지 네비게이션')
    # # 로그인이 되었을 시,
    list_idx = np.argwhere(content3=='글쓰기')
    idx_page = list_idx[0][0]
    infos = content3[:idx_page].reshape(-1 , 4)
    infos = infos[:, :2]
    infos[:, 1] = [info.split('로또6/45 제')[1].split('회')[0] for info in infos[:, 1]]
    text_list = np.vstack((text_list, infos))
    page_idx+=1
print('Max Page: ', page_idx)

#

history_numbers = history[:, 1]

# 로그인 필요함.
result_history = []
ball_dirs = []
machine = []
for name_tag, tap in text_list[::-1]:
    
    if len(history[history_numbers == int(tap)]) == 2:
        for i in history[history_numbers == int(tap)]:
            result_history.append(i)
        continue
    
    tap_url = "https://cafe.naver.com/dhlottery?iframe_url_utf8=%2FArticleRead.nhn%253Fclubid%3D29572332%2526page%3D1%2526menuid%3D22%2526boardtype%3DL%2526articleid%3D"+name_tag+"%2526referrerAllArticles%3Dfalse"
    driver.get(tap_url)
    # iframe = driver.find_element(By.ID, 'cafe_main')
    iframe = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, 'cafe_main'))
    )
    driver.switch_to.frame(iframe)
    
    try:
        container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "se-main-container")))
    except:
        container = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CLASS_NAME, "ContentRenderer")))
    
    content = container.text
    
    # print(content)
    content2 = np.array(content.split("\n"))
    
    content3 = np.array([cont.replace(" ", "") for cont in content2])
    
    try:
        name_ballset = np.argwhere(content3 == "1.볼세트")[0][0]
        ballset = content3[name_ballset+1]
    except:
        ballset = int(content2[0][-1])
    

    name_rehearsal = np.argwhere(content3=="2.모의추첨번호(리허설)")[0][0]    
    name_winning = np.argwhere(content3=="3.당첨번호(추첨순)")[0][0]
    name_nextwin = np.argwhere(content3=="4.당첨번호(오름차순)")[0][0]
    
    # 922회부터 내용물이 달라진다. 왜일까?
    if( name_winning - name_rehearsal == 2):
        numbers_rehearsal = np.array(content2[name_rehearsal+1].split(' '))
    elif( name_winning - name_rehearsal == 8):
        numbers_rehearsal = content3[name_rehearsal+1 : name_rehearsal + 1 + 7]
    else:
        print("Error:", name_nextwin - name_winning)
    
    if ( name_nextwin - name_winning == 2):
        numbers_winning = np.array(content2[name_winning+1].split(' '))
    elif (name_nextwin - name_winning == 3):
        numbers_winning = np.array(sum([cont.split(" ") for cont in content2[name_winning+1: name_nextwin]], []))
    elif (name_nextwin - name_winning == 4):
        numbers_winning = np.array(sum([cont.split(" ") for cont in content2[name_winning+1: name_nextwin]], []))
    elif (name_nextwin - name_winning == 8):
        numbers_winning = content3[name_winning+1 : name_winning + 1 + 7]
    else:
        print("Error:", name_nextwin - name_winning)
    
    res_rehearsal = np.append(np.array([ballset, tap]), numbers_rehearsal)
    res_winning = np.append(np.array([ballset, tap]), numbers_winning)
    
    result_history.append(res_rehearsal)
    result_history.append(res_winning)
    
    # print("")
    print(tap, "회차")
    print(res_rehearsal)
    print(res_winning)
    print("")
    
    # for cont in content3:
    #     if "볼배열방식" in cont:
    #         ball_dir = cont.split(":")[-1]
    #         ball_dirs.append(ball_dir)
    #         print("볼배열:", ball_dir)
    #     if "추첨기" in cont:
    #         num_machine = cont.split(":")[-1][0]
    #         machine.append(num_machine)
    #         print("기계번호:", num_machine)
    
    
            
driver.quit()

result_array = np.array(result_history)
np.savetxt(history_path, result_array, delimiter=",", fmt="%s")
print("done")


# 862회차 확인하기.
