from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import numpy as np

AllClassFlag = True
# chrome headless mode
chrome_options = Options()
# chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")

# open chrome driver
driver = webdriver.Chrome(options=chrome_options)

page_url = "https://cafe.naver.com/dhlottery?iframe_url=/ArticleList.nhn%3Fsearch.clubid=29572332%26search.menuid=22%26search.boardtype=L%26search.totalCount=151%26search.cafeId=29572332%26search.page=" 
page_idx = 1

while True:
    cur_url = page_url + str(page_idx)
    driver.get(cur_url)

    iframe = driver.find_element(By.ID, 'cafe_main')
    driver.switch_to.frame(iframe)
    content = driver.find_element(By.TAG_NAME, 'body').text

    print(content)
    if "등록된 게시글이 없습니다." in content:
        break

    page_idx+=1

print('Max Page: ', page_idx)
exit()

# start_url = "https://cafe.naver.com/dhlottery?iframe_url=/ArticleList.nhn%3Fsearch.clubid=29572332%26search.menuid=22%26search.boardtype=L"
#              https://cafe.naver.com/dhlottery?iframe_url=/ArticleList.nhn%3Fsearch.clubid=29572332%26search.menuid=22%26search.boardtype=L
# driver.get(start_url)

# iframe = driver.find_element(By.ID, 'cafe_main')
# driver.switch_to.frame(iframe)

# content = driver.find_element(By.TAG_NAME, 'body').text
# # print(content)

# content2 = np.array(content.split("\n"))

# list_idx = np.argwhere(content2=='게시물 목록')
# if len(list_idx) == 2:
#     idx_notice = list_idx[0][0]
#     idx_content = list_idx[1][0]
# elif len(list_idx) == 1:
#     idx_content = list_idx[0][0]
# else:
#     print('게시물 목록이 한 개 혹은 두 개가 나오지 않는 경우가 있네요? 확인하세요')
#     exit()

# content3 = content2[idx_content+1:]

# list_idx = np.argwhere(content3=='페이지 네비게이션')
# idx_page = list_idx[0][0]

# content4 = content3[idx_page+1:][0].reshape(4, -1)
# content3 = content3[:idx_page].split(" ")

# print('Check')


# driver.quit()
# # import pandas as pd
# # df = pd.DataFrame(total_data)
# # df.to_excel("./naver_jlpt_words_allclass.xlsx", index=False)

# print("done")

