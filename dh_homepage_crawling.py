import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from tqdm import tqdm

def get_info(start, end, 
             basic_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin&drwNo=" # 임의의 회차를 얻기 위한 주소
             ):
        all_info = []
        for i, round in enumerate(range(start, end+1)):
            target_url = basic_url + str(round)
            resp = requests.get(target_url)
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.text
            
            s_idx = text.find(" 당첨결과")
            s_idx = text.find("당첨번호", s_idx) + 4
            e_idx = text.find("보너스", s_idx)
            numbers = text[s_idx:e_idx].strip().split()
            
            s_idx = e_idx + 3
            e_idx = s_idx + 3
            bonus = text[s_idx:e_idx].strip()
            
            round_info = np.append(np.append(round, numbers), bonus).astype(int)
            
            print('4-%d, %d info'%(i, round), round_info)
            all_info.append(round_info)
        all_info = np.array(all_info)
        return all_info

def main(): 
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    device = 'cpu'
    history_path = "./history.csv"
    main_url = "https://www.dhlottery.co.kr/gameResult.do?method=byWin" # 마지막 회차를 얻기 위한 주소

    start_round = 733 # Change machine
    resp = requests.get(main_url)
    soup = BeautifulSoup(resp.text, "lxml")
    result = str(soup.find("meta", {"id" : "desc", "name" : "description"})['content'])
    s_idx = result.find(" ")
    e_idx = result.find("회")
    last_round = int(result[s_idx + 1 : e_idx])

    if os.path.exists(history_path):
        all_info = np.loadtxt(history_path, delimiter=",").astype(int)
        last_info = all_info[-1][0]
        if last_round != last_info:
            if last_info < last_round:
                new_info = get_info(last_info+1, last_round)
                all_info = np.concatenate((all_info, new_info))
                np.savetxt(history_path, all_info, delimiter=',', fmt="%d")
            else:
                print("Error: saved info is something wrong.")
                exit()
        else:
            pass
    else:
        all_info = get_info(start_round, last_round)
        np.savetxt(history_path, all_info, delimiter=',', fmt="%d")
    print("Get data done.")
    
if __name__ == "__main__":
    main()
