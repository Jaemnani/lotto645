from fastapi import FastAPI
from pydantic import BaseModel

import numpy as np
import torch
import requests
from bs4 import BeautifulSoup
from m01_model import LSTMNet
import os

app = FastAPI()

class LottoInput(BaseModel):
    history: list[list[int]]  # one-hot 형태


# device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
device = 'cpu'

# torch.serialization.add_safe_globals({'LSTMNet': LSTMNet})

win_size = 64
model = LSTMNet(input_size=45, hidden_size=128, num_layers=2)
model.load_state_dict(torch.load("m01_model.pt", map_location=device))
model.eval()

@app.post("/predict")
def predict(input: LottoInput):
    x = torch.tensor([input.history], dtype=torch.float32)
    with torch.no_grad():
        pred = model(x)
        pred = sorted((pred[0].argsort()[-6:] + 1).numpy())
        return {"numbers": pred}

def main():
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
    rounds = all_info[:, 0]
    numbers_all = all_info[:, 1:]
    numbers_sorted = np.sort(numbers_all, axis=1)

    numbers_onehot = np.zeros((len(numbers_sorted), 45))
    for i, numbers in enumerate(numbers_sorted):
        for j, number in enumerate(numbers):
            numbers_onehot[i][number-1] = 1
            
    train_onehots = numbers_onehot[:-1]
    valid_onehots = numbers_onehot
    x_valid = np.expand_dims(valid_onehots[-win_size-1:-1], 0)
    x_valid = torch.from_numpy(x_valid).to(device, dtype=torch.float32)
    y_correct = torch.from_numpy( np.array([valid_onehots[-1]]) ).to(device, dtype=torch.float32)
    y_valid = valid_onehots[-1].argsort()[-6:] + 1
    last_input = torch.from_numpy(np.expand_dims(valid_onehots[-win_size:],0)).to(device, dtype=torch.float32)

    last_pred = model(last_input)
    last_pred = sorted((last_pred[0].argsort()[-6:] + 1).numpy())
    print(last_pred)
    print("Predict done.")


if __name__ == "__main__":
    main()