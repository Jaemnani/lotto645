import numpy as np
import requests
from bs4 import BeautifulSoup
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data

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
    
class LHistory(data.Dataset):
    def __init__(self, input_datas, win_size=1):
        self.win_size = win_size
        self.datas = input_datas
        if win_size == 1:
            self.x_onehot = input_datas[:-1]
            self.y_onehot = input_datas[1:]
        elif win_size >= 2:
            margin = np.zeros((win_size-1, 45))
            self.x_onehot = np.vstack((margin, input_datas[:-1]))
            self.y_onehot = input_datas[1:]
        else:
            assert "Error"
    def __getitem__(self, index):
        if self.win_size == 1:
            data = self.x_onehot[index]
            data = np.expand_dims(data, 0)
        elif self.win_size >= 2:
            data = self.x_onehot[index: index+self.win_size]
        target = self.y_onehot[index]
        
        return data, target
    def __len__(self):
        return len(self.datas)-1
    
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=45):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.activate(out)
        return out

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

    input_size = 45
    hidden_size = 128
    num_layers = 2 # mininum
    output_size = 45

    model = LSTMNet(input_size, hidden_size, num_layers, output_size)

    win_size = 32
    train_dst = LHistory(train_onehots, win_size)
    train_loader = data.DataLoader(
        train_dst, 
        batch_size = 32, 
        # shuffle = True,
        shuffle = False,
        # num_workers=1,
        drop_last=False
    )

    criterion = nn.CrossEntropyLoss() # softmax
    # criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr= 0.001)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    cur_iters = 0
    cur_epochs = 0

    curve_1st = []
    curve_mean = []
    num_epochs = 10000
    for epoch in range(num_epochs):
        model.train()
        for (xs, ys) in train_loader:
            cur_iters += 1
            xs = xs.to(device, dtype=torch.float32)
            ys = ys.to(device, dtype=torch.float32)
        
            optimizer.zero_grad()
            outputs = model(xs)
            loss = criterion(outputs, ys)
            loss.backward()
            optimizer.step()
        
        # valid
        model.eval()
        val_loss = 0
        with torch.no_grad():
            from collections import Counter
            def count_matches(row1, row2):
                common_counts = Counter(row1) & Counter(row2)
                # if len(common_counts) == 0:
                #     return 0
                # else:
                return common_counts
            x_valid = np.expand_dims(valid_onehots[-win_size-1:-1], 0)
            x_valid = torch.from_numpy(x_valid).to(device, dtype=torch.float32)
            y_correct = torch.from_numpy( np.array([valid_onehots[-1]]) ).to(device, dtype=torch.float32)
            y_valid = valid_onehots[-1].argsort()[-6:] + 1

            pred = model(x_valid)
            # res = np.zeros(45)
            # res[pred[0].argsort()[-6:]] = 1.
            pred[0, pred[0].argsort()[-6:]] = 1.
            pred[0, pred[0].argsort()[:-6]] = 0.
            
            val_loss = criterion(pred, y_correct)
            p_valid = (np.sort(pred.argsort()[:, -6:]) + 1).flatten()
            
            res = count_matches(y_valid, p_valid)
            
        last_input = torch.from_numpy(np.expand_dims(valid_onehots[-win_size:],0)).to(device, dtype=torch.float32)
        last_pred = model(last_input)
        last_pred = last_pred[0].argsort()[-6:] + 1
        if sum(res.values()) >= 3:
            print(f'Epoch [{epoch+1}/{num_epochs} || iter:{cur_iters}], Loss: {loss.item():.4f}, val_Loss: {val_loss.item():.4f} /', res, np.sort(last_pred) )
        
        # scheduler.step()
            # print("val lossres , "Check Count : ", sum(res.values()))

    print("Train done.")
    
if __name__ == "__main__":
    main()