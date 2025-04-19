import numpy as np
# import requests
# from bs4 import BeautifulSoup
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt

reward = [100_000, 5_000, 200, 5, 0.5]
def gen_numbers_from_weighted_probability(weights, n=6):
    numbers = np.arange(1, 46)  # 1~45까지의 번호
    selected_numbers = []

    while len(selected_numbers) < n:
        chosen_number = np.random.choice(numbers, p=weights)
        if chosen_number not in selected_numbers:
            selected_numbers.append(chosen_number)
    
    selected_numbers.sort()
    return selected_numbers

def gen_numbers_from_probability(nums_prob):
    ball_box = []
    for n in range(45):
        ball_count = int(nums_prob[n] * 10000 + 1)
        ball = np.full((ball_count), n+1) #1부터 시작
        ball_box += list(ball)
    selected_balls = []
    while True:
        if len(selected_balls) == 6:
            break
        ball_index = np.random.randint(len(ball_box), size=1)[0]
        ball = ball_box[ball_index]
        if ball not in selected_balls:
            selected_balls.append(ball)
    selected_balls.sort()
    return selected_balls

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
    def __init__(self, input_size, hidden_size, num_layers, device, output_size=45):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        # self.activate = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        # out = self.activate(out)
        return out

class DeepLSTMNet(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size=45):
                super(DeepLSTMNet, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc1 = nn.Linear(hidden_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc_out = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()

            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                out, _ = self.lstm(x, (h0, c0))
                out = self.relu(self.fc1(out[:, -1, :]))
                out = self.relu(self.fc2(out))
                out = self.fc_out(out)
                return out
        
class MLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x).to(x.device)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)  # 확률값으로 변환
        return out

class DeepMLPNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepMLPNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc5(x))
        x = self.sigmoid(self.fc_out(x))
        return x

def main():     
    history_path = "./history_from_cafe.csv"
    is_exists_history_file = os.path.exists(history_path)
    if is_exists_history_file:
        history = np.loadtxt(history_path, delimiter=',').astype(int)
    else:
        print("데이터를 세팅해주세요.")
        return -1
    
    result_numbers = []
    ballset_list = [1,2,3,4,5]
    for bn in ballset_list:
        print("ballset number = ", bn)
        hist = history[history[:,0] == bn][:, 1:]    
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        # device = 'cpu'
        
        rounds = hist[:, 0]
        numbers_all = hist[:, 1:-1]
        numbers_sorted = np.sort(numbers_all, axis=1)

        numbers_onehot = np.zeros((len(numbers_sorted), 45))
        for i, numbers in enumerate(numbers_sorted):
            for j, number in enumerate(numbers):
                numbers_onehot[i][number-1] = 1
                
        gap = 10
        train_onehots = numbers_onehot[: -gap-1]
        valid_onehots = numbers_onehot[-gap-1:-1]
        test_onehots = numbers_onehot[-1]

        print("Train_valid_test Split Shape")
        print(train_onehots.shape)
        print(valid_onehots.shape)
        print(test_onehots.shape)


        # MLP
        model = DeepMLPNet(input_size=45, hidden_size=256, output_size=45).to(device)
        criterion = nn.BCELoss()

        learning_rate = 0.01
        weight_decay = 0.0001
        # optimizer = optim.Adam(model.parameters(), lr=0.001)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = weight_decay)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        win_size = 1
        train_dst = LHistory(train_onehots, win_size)
        train_loader = data.DataLoader(
            train_dst, 
            batch_size = 1, 
            # shuffle = True,
            shuffle = False,
            # num_workers=1,
            drop_last=False
        )

        # # LSTM
        # input_size = 45
        # hidden_size = 128
        # num_layers = 4 # mininum
        # output_size = 45

        # model = LSTMNet(input_size, hidden_size, num_layers, output_size).to(device)

        # win_size = 8
        # train_dst = LHistory(train_onehots, win_size)
        # train_loader = data.DataLoader(
        #     train_dst, 
        #     batch_size = 8, 
        #     # shuffle = True,
        #     shuffle = False,
        #     # num_workers=1,
        #     drop_last=False
        # )

        # criterion = nn.CrossEntropyLoss() # softmax
        # # criterion = nn.BCEWithLogitsLoss()
        # optimizer = optim.Adam(model.parameters(), lr= 0.001)
        # # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # # optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


        

        cur_iters = 0
        cur_epochs = 0

        curve_1st = []
        curve_mean = []

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_val_accuracy = 0
        lowest_val_loss = 99
        best_epoch = 0

        num_epochs = 500
        for epoch in range(num_epochs):
            final_numbers = []

            model.train()

            epoch_train_loss = 0
            epoch_train_correct = 0
            total_train_samples = 0

            for (xs, ys) in train_loader:
                cur_iters += 1
                xs = xs.to(device, dtype=torch.float32)
                #MLP 
                xs = xs[0]
                ys = ys.to(device, dtype=torch.float32)
            
                optimizer.zero_grad()
                outputs = model(xs)
                outputs = F.softmax(outputs, dim=1)

                loss = criterion(outputs, ys)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

                # correct calculation
                pred_argsort6 = outputs.argsort(axis=1)[:, -6:]
                len_pred_argsort = len(pred_argsort6)
                preds_onehot = np.zeros((len_pred_argsort, 45))
                for i, numbers in enumerate(pred_argsort6):
                    for j, number in enumerate(numbers):
                        preds_onehot[i][number] = 1
                preds_onehot = torch.tensor(preds_onehot).to(torch.float32).to("mps")

                # 정확도 계산: 실제 값과 예측 값이 일치하는 부분만 정확도로 계산
                correct_preds = (preds_onehot == ys).float()  # 맞는 값은 1, 틀린 값은 0
                valid_mask = ys != 0  # 정답이 0이 아닌 부분만 유효한 값으로 간주
                correct_preds = correct_preds * valid_mask  # 0인 부분을 제외한 맞는 값만 남김
                epoch_train_correct += correct_preds.sum().item()  # 맞는 예측의 개수 합산
                total_train_samples += valid_mask.sum().item()  # 유효한 샘플 수

            train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = epoch_train_correct / total_train_samples
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            
            # valid
            model.eval()
            epoch_val_loss = 0
            epoch_val_correct = 0
            total_val_samples = 0
            # val_loss = 0

            with torch.no_grad():

                x_valid = np.expand_dims(valid_onehots[-win_size-1:-1], 0)
                x_valid = torch.from_numpy(x_valid).to(device, dtype=torch.float32)
                #MLP
                x_valid = x_valid[0]
                y_correct = torch.from_numpy( np.array([valid_onehots[-1]]) ).to(device, dtype=torch.float32)
                
                pred = model(x_valid)
                pred = F.softmax(pred, dim=1)
                
                val_loss = criterion(pred, y_correct)

                epoch_val_loss += val_loss.item()

                pred_argsort6 = pred.argsort(axis=1)[:, -6:]
                len_pred_argsort = len(pred_argsort6)
                preds_onehot = np.zeros((len_pred_argsort, 45))
                for i, numbers in enumerate(pred_argsort6):
                    for j, number in enumerate(numbers):
                        preds_onehot[i][number] = 1
                preds_onehot = torch.tensor(preds_onehot).to(torch.float32).to("mps")
                
                # 정확도 계산: 실제 값과 예측 값이 일치하는 부분만 정확도로 계산
                correct_val_preds = (preds_onehot == y_correct).float()  # 맞는 값은 1, 틀린 값은 0
                valid_mask = y_correct != 0  # 정답이 0이 아닌 부분만 유효한 값으로 간주
                correct_val_preds = correct_val_preds * valid_mask  # 0인 부분을 제외한 맞는 값만 남김
                epoch_val_correct += correct_val_preds.sum().item()  # 맞는 예측의 개수 합산
                total_val_samples += valid_mask.sum().item()  # 유효한 샘플 수

            val_loss = epoch_val_loss    
            val_accuracy = epoch_val_correct / total_val_samples
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # print("val diff ", val_accuracy, best_val_accuracy, val_accuracy > best_val_accuracy)
            # print("lossdiff ", val_loss, lowest_val_loss, val_loss < lowest_val_loss)
            if val_accuracy >= best_val_accuracy:
                
                best_val_accuracy = val_accuracy

                if val_loss < lowest_val_loss:
                    lowest_val_loss = val_loss
                    best_epoch = epoch
                    torch.save(model.state_dict(), "best_model_ball_%d.pth"%(bn))
                    print("val loss(%f) acc(%f)"%(lowest_val_loss, best_val_accuracy))

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss:.8f}, Train Accuracy: {train_accuracy:.8f}, '
                f'Val Loss: {val_loss:.8f}, Val Accuracy: {val_accuracy:.8f}')
            
        
        # torch.load("best_model_ball_%d.pth"%(bn))
        model.load_state_dict(torch.load("best_model_ball_%d.pth"%(bn)))
        model.eval()

        test_pred = model(torch.tensor([test_onehots]).to(torch.float32).to("mps"))
        test_pred = F.softmax(test_pred, dim=1)
        test_pred = test_pred.cpu().detach().numpy().flatten()
        for i in range(5):
            final_numbers.append(gen_numbers_from_probability(test_pred))
        final_numbers = np.array(final_numbers)
        print("final_numbers : \n", final_numbers)
        print("")
        result_numbers.append(final_numbers)


        # 학습 및 검증 손실과 정확도 시각화
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss over Epochs')

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy over Epochs')

        # plt.show()
        plt.savefig("result%.2d.jpg"%(bn))
        plt.close()

        

        # break
    print(result_numbers)
    print("Train done.")
    
if __name__ == "__main__":
    main()
