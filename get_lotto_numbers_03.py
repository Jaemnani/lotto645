import numpy as np
# import requests
# from bs4 import BeautifulSoup
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
import matplotlib.pyplot as plt

reward = [100_000, 5_000, 200, 5, 0.5]

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
        self.activate = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.activate(out)
        return out

def main():     
    history_path = "./history_from_cafe.csv"
    is_exists_history_file = os.path.exists(history_path)
    if is_exists_history_file:
        history = np.loadtxt(history_path, delimiter=',').astype(int)
    else:
        print("데이터를 세팅해주세요.")
        return -1
    
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
                
        gap = int(len(numbers_onehot) / 10)
        train_onehots = numbers_onehot[: -gap*2]
        valid_onehots = numbers_onehot[-gap*2:-gap]
        test_onehots = numbers_onehot[-gap:]

        input_size = 45
        hidden_size = 256
        num_layers = 4 # mininum
        output_size = 45

        model = LSTMNet(input_size, hidden_size, num_layers, output_size).to(device)

        win_size = 16
        train_dst = LHistory(train_onehots, win_size)
        train_loader = data.DataLoader(
            train_dst, 
            batch_size = 128, 
            # shuffle = True,
            shuffle = False,
            # num_workers=1,
            drop_last=False
        )

        # criterion = nn.CrossEntropyLoss() # softmax
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr= 0.001)
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        # optimizer = optim.SGD(model.parameters(), lr=0.1)
        
        # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        cur_iters = 0
        cur_epochs = 0

        curve_1st = []
        curve_mean = []

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        num_epochs = 100
        for epoch in range(num_epochs):
            model.train()

            epoch_train_loss = 0
            epoch_train_correct = 0
            total_train_samples = 0

            for (xs, ys) in train_loader:
                cur_iters += 1
                xs = xs.to(device, dtype=torch.float32)
                ys = ys.to(device, dtype=torch.float32)
            
                optimizer.zero_grad()
                outputs = model(xs)

                loss = criterion(outputs, ys)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

                # correct calculation
                preds = torch.round(outputs)
                correct_preds = (preds == ys).sum().item()
                epoch_train_correct += correct_preds
                total_train_samples += ys.numel()

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
                from collections import Counter
                # def count_matches(row1, row2):
                #     common_counts = Counter(row1) & Counter(row2)
                #     # if len(common_counts) == 0:
                #     #     return 0
                #     # else:
                #     return common_counts
                x_valid = np.expand_dims(valid_onehots[-win_size-1:-1], 0)
                x_valid = torch.from_numpy(x_valid).to(device, dtype=torch.float32)
                y_correct = torch.from_numpy( np.array([valid_onehots[-1]]) ).to(device, dtype=torch.float32)
                # y_valid = valid_onehots[-1].argsort()[-6:] + 1
                
                pred = model(x_valid)
                # # res = np.zeros(45)
                # # res[pred[0].argsort()[-6:]] = 1.
                # pred[0, pred[0].argsort()[-6:]] = 1.
                # pred[0, pred[0].argsort()[:-6]] = 0.
                
                val_loss = criterion(pred, y_correct)

                epoch_val_loss += val_loss.item()

                pred = torch.round(pred)
                correct_val_preds = (pred == y_correct).sum().item()
                epoch_val_correct += correct_val_preds
                total_val_samples += y_correct.numel()
                # p_valid = (np.sort(pred.cpu().argsort()[:, -6:]) + 1).flatten()
                
                # res = count_matches(y_valid, p_valid)
                # res_check = torch.sum(pred == y_valid)
            val_loss = epoch_val_loss    
            val_accuracy = epoch_val_correct / total_val_samples
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

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

        plt.show()

            # last_input = torch.from_numpy(np.expand_dims(valid_onehots[-win_size:],0)).to(device, dtype=torch.float32)
            # last_pred = model(last_input)
            # last_pred = last_pred[0].argsort()[-6:] + 1
            # if sum(res.values()) >= 3:
            #     print(f'Epoch [{epoch+1}/{num_epochs} || iter:{cur_iters}], Loss: {loss.item():.1f}, val_Loss: {val_loss.item():.1f} /', res, np.sort(last_pred.cpu()) )
            #     print(epoch, ":", y_valid, p_valid, np.sort(last_pred.cpu()))
            # else:
            #     if epoch == num_epochs-1:
            #         print(epoch, ":", y_valid, p_valid, np.sort(last_pred.cpu()))

                    

            
            # scheduler.step()
                # print("val lossres , "Check Count : ", sum(res.values()))
        # break
    print("Train done.")
    
if __name__ == "__main__":
    main()
