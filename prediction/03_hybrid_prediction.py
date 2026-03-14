import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Configuration
FILE_PATH = '/Users/jeremyye/workspace/lotto645/history_from_cafe.csv'
WINDOW_SIZE = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 1
EPOCHS = 1000
LEARNING_RATE = 0.001
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1

class LottoFilteredDataset(Dataset):
    def __init__(self, all_numbers, all_ballsets, target_ballset_id, window_size):
        self.x_data = []
        self.y_data = []
        
        num_rounds = len(all_numbers)
        one_hot_numbers = np.zeros((num_rounds, 45), dtype=np.float32)
        for i, row in enumerate(all_numbers):
            for num in row:
                one_hot_numbers[i, int(num)-1] = 1.0
        
        for i in range(num_rounds - window_size):
            target_idx = i + window_size
            if all_ballsets[target_idx] == target_ballset_id:
                x = one_hot_numbers[i : target_idx]
                y = one_hot_numbers[target_idx]
                self.x_data.append(x)
                self.y_data.append(y)
                
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data[idx]), torch.FloatTensor(self.y_data[idx])

class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LottoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :] 
        out = self.fc(out)
        out = self.sigmoid(out)
        return out

def get_cold_numbers(df_history, top_n=10):
    """
    Identifies 'Cold' numbers: Numbers that haven't appeared for the longest time.
    Returns: List of (number, days_since_last_appearance) tuples.
    """
    all_numbers = df_history.iloc[:, 2:8].values
    last_appearance_idx = {}
    
    total_rounds = len(all_numbers)
    
    # Initialize all numbers as 'never seen' (index -1)
    for num in range(1, 46):
        last_appearance_idx[num] = -1
        
    # Scan history forward
    for idx, row in enumerate(all_numbers):
        for num in row:
            last_appearance_idx[num] = idx
            
    # Calculate gap (Current Round - Last Seen Round)
    cold_stats = []
    for num in range(1, 46):
        last_idx = last_appearance_idx[num]
        if last_idx == -1:
             gap = total_rounds # Never seen
        else:
            gap = total_rounds - 1 - last_idx
        cold_stats.append((num, gap))
    
    # Sort by gap descending (Longest gap = Coldest)
    cold_stats.sort(key=lambda x: x[1], reverse=True)
    
    return cold_stats[:top_n]

def train_lstm(ballset_id, all_numbers, all_ball_sets, device):
    print(f"\n[AI] Training Model for Ball Set #{ballset_id}...")
    dataset = LottoFilteredDataset(all_numbers, all_ball_sets, ballset_id, WINDOW_SIZE)
    
    train_len = int(len(dataset) * 0.9)
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LottoLSTM(45, HIDDEN_SIZE, 45, NUM_LAYERS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
    return model

def main():
    # 1. Load Data
    df = pd.read_csv(FILE_PATH, header=None)
    
    # --- SIMULATION MODE: Hide the last row (Draw 1203) ---
    real_target_row = df.iloc[-1]
    df_train = df.iloc[:-1] # Train on everything EXCEPT the last row
    
    print(f"--- HYBRID SYSTEM SIMULATION [Draw {real_target_row[1]}] ---")
    print(f"Target Numbers: {real_target_row[2:8].values}")
    
    all_ball_sets = df_train.iloc[:, 0].values.astype(int)
    all_numbers = df_train.iloc[:, 2:8].values.astype(int)
    
    # 2. Get Cold Numbers (Statistical Layer)
    cold_numbers_stat = get_cold_numbers(df_train, top_n=15)
    cold_candidates_list = [x[0] for x in cold_numbers_stat]
    print(f"\n[Stats] Top Cold Numbers (Num:Gap): {cold_numbers_stat[:10]}")
    
    # 3. Get AI Predictions (ML Layer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    target_bs_id = int(real_target_row[0]) # Use the true ballset for fairness in testing logic
    
    print(f"Assuming Ball Set: {target_bs_id}")
    model = train_lstm(target_bs_id, all_numbers, all_ball_sets, device)
    
    # Predict
    last_window = all_numbers[-WINDOW_SIZE:]
    input_one_hot = np.zeros((1, WINDOW_SIZE, 45), dtype=np.float32)
    for i, row in enumerate(last_window):
        for num in row:
            input_one_hot[0, i, int(num)-1] = 1.0
    
    model.eval()
    with torch.no_grad():
        pred_probs = model(torch.FloatTensor(input_one_hot).to(device))
    probs = pred_probs.cpu().numpy()[0]
    
    # Get Top Hot Candidates
    hot_indices = np.argsort(probs)[::-1]
    hot_candidates_list = [(idx+1, probs[idx]) for idx in hot_indices]
    
    print(f"\n[AI] Top Hot Numbers (Num:Prob): {[(n, f'{p:.2f}') for n, p in hot_candidates_list[:10]]}")
    
    # 4. Hybrid Selection (4 Hot + 2 Cold)
    print("\n" + "="*40)
    print("HYBRID SELECTION (4 Hot + 2 Cold)")
    print("="*40)
    
    # Strategy:
    # A. Draw 4 from Top 15 Hot (Weighted by Prob)
    # B. Draw 2 from Top 10 Cold (Weighted by Gap)
    
    # Normalize Hot Probs
    hot_pool_size = 15
    hot_pool = hot_candidates_list[:hot_pool_size]
    hot_nums = [x[0] for x in hot_pool]
    hot_probs = np.array([x[1] for x in hot_pool])
    hot_probs /= hot_probs.sum() # Normalize
    
    # Normalize Cold Weights
    cold_pool_size = 10
    cold_pool = cold_numbers_stat[:cold_pool_size]
    cold_nums = [x[0] for x in cold_pool]
    cold_gaps = np.array([x[1] for x in cold_pool], dtype=float)
    cold_probs = cold_gaps / cold_gaps.sum()
    
    # Generate 5 sets
    for i in range(5):
        selected_hot = np.random.choice(hot_nums, size=4, replace=False, p=hot_probs)
        selected_cold = np.random.choice(cold_nums, size=2, replace=False, p=cold_probs)
        
        final_set = np.concatenate((selected_hot, selected_cold))
        final_set = np.unique(final_set) # Handle rare overlap
        
        # If overlap occurred and we have < 6, fill from Hot pool
        while len(final_set) < 6:
            extra = np.random.choice(hot_nums, size=1)[0]
            if extra not in final_set:
                final_set = np.append(final_set, extra)
                
        final_set = np.sort(final_set)
        
        # Check hitting
        match_count = len(set(final_set) & set(real_target_row[2:8].values))
        print(f"Set #{i+1}: {final_set} | Matches: {match_count}")
        if match_count >= 3:
            print(f"   >>> GOOD MATCH! ({match_count} hits)")
            
    # Verification Info
    print("\n[Analysis of Target Numbers vs Hybrid Pools]")
    target_set = set(real_target_row[2:8].values)
    
    hot_pool_set = set(hot_nums)
    cold_pool_set = set(cold_nums)
    
    print(f"Target Numbers: {target_set}")
    print(f"In Hot Pool (Top {hot_pool_size}): {target_set & hot_pool_set}")
    print(f"In Cold Pool (Top {cold_pool_size}): {target_set & cold_pool_set}")
    
    missed = target_set - hot_pool_set - cold_pool_set
    if missed:
        print(f"Still Missed: {missed} (Neither Hot nor Cold enough?)")

if __name__ == "__main__":
    main()
