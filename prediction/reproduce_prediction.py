import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

# Configuration
FILE_PATH = '/Users/jeremyye/workspace/lotto645/history_from_cafe.csv'
WINDOW_SIZE = 5     # Number of past rounds to consider
HIDDEN_SIZE = 64    # Reduced from 128 (Smaller dataset per model)
NUM_LAYERS = 1      # Reduced from 2
EPOCHS = 1000       # Increased epochs as data is smaller / faster
LEARNING_RATE = 0.001
BATCH_SIZE = 16     # Smaller batch size
VALIDATION_SPLIT = 0.1 # Use small validation set

# 1. Data Utils
def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    # --- MODIFICATION: Drop the last row to simulate predicting it ---
    print(f"Original data length: {len(df)}")
    last_row = df.iloc[-1]
    print(f"Target (Last Row) - Draw {last_row[1]}: BallSet={last_row[0]}, Numbers={last_row[2:8].values}")
    
    df = df.iloc[:-1] # Remove the last row
    print(f"Training data length: {len(df)}")
    # ----------------------------------------------------------------
    
    ball_sets = df.iloc[:, 0].values.astype(int)
    numbers = df.iloc[:, 2:8].values.astype(int)
    return ball_sets, numbers, last_row

class LottoFilteredDataset(Dataset):
    def __init__(self, all_numbers, all_ballsets, target_ballset_id, window_size):
        self.x_data = []
        self.y_data = []
        
        num_rounds = len(all_numbers)
        
        # Pre-calculate one-hot for all numbers
        one_hot_numbers = np.zeros((num_rounds, 45), dtype=np.float32)
        for i, row in enumerate(all_numbers):
            for num in row:
                one_hot_numbers[i, int(num)-1] = 1.0
        
        # Iterate over all rounds
        for i in range(num_rounds - window_size):
            # Target round index
            target_idx = i + window_size
            
            # Check if this round used the target ball set
            if all_ballsets[target_idx] == target_ballset_id:
                # Yes, this is a training sample for this Ball Set Model
                
                # Input: Previous N rounds (Global history)
                x = one_hot_numbers[i : target_idx]
                y = one_hot_numbers[target_idx]
                
                self.x_data.append(x)
                self.y_data.append(y)
                
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.x_data[idx]), torch.FloatTensor(self.y_data[idx])

# 2. Model Definition
class LottoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.2):
        super(LottoLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # self.dropout = nn.Dropout(dropout) # No dropout for single layer usually, or use filters
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

# 3. Training Function
def train_model(ballset_id, all_numbers, all_ballsets, device):
    print(f"\nTraining Model for Ball Set #{ballset_id}...")
    
    dataset = LottoFilteredDataset(all_numbers, all_ballsets, ballset_id, WINDOW_SIZE)
    
    if len(dataset) < 10:
        print(f"Warning: Not enough data for Ball Set #{ballset_id} (Count: {len(dataset)}). Skipping.")
        return None
        
    # Split Train/Val
    total_len = len(dataset)
    val_len = max(int(total_len * VALIDATION_SPLIT), 5) # At least 5 items for val
    train_len = total_len - val_len
    
    train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
    val_dataset = torch.utils.data.Subset(dataset, range(train_len, total_len))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = LottoLSTM(input_size=45, hidden_size=HIDDEN_SIZE, output_size=45, num_layers=NUM_LAYERS).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Simple Early Stopping
    best_val_loss = float('inf')
    patience = 30
    counter = 0
    best_model_state = None
    
    for epoch in range(EPOCHS):
        model.train()
        avg_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            avg_train_loss += loss.item()
        
        avg_train_loss /= len(train_loader)

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # if (epoch+1) % 100 == 0:
        #    print(f"  Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                # print(f"  Early stopping at epoch {epoch+1}")
                break
                
    if best_model_state:
        model.load_state_dict(best_model_state)
        
    print(f"  Finished. Samples: {total_len}. Best Val Loss: {best_val_loss:.4f}")
    return model

# 4. Main Execution
def main():
    all_ball_sets, all_numbers, last_row = load_data(FILE_PATH)
    target_ballset_id = int(last_row[0])
    target_numbers = last_row[2:8].values.astype(int)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    models = {}
    
    # Train only the target ball set model for efficiency, or train all?
    # Let's train only the target one to save time, unless we want to see if other models would have coincidentally guessed it.
    # But strictly speaking, if we knew the ball set was 2, we'd use Model 2.
    print(f"Target Round used Ball Set: {target_ballset_id}")
    
    # Train specific model
    model = train_model(target_ballset_id, all_numbers, all_ball_sets, device)
    
    if model is None:
        print("Model training failed.")
        return

    # --- Prediction Phase ---
    print("\n" + "="*50)
    print("PREDICTION FOR TARGET ROUND (Simulation)")
    print("="*50)
    
    # Global Input: Last 5 rounds of history (from the TRUNCATED dataset)
    last_window_numbers = all_numbers[-WINDOW_SIZE:]
    
    # One-hot
    input_one_hot = np.zeros((1, WINDOW_SIZE, 45), dtype=np.float32)
    for i, row in enumerate(last_window_numbers):
        for num in row:
            input_one_hot[0, i, int(num)-1] = 1.0
            
    input_tensor = torch.FloatTensor(input_one_hot).to(device)
    
    print(f"Input History:\n{last_window_numbers}")
    print("-" * 50)
    
    model.eval()
    with torch.no_grad():
        pred_probs = model(input_tensor)
        
    probs = pred_probs.cpu().numpy()[0]
    
    # Analyze probabilties of the ACTUAL winning numbers
    print("Actual Winning Numbers:", target_numbers)
    print("Probabilities assigned to winning numbers:")
    
    ranked_indices = np.argsort(probs)[::-1] # High to low
    
    for num in target_numbers:
        idx = num - 1
        prob = probs[idx]
        rank = np.where(ranked_indices == idx)[0][0] + 1
        print(f"  Number {num}: Probability {prob:.4f}, Rank #{rank}")
        
    # Top 10 Predictions
    print("-" * 20)
    print("Top 10 High Probability Numbers:")
    top10_indices = ranked_indices[:10]
    for idx in top10_indices:
        print(f"  Number {idx+1}: Probability {probs[idx]:.4f}")

if __name__ == "__main__":
    main()
