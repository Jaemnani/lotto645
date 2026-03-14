import pandas as pd
import numpy as np

# Configuration
FILE_PATH = '/Users/jeremyye/workspace/lotto645/history_from_cafe.csv'

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    # Exclude the very last row (the target draw) to simulate "past knowledge"
    # Target draw is 1203 (index 738 in 0-indexed df if total is 739)
    # But wait, original len was 739. Last row is 1203.
    # We want to see status relative to history UP TO 1202.
    
    df_history = df.iloc[:-1] 
    last_row = df.iloc[-1]
    
    return df_history, last_row

def analyze_frequency(df, lookback, target_numbers):
    # Get last N draws
    recent_df = df.iloc[-lookback:]
    
    numbers_history = recent_df.iloc[:, 2:8].values.flatten()
    
    # Calculate counts
    counts = pd.Series(numbers_history).value_counts().sort_values(ascending=False)
    
    print(f"\n--- Lookback Period: Last {lookback} Draws ---")
    print(f"Total numbers drawn: {len(numbers_history)}")
    
    for num in target_numbers:
        count = counts.get(num, 0)
        rank = list(counts.index).index(num) + 1 if num in counts.index else 'N/A'
        
        # Calculate percentile/status
        # If rank is high (e.g., 1-10), it's HOT.
        # If count is 0, it's COLD.
        print(f"  Target {num}: Count {count}, Rank #{rank}")

def main():
    df, last_row = load_data(FILE_PATH)
    target_numbers = last_row[2:8].values.astype(int)
    print(f"Target Numbers (Draw {last_row[1]}): {target_numbers}")
    
    # Analyze Last 5 (Short Term)
    analyze_frequency(df, 5, target_numbers)
    
    # Analyze Last 10
    analyze_frequency(df, 10, target_numbers)
    
    # Analyze Last 30 (Medium Term)
    analyze_frequency(df, 30, target_numbers)
    
    # Analyze Last 100 (Long Term)
    analyze_frequency(df, 100, target_numbers)

if __name__ == "__main__":
    main()
