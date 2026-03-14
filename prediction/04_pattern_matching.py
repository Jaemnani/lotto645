import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Configuration
FILE_PATH = '/Users/jeremyye/workspace/lotto645/history_from_cafe.csv'
WINDOW_SIZE = 5  # Length of the pattern sequence to match (e.g., last 5 draws)

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    # Exclude last row (Draw 1203) for simulation
    target_row = df.iloc[-1]
    df_history = df.iloc[:-1]
    
    return df_history, target_row

def create_vectors(df):
    # Convert each draw into a 45-dim one-hot vector
    all_numbers = df.iloc[:, 2:8].values.astype(int)
    num_draws = len(all_numbers)
    
    vectors = np.zeros((num_draws, 45), dtype=int)
    for i, row in enumerate(all_numbers):
        # 0-indexed one-hot
        for num in row:
            vectors[i, num-1] = 1
            
    return vectors

def find_nearest_patterns(vectors, window_size, k=10):
    # Prepare the "Query Sequence" (The most recent window)
    query_sequence = vectors[-window_size:] 
    # Flatten the query for KNN (shape: 1 x (window_size * 45))
    query_flat = query_sequence.flatten().reshape(1, -1)
    
    # Prepare Training Data
    # We slide the window across history. 
    # Limit: We can't use the very last window as training data (it's the query).
    # And we must ensure there is a "Next Draw" available for each window.
    
    X_train = []
    y_next_indices = [] # Index of the draw *immediately following* the window
    
    # Stop before the query window starts
    # Total vectors: N. Query is indices [N-W, N].
    # So training windows can end at N-W-1.
    
    num_samples = len(vectors) - window_size 
    
    for i in range(num_samples):
        # Window: [i, i+window_size]
        window = vectors[i : i+window_size]
        
        # We need the flattened window
        X_train.append(window.flatten())
        
        # The "target" is the NEXT draw's index (i + window_size)
        # Check if i + window_size is within valid range
        if i + window_size < len(vectors):
            y_next_indices.append(i + window_size)
            
    X_train = np.array(X_train)
    
    # Use KNN
    # Metric: 'cityblock' (Manhattan) matches bit differences well for binary data, 
    # or 'euclidean'. Let's use euclidean.
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(X_train)
    
    distances, indices = knn.kneighbors(query_flat)
    
    return distances[0], indices[0], y_next_indices

def main():
    # 1. Load Data
    df_history, target_row = load_data(FILE_PATH)
    target_numbers = target_row[2:8].values.astype(int)
    target_set = set(target_numbers)
    
    print(f"--- KNN Pattern Matching Simulation [Draw {target_row[1]}] ---")
    print(f"Target Numbers: {target_numbers}")
    
    # 2. Vectorize
    vectors = create_vectors(df_history)
    
    # 3. Find Neighbors
    K = 20
    distances, neighbor_indices, y_next_map = find_nearest_patterns(vectors, WINDOW_SIZE, k=K)
    
    print(f"\nSearching for similar patterns (Window Size: {WINDOW_SIZE})...")
    print("-" * 60)
    
    matched_any = False
    best_hit_count = 0
    best_set = None
    
    for rank, (dist, idx) in enumerate(zip(distances, neighbor_indices)):
        # idx is the index in X_train. 
        # The 'window' was df_history[idx : idx+W].
        # The 'prediction' is df_history[idx+W].
        
        next_draw_idx = idx + WINDOW_SIZE
        # Safety check
        if next_draw_idx >= len(vectors):
            continue
            
        next_draw_vec = vectors[next_draw_idx]
        # Convert back to numbers
        predicted_numbers = np.where(next_draw_vec == 1)[0] + 1
        
        # Compare with Actual Target
        hits = target_set & set(predicted_numbers)
        hit_count = len(hits)
        
        # Historical Context
        # Map back to DataFrame index to get Draw Number
        # df_history index aligns with vectors index
        hist_draw_row = df_history.iloc[next_draw_idx]
        hist_draw_no = hist_draw_row[1]
        
        print(f"Neighbor #{rank+1} (Dist: {dist:.2f}): Similar to Draw ~{hist_draw_no}")
        print(f"   -> Followed by: {predicted_numbers}")
        print(f"   -> Hits with Target: {len(hits)} {hits if hits else ''}")
        
        if hit_count > best_hit_count:
            best_hit_count = hit_count
            best_set = predicted_numbers
            
        if hit_count >= 5: # 5 or 6 matches is Amazing
             matched_any = True
             print("   >>> FOUND HIGH MATCH! <<<")
             
    print("-" * 60)
    print(f"Best Match in Top {K}: {best_hit_count} hits")
    if best_set is not None:
        print(f"Best Set: {best_set}")
        
    if best_hit_count >= 5:
        print("\nCONCLUSION: YES, this result WAS predictable using historical pattern matching.")
    else:
        print("\nCONCLUSION: NO, even historical pattern matching failed to find a close precedent.")

if __name__ == "__main__":
    main()
