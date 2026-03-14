import numpy as np
import random
import matplotlib.pyplot as plt

# 1. 단순화된 모델 사용한 랜덤 추첨기
def simple_random_lotto(draw_count=5):
    # 번호는 1부터 45까지
    numbers = np.arange(1, 46)
    results = []

    for _ in range(draw_count):
        # 6개의 번호를 랜덤하게 뽑기
        selected_numbers = random.sample(list(numbers), 6)
        selected_numbers.sort()
        results.append(selected_numbers)
    
    return np.array(results)

# 2. 빈도수 기반 확률 수정 랜덤 추첨기
def frequency_based_lotto(history, draw_count=5):
    # 번호는 1부터 45까지
    numbers = np.arange(1, 46)
    frequency = np.zeros(45)
    
    # 과거 데이터를 기반으로 빈도수 계산
    for draw in history:
        for number in draw:
            frequency[number - 1] += 1
    
    # 각 번호의 확률을 계산 (빈도수로 확률 계산)
    total_draws = len(history)
    probabilities = frequency / total_draws / 6.
    # 확률에 따라 번호를 추첨
    results = []
    for _ in range(draw_count):
        selected_numbers = np.random.choice(numbers, 6, p=probabilities, replace=False)
        selected_numbers.sort()
        results.append(selected_numbers)
    
    return np.array(results)

# 예시 데이터 (1~45 사이 번호를 랜덤으로 생성한 100번의 추첨 기록)
history = []
for _ in range(100):
    history.append(np.random.choice(np.arange(1, 46), 6, replace=False))

# 1. 단순 랜덤 추첨기 결과
simple_results = simple_random_lotto()
print("Simple Random Lotto Results:")
print(simple_results)

# 2. 빈도수 기반 확률 수정 랜덤 추첨기 결과
frequency_results = frequency_based_lotto(history)
print("Frequency Based Lotto Results:")
print(frequency_results)

# 시각화: 빈도수 기반 확률 수정 추첨기의 번호 출현 빈도 시각화
flat_history = np.array(history).flatten()
unique, counts = np.unique(flat_history, return_counts=True)
frequency_dict = dict(zip(unique, counts))

# 출현 빈도 시각화
plt.bar(frequency_dict.keys(), frequency_dict.values())
plt.xlabel('Lotto Numbers')
plt.ylabel('Frequency')
plt.title('Lotto Number Frequency in History')
plt.show()
