import numpy as np
from time import time
result = np.array(
    [[[4, 17, 18, 34, 39, 42], [2, 3, 14, 22, 29, 37], [10, 22, 25, 38, 39, 44], [6, 10, 17, 28, 38, 40], [3, 6, 22, 24, 30, 32]], [[7, 9, 10, 30, 39, 42], [12, 20, 21, 35, 38, 39], [9, 12, 13, 26, 33, 36], [9, 11, 17, 29, 30, 36], [4, 12, 17, 22, 27, 41]], [[7, 10, 11, 26, 42, 44], [1, 4, 25, 26, 29, 37], [5, 7, 12, 26, 35, 44], [5, 7, 11, 26, 30, 36], [5, 12, 20, 21, 26, 32]], [[3, 4, 22, 25, 35, 45], [7, 8, 16, 35, 38, 39], [2, 10, 13, 16, 24, 40], [1, 3, 15, 16, 35, 45], [7, 8, 20, 22, 30, 44]], [[2, 7, 9, 21, 33, 37], [3, 12, 16, 18, 33, 41], [5, 21, 30, 33, 41, 45], [4, 5, 7, 8, 18, 36], [2, 10, 11, 12, 20, 33]]]
    )

    
def get_result_numbers(multi_results, rng):
    """다수결 확정: 빈도 우선, 동률은 랜덤으로 섞어 상위 6개 선택."""
    unique_vals , counts = np.unique(multi_results, return_counts=True)
    # 동률 편향 방지를 위해 먼저 섞고, 그 위에서 빈도 내림차순 정렬
    rng_perm = rng.permutation(len(unique_vals))
    # rng_perm = np.random.permutation(len(unique_vals))
    unique_vals = unique_vals[rng_perm]
    counts = counts[rng_perm]
    order = np.argsort(-counts)  # 빈도 내림차순, 동률은 섞인 순서 유지
    top6 = np.sort(unique_vals[order][:6])
    return top6

def original_get_result_numbers(multi_results):
    """원본 랜덤 보충 방식 (보관용)."""
    unique_vals , counts = np.unique(multi_results, return_counts=True)
    order1 = counts>1
    selected_unique_vals = unique_vals[order1]
    np.random.seed(None)
    if len(selected_unique_vals) > 6:
        choice_vals = np.random.choice(unique_vals, 6, replace=False)
    elif len(selected_unique_vals) == 6:
        choice_vals = selected_unique_vals
    else:
        choice_cnt = (6 - len(selected_unique_vals))
        order2 = counts == 1
        except_unique_vals = unique_vals[order2]
        choice_vals = np.random.choice(except_unique_vals, choice_cnt)
        choice_vals = np.concatenate((selected_unique_vals, choice_vals))
    choice_vals = np.sort(choice_vals)
    return choice_vals

USE_ORIGINAL = False  # True면 기존 랜덤 보충 로직 실행

rng = np.random.default_rng()

for i, numbers in enumerate(result):
    for cnt in range(2):
        if USE_ORIGINAL:
            choice_vals = original_get_result_numbers(numbers)
        else:
            choice_vals = get_result_numbers(numbers, rng)
        print(f"{i+1}th numbers : ", choice_vals)

print('done')
