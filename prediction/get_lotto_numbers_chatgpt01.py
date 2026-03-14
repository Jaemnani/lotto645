import pandas as pd
from collections import Counter, defaultdict
import random

# 1) CSV 데이터 불러오기
df = pd.read_csv('history_from_cafe.csv', header=None)

# 만약 컬럼명을 지정하고 싶다면:
df.columns = [
    'ball',   # 공 번호 (1~5)
    'round',  # 회차
    'n1', 'n2', 'n3', 'n4', 'n5', 'n6',  # 6개 메인
    'bonus'   # 보너스 번호
]

# 2) 공 번호별로 등장한 숫자 빈도 계산
#    (보너스 번호도 동일하게 한 번 등장한 것으로 세어 줍니다)
#    freq_by_ball[공번호] = {숫자: 등장횟수, ...}
row_count_by_ball = df['ball'].value_counts().to_dict()

freq_by_ball = defaultdict(Counter)

for idx, row in df.iterrows():
    ball_num = row['ball']
    
    # 메인 6개와 보너스까지 총 7개
    numbers = [
        row['n1'], row['n2'], row['n3'],
        row['n4'], row['n5'], row['n6'],
        row['bonus']
    ]
    
    for num in numbers:
        freq_by_ball[ball_num][num] += 1

# 3) 공 번호별 (1~45) 번호 등장 순위 확인
#    -> freq_by_ball_sorted[공번호] = [(번호, 출현횟수), ...] (출현횟수 내림차순)
freq_by_ball_sorted = {}
for ball_num in freq_by_ball:
    sorted_list = sorted(freq_by_ball[ball_num].items(), key=lambda x: x[1], reverse=True)
    freq_by_ball_sorted[ball_num] = sorted_list

# # 예시 출력(각 공에 대해 상위 10개만)
# for ball_num in sorted(freq_by_ball_sorted.keys()):
#     count_rows = row_count_by_ball.get(ball_num, 0)
#     print(f"=== 공 {ball_num} ===")
#     print(f"  - 데이터(행) 개수: {count_rows}개")

#     print(f"=== 공 {ball_num}에서 많이 나온 번호 TOP 10 ===")
#     for num, cnt in freq_by_ball_sorted[ball_num][:45]:
#         print(f"  번호 {num}: {cnt}회")
#     print()

# 4) 간단한 '예측' 함수 만들기
#    - "다음에 공 X가 뽑힌다면" 상위 6개 번호를 고정으로 내놓거나,
#      혹은 빈도수를 가중치 삼아 무작위 추출(가중 랜덤) 하는 방식을 시연
def predict_numbers(ball_num, top_k=6, use_weighted=False):
    """
    ball_num: 공 번호 (1~5)
    top_k   : 몇 개 번호를 예측할지 (default: 6)
    use_weighted: True면 빈도수를 가중치로 삼아 랜덤 추출
                  False면 단순 빈도 TOP_k 고정 리턴
    """
    if ball_num not in freq_by_ball_sorted:
        # 혹시 없는 공번호 들어오면 예외처리
        return []
    
    if not use_weighted:
        # 빈도수 상위 top_k개를 그대로 리턴
        return [num for (num, _) in freq_by_ball_sorted[ball_num][:top_k]]
    else:
        # 빈도수를 가중치로 하여 무작위 표본추출
        counter = freq_by_ball[ball_num]
        all_nums = list(counter.keys())      # 공 번호 내에 한번이라도 등장한 번호들
        weights = [counter[n] for n in all_nums]  # 해당 번호의 등장 횟수를 가중치로
        # random.choices(리스트, 가중치, k=뽑을 개수)
        chosen = random.choices(all_nums, weights=weights, k=top_k)
        chosen = sorted(set(chosen), key=lambda x: counter[x], reverse=True)
        # 중복이 있을 수 있으므로 set으로 한번 줄이고, 등장 빈도 기준으로 정렬
        # 만약 완전히 중복 포함해서 k개 뽑고 싶다면 set() 과정 제거
        return chosen


# 5) 예측 사용 예시
result = []
for i in range(5):
    bn = i+1
    print(f"=== 공 {bn}로 추첨된다고 가정했을 때, 빈도 TOP 6 예측 ===")
    res = predict_numbers(bn, top_k=6, use_weighted=False)
    res.sort()
    result.append(res)
    print(res)




# print("\n=== 공 3으로 추첨된다고 가정했을 때, 가중 랜덤 6개 예측 ===")
# print(predict_numbers(3, top_k=6, use_weighted=True))

print()