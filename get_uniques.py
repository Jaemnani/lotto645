import numpy as np
result = np.array([
    [[ 3,  4,  7, 13, 32, 42],
       [ 2,  4,  5,  8, 26, 40],
       [ 2,  6, 22, 24, 28, 31],
       [10, 11, 18, 27, 33, 43],
       [ 6, 17, 18, 21, 27, 36]], [[ 5, 10, 11, 14, 16, 19],
       [ 1,  4, 10, 13, 36, 43],
       [ 2, 13, 16, 17, 19, 26],
       [ 4,  6, 15, 24, 29, 36],
       [ 6, 17, 18, 27, 32, 43]], [[ 3,  7, 26, 36, 42, 44],
       [11, 16, 21, 25, 34, 36],
       [ 4,  5, 14, 21, 27, 36],
       [11, 27, 30, 31, 40, 44],
       [ 7, 11, 16, 28, 37, 38]], [[ 2,  4,  5,  7, 20, 33],
       [ 1,  3,  4, 20, 28, 41],
       [21, 22, 32, 37, 40, 41],
       [14, 22, 28, 37, 42, 45],
       [17, 27, 29, 37, 39, 41]], [[ 1,  7, 19, 24, 40, 41],
       [ 1,  2,  3, 29, 36, 39],
       [ 1, 14, 16, 23, 31, 32],
       [ 2,  3,  6, 20, 33, 40],
       [ 3, 30, 37, 38, 39, 40]]
       ])

for i, numbers in enumerate(result):
    unique_vals, counts = np.unique(numbers, return_counts=True)
    order1 = counts>1
    selected_unique_vals = unique_vals[order1]
    np.random.seed(42)
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
    print(f"{i+1}th numbers : ", choice_vals)
        

    


    # print("check")
print('done')

def get_result_numbers(multi_results):
    unique_vals , counts = np.unique(multi_results, return_counts=True)
    order1 = counts>1
    selected_unique_vals = unique_vals[order1]
    np.random.seed(42)
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

