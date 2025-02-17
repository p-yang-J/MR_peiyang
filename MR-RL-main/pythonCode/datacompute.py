import pandas as pd


columns = ["Subjects", "Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

data_with_mr = [
    ["A", 8, 12, 7, 2, 4, 3],
    ["B", 14, 10, 12, 9, 12, 16],
    ["C", 2, 0, 1, 1, 0, 1],
    ["D", 0, 1, 2, 3, 4, 1],
    ["E", 4, 5, 0, 5, 8, 3],
    ["F", 1, 2, 4, 2, 2, 4],
    ["G", 0, 10, 0, 0, 8, 0]
]

data_without_mr = [
    ["A", 7, 9, 12, 7, 9, 5],
    ["B", 7, 7, 6, 5, 14, 6],
    ["C", 2, 1, 4, 3, 4, 1],
    ["D", 2, 3, 4, 4, 5, 4],
    ["E", 6, 8, 9, 10, 10, 10],
    ["F", 11, 9, 10, 9, 8, 9],
    ["G", 0, 0, 0, 1, 3, 0]
]


df_with_mr = pd.DataFrame(data_with_mr, columns=columns)
df_without_mr = pd.DataFrame(data_without_mr, columns=columns)

# 计算每个问题的平均分和方差
average_scores_with_mr = df_with_mr.mean()
average_scores_without_mr = df_without_mr.mean()

variance_scores_with_mr = df_with_mr.var()
variance_scores_without_mr = df_without_mr.var()

print("平均分 (with MR):\n", average_scores_with_mr)
print("平均分 (without MR):\n", average_scores_without_mr)

print("方差 (with MR):\n", variance_scores_with_mr)
print("方差 (without MR):\n", variance_scores_without_mr)

