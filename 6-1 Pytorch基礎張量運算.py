# 匯入torch套件
import torch

torch.manual_seed(3)  # 固定亂數種子，方便對照結果
X = torch.randn((1, 6))  # 宣告亂數矩陣(常態分佈，平均為0、標準差為1)
W = torch.randn_like(X)  # 宣告與X同大小的亂數矩陣
B = torch.randn((1, 1))
W = W.view(6, 1)  # 將W大小轉換為(6, 1)

# 輸入乘權重加偏差，傳入激勵函數sigmoid得到輸出
Y = torch.sigmoid(torch.mm(X, W) + B)
print(Y)
