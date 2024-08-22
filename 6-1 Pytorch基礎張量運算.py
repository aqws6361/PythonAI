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

X = torch.randn((3, 4, 5))
print(X.shape)  # 原張量大小
print(X.view(3, 4, 5, 1).shape)
print(X.view(6, 10).shape)
print(X.view(3, 2, -1).shape)

torch.manual_seed(3)
Date_Count = 2  # 資料量
N_Input = 3  # 單筆資料向量長(輸入層神經元數)
N_Hidden = 2  # 隱藏層神經元數
N_Output = 2  # 輸出層神經元數

X = torch.randn(Date_Count, N_Input)  # 輸入張量

W1 = torch.randn(N_Input, N_Hidden)  # 隱藏層權重張量
B1 = torch.randn((Date_Count, N_Hidden))  # 隱藏層偏差張量

W2 = torch.randn(N_Hidden, N_Output)  # 輸出層權重張量
B2 = torch.randn((Date_Count, N_Output))  # 輸出層偏差張量

Hidden = torch.sigmoid(torch.mm(X, W1) + B1)  # 輸出層傳至隱藏層
Output = torch.sigmoid(torch.mm(Hidden, W2) + B2)  #隱藏層傳至輸出層

print(Output)