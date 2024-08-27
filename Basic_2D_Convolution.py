import torch

# 輸入張量
Input = torch.Tensor([
    [4, 2, 9, 2, 6],
    [2, 6, 1, 7, 9],
    [8, 3, 5, 0, 3],
    [8, 1, 3, 3, 0]
])

# 卷積核
Kernel = torch.Tensor([
    [1, 2],
    [3, 4],
    [5, 6]
])

input_height = Input.shape[0]
input_width = Input.shape[1]
kernel_height = Kernel.shape[0]
kernel_width = Kernel.shape[1]

Output = torch.zeros((input_height - kernel_height + 1, input_width - kernel_width + 1))  # 空白輸出張量

for i in range(input_height - kernel_height + 1):
    for j in range(input_width - kernel_width + 1):  # 決定輸出張量的一項
        for k1 in range(kernel_height):
            for k2 in range(kernel_width):  # 以kernel的各個元素進行加權運算
                Output[i][j] += Input[i + k1][j + k2] * Kernel[k1][k2]
                # 卷積運算

print(Output)
