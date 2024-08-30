# 匯入套件
import cv2
import numpy as np
import matplotlib.pyplot as plt

original_img = cv2.imread('BigBen.jpg')  # 讀入圖片
original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 交換通道
plt.figure(figsize=(9, 6))  # 設定輸出大小
plt.imshow(original_img)
plt.show()

# 平均模糊
kernal = np.ones((5, 5)) / 25
filtered_img = cv2.filter2D(original_img, -1, kernal)
fig, axes = plt.subplots(nrows=1, ncols=2)  # 建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(original_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()

# 高斯模糊
original_img = cv2.GaussianBlur(original_img, (5, 5), 0)
fig, axes = plt.subplots(nrows=1, ncols=2)  # 建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(original_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()

# 中位數模糊

rand = np.random.uniform(0, 1, size=original_img.shape)  # 亂數矩陣
ratio = 0.1  # 決定胡椒鹽像素比率
salt_and_pepper_img = (ratio <= rand) * (rand <= 1 - ratio) * original_img + (rand > 1 - ratio) * 255  # 生成胡椒鹽雜訊圖片
salt_and_pepper_img = salt_and_pepper_img.astype('uint8')  # 轉為8位元陣列

filtered_img = cv2.medianBlur(salt_and_pepper_img, 5)
fig, axes = plt.subplots(nrows=1, ncols=2)  # 建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(salt_and_pepper_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()

filtered_img = cv2.GaussianBlur(salt_and_pepper_img, (5, 5), 0)
fig, axes = plt.subplots(nrows=1, ncols=2)  # 建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(salt_and_pepper_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()

# 銳化濾波
kernal = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
filtered_img = cv2.filter2D(original_img, -1, kernal)
fig, axes = plt.subplots(nrows=1, ncols=2)  # 建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(original_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()

# 邊緣濾波
kernal = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
filtered_img = cv2.filter2D(original_img, -1, kernal)
fig, axes = plt.subplots(nrows=1, ncols=2)  #建構圖表陣列
fig.set_size_inches(15, 5)
axes[0].imshow(original_img)  # 左邊輸出原圖
axes[1].imshow(filtered_img)  # 右邊輸出濾波後的圖
plt.show()
