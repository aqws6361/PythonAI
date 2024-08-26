# 匯入套件
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.pylab import plt


# 建構神經網路類別
class Network(nn.Module):
    # 初始化
    def __init__(self, input_size, output_size):
        super().__init__()  # 初始化父類別nn.Module
        # 宣告各神經網路層
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, output_size)

    # 正向傳播函數
    def forward(self, x):
        # 將輸入依序傳入各個物件
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# 超參數
batch_size = 64
lr = 0.001
epochs = 10

NN = Network(784, 10)  # 宣告神經網路
criterion = nn.CrossEntropyLoss()  # 宣告損失函數
optimizer = optim.Adam(NN.parameters(), lr=lr)  # 宣告梯度優化函數

# 資料迭代器
train_set = datasets.MNIST('./', download=True, train=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./', download=True, train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


def visualize(model, data_loader):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)  # 隨機取得一組資料

    model.eval()  # 轉為測試模式(不訓練)
    print(f'Labels: {" ".join(str(int(label)) for label in labels[: 5])}')  # 輸出標籤
    predict = model(images.view(images.shape[0], -1))
    pred_label = torch.max(predict.data, 1).indices
    print(f'Predictions: {" ".join(str(int(label)) for label in pred_label[: 5])}')  # 輸出預測

    fig, axes = plt.subplots(nrows=2, ncols=5)  # 建構圖表陣列
    fig.set_size_inches(13, 8)  # 設定圖表大小
    plt.subplots_adjust(wspace=1, hspace=0.1)  # 設定子圖表間距
    for i in range(5):
        axes[0][i].imshow(images[i].permute(1, 2, 0).numpy().squeeze(), cmap=plt.cm.gray)  # 輸出圖片
        x = list(range(10))
        y = torch.softmax(predict.data[i], 0)  # 取得預測機率
        axes[1][i].barh(x, y)  # 輸出圖表
        for j, v in enumerate(y):
            axes[1][i].text(1.1 * max(y), j - 0.1, str("{:1.4f}".format(v)), color='black')  # 輸出機率
    plt.show()


# visualize(NN, train_loader)

def train(model, epochs, train_loader, test_loader):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for e in range(epochs):  # 訓練epochs個週期
        model.train()  # 將神經網路設為訓練模式
        loss_sum, correct_cnt = 0, 0
        for image, label in train_loader:
            optimizer.zero_grad()  # 清除各參數梯度

            image = image.view(image.shape[0], -1)  # 把二維張量的圖片壓成一維張量
            predict = model(image)  # 輸入一批資料，獲得預測機率分布
            loss = criterion(predict, label)  # 由模型輸出和正確標籤算得損失

            pred_label = torch.max(predict.data, 1).indices  # 取得預測分類
            correct_cnt += (pred_label == label).sum()  # 取得預測正確次數
            loss_sum += loss.item()  # 加總批次損失

            loss.backward()  # 損失反向傳播
            optimizer.step()  # 更新參數

        train_loss.append(loss_sum / len(train_loader))  # 計算週期平均損失
        train_acc.append(float(correct_cnt) / (len(train_loader) * batch_size))  # 計算週期平均準確率
        print(f'Epoch {e + 1:2d} Train Loss: {train_loss[-1]:.10f} Train Acc: {train_acc[-1]:.4f}', end=' ')
        model.eval()  # 轉為測試模式(不訓練)
        loss_sum, correct_cnt = 0, 0
        with torch.no_grad():
            for image, label in test_loader:
                predict = model(image.view(image.shape[0], -1))
                loss = criterion(predict, label)
                pred_label = torch.max(predict.data, 1).indices
                correct_cnt += (pred_label == label).sum()
                loss_sum += loss.item()
        test_loss.append(loss_sum / len(test_loader))
        test_acc.append(float(correct_cnt) / (len(test_loader) * batch_size))
        print(f'Test Loss: {test_loss[-1]:.10f} Test Acc: {test_acc[-1]:.4f}')
    return train_loss, train_acc, test_loss, test_acc


train_loss, train_acc, test_loss, test_acc = train(NN, epochs, train_loader, test_loader)  # 訓練

# 損失函數圖表
plt.xlabel('Epochs')  # x軸意義
plt.ylabel('Loss Value')  # y軸意義
plt.plot(train_loss, label='Train Set')  # 訓練集損失折線
plt.plot(test_loss, label='Test Set')  # 測試集損失折線
plt.legend()  # 畫圖例
plt.show()
# 預測準確率圖表
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc, label='Train Set')
plt.plot(test_acc, label='Test Set')
plt.legend()
plt.show()

visualize(NN, test_loader)
