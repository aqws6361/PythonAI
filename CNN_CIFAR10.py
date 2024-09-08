# 匯入套件
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.pylab import plt

# 超參數
batch_size = 100
lr = 0.001
epochs = 10

# 資料迭代器
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.CIFAR10('./', download=True, train=True, transform=transform)
test_set = datasets.CIFAR10('./', download=True, train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


# 建構CNN類別
class CNN_CIFAR10(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(input_size, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷積與池化
        x = self.conv1(x)  # (input_size, 32, 32) => (6, 28, 28)
        x = self.relu(x)
        x = self.pool(x)  # (6, 28, 28) => (9, 14, 14)
        x = self.conv2(x)  # (9, 14, 14) => (16, 10, 10)
        x = self.relu(x)
        x = self.pool(x)  # (16, 10, 10) => (16, 5, 5)
        # 四維張量轉二維張量
        x = x.view(-1, 16 * 5 * 5)  # (16, 5, 5) => (16 * 5 * 5)
        # 全連接層
        x = self.fc1(x)  # (16 * 5 * 5) => (120)
        x = self.relu(x)
        x = self.fc2(x)  # (120) => (84)
        x = self.relu(x)
        x = self.fc3(x)  # (84) => (output_size)
        return x


CNN = CNN_CIFAR10(3, 10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters(), lr=lr)
print(CNN)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 視覺化函數
def visualize(model, data_loader):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)  # 隨機取得一組資料

    model.eval()  # 轉為測試模式(不訓練)
    print(f'Labels: {" ".join(f"{classes[label]:>5s}" for label in labels[: 5])}')  # 輸出標籤
    predict = model(images)
    pred_label = torch.max(predict.data, 1).indices
    print(f'Predictions: {" ".join(f"{classes[label]:>5s}" for label in pred_label[: 5])}')  # 輸出預測

    fig, axes = plt.subplots(nrows=2, ncols=5)  # 建構圖表陣列
    fig.set_size_inches(13, 8)  # 設定圖表大小
    plt.subplots_adjust(wspace=1, hspace=0.1)  # 設定子圖表間距
    for i in range(5):
        axes[0][i].imshow((images[i] * 0.5 + 0.5).permute(1, 2, 0).numpy().squeeze())  # 輸出圖片
        x = list(range(10))
        y = torch.softmax(predict.data[i], 0)  # 取得預測機率
        axes[1][i].barh(x, y)  # 輸出圖表
        for j, v in enumerate(y):
            axes[1][i].text(1.1 * max(y), j - 0.1, str("{:1.4f}".format(v)), color='black')  # 輸出機率
    plt.show()


visualize(CNN, train_loader)


# 訓練函式
def train(model, epochs, train_loader, test_loader):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []  # 訓練紀錄陣列
    for e in range(epochs):
        model.train()
        loss_sum, correct_cnt = 0, 0
        for image, label in train_loader:
            optimizer.zero_grad()
            predict = model(image)
            loss = criterion(predict, label)
            pred_label = torch.max(predict.data, 1).indices
            correct_cnt += (pred_label == label).sum()
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        train_loss.append(loss_sum / len(train_loader))
        train_acc.append(float(correct_cnt) / (len(train_loader) * batch_size))
        print(f'Epoch {e + 1:2d} Train Loss: {train_loss[-1]:.10f} Train Acc: {train_acc[-1]:.4f}', end=' ')
        model.eval()
        loss_sum, correct_cnt = 0, 0
        with torch.no_grad():
            for image, label in test_loader:
                predict = model(image)
                loss = criterion(predict, label)
                pred_label = torch.max(predict.data, 1).indices
                correct_cnt += (pred_label == label).sum()
                loss_sum += loss.item()
        test_loss.append(loss_sum / len(test_loader))
        test_acc.append(float(correct_cnt) / (len(test_loader) * batch_size))
        print(f'Test Loss: {test_loss[-1]:.10f} Test Acc: {test_acc[-1]:.4f}')
    return train_loss, train_acc, test_loss, test_acc


train_loss, train_acc, test_loss, test_acc = train(CNN, epochs, train_loader, test_loader)  # 訓練

# 損失函數圖表
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.plot(train_loss, label='Train Set')
plt.plot(test_loss, label='Test Set')
plt.legend()
plt.show()
# 準確率圖表
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc, label='Train Set')
plt.plot(test_acc, label='Test Set')
plt.legend()
plt.show()

visualize(CNN, test_loader)  # 結果視覺化