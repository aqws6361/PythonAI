# 匯入套件
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.pylab import plt

# 超參數
batch_size = 100
lr = 0.003
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
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv3 = nn.Conv2d(16, 100, 3)
        self.pool = nn.MaxPool2d(2)
        self.bnc1 = nn.BatchNorm2d(6)
        self.bnc2 = nn.BatchNorm2d(16)
        self.bnc3 = nn.BatchNorm2d(100)
        self.fc1 = nn.Linear(100 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_size)
        self.bnf1 = nn.BatchNorm1d(120)
        self.bnf2 = nn.BatchNorm1d(84)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 卷積與池化
        x = self.conv1(x)  # (3, 32, 32) => (6, 28, 28)
        x = self.bnc1(x)
        x = self.relu(x)
        x = self.pool(x)  # (6, 28, 28) => (6, 14, 14)
        x = self.conv2(x)  # (6, 14, 14) => (16, 12, 12)
        x = self.bnc2(x)
        x = self.relu(x)
        x = self.pool(x)  # (16, 12, 12) => (16, 6, 6)
        x = self.conv3(x)  # (16, 6, 6) => (100, 4, 4)
        x = self.bnc3(x)
        x = self.relu(x)
        x = self.pool(x)  # (100, 4, 4) => (100, 2, 2)
        # 四維張量轉二維張量
        x = x.view(x.shape[0], -1)  # (100, 2, 2) => (400)
        # 全連接層
        x = self.fc1(x)  # (400) => (120)
        x = self.bnf1(x)
        x = self.relu(x)
        x = self.fc2(x)  # (120) => (84)
        x = self.bnf2(x)
        x = self.relu(x)
        x = self.fc3(x)  # (84) => (10)
        return x


# 宣告模型
CNN = CNN_CIFAR10(3, 10)
criterion = nn.CrossEntropyLoss()
optimzer = optim.Adam(CNN.parameters(), lr=lr)
print(CNN)

device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 選擇運算硬體
CNN = CNN.to(device)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def visualize(model, data_loader):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    model.eval()
    print(f'Labels: {" ".join(f"{classes[label]:>5s}" for label in labels[: 5])}')
    predict = model(images.to(device)).cpu()  # 把訓練資料送到GPU上，再把預測機率送回主記憶體
    pred_label = torch.max(predict.data, 1).indices
    print(f'Predictions: {" ".join(f"{classes[label]:>5s}" for label in pred_label[: 5])}')
    torch.cuda.empty_cache()  # 清空GPU記憶體快取
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(13, 8)
    plt.subplots_adjust(wspace=1, hspace=0.1)
    for i in range(5):
        axes[0][i].imshow((images[i] * 0.5 + 0.5).permute(1, 2, 0).numpy().squeeze())
        x = list(range(10))
        y = torch.softmax(predict.data[i], 0)
        axes[1][i].barh(x, y)
        for j, v in enumerate(y):
            axes[1][i].text(1.1 * max(y), j - 0.1, str("{:1.4f}".format(v)), color='black')
    plt.show()


visualize(CNN, train_loader)


def train(model, epochs, train_loader, test_loader):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for e in range(epochs):
        model.train()
        loss_sum, correct_cnt = 0, 0
        for image, label in train_loader:
            optimzer.zero_grad()
            predict = model(image.to(device)).cpu()
            loss = criterion(predict, label)
            pred_label = torch.max(predict.data, 1).indices
            correct_cnt += (pred_label == label).sum()
            loss_sum += loss.item()
            loss.backward()
            optimzer.step()
        train_loss.append(loss_sum / len(train_loader))
        train_acc.append(float(correct_cnt) / (len(train_loader) * batch_size))
        print(f'Epoch {e + 1:2d} Train Loss: {train_loss[-1]:.10f} Train Acc: {train_acc[-1]:.4f}', end=' ')
        model.eval()
        loss_sum, correct_cnt = 0, 0
        with torch.no_grad():
            for image, label in test_loader:
                predict = model(image.to(device)).cpu()
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
