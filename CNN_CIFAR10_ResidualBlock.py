import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.pylab import plt

batch_size = 16
lr = 0.003
epochs = 10

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = datasets.CIFAR10('./', download=True, train=True, transform=transform)
test_set = datasets.CIFAR10('./', download=True, train=False, transform=transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


# 建構Sequential
def block(channels, kernel_size):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size, padding=padding),
        nn.BatchNorm2d(channels),
        nn.Dropout(0.1),
        nn.Conv2d(channels, channels, kernel_size, padding=padding),
        nn.BatchNorm2d(channels),
        nn.Dropout(0.1)
    )


# 建構CNN類別
class CNN_CIFAR10(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv = nn.Conv2d(input_size, 32, 1)
        self.res1 = block(32, 7)
        self.res2 = block(64, 5)
        self.res3 = block(128, 3)
        self.res4 = block(256, 3)
        self.res5 = block(512, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(512, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        # 卷積與池化
        x = self.conv(x)
        x = self.relu(self.res1(x) + x)
        x = self.pool2(x)
        x = x.repeat(1, 2, 1, 1)
        x = self.relu(self.res2(x) + x)
        x = self.pool2(x)
        x = x.repeat(1, 2, 1, 1)
        x = self.relu(self.res3(x) + x)
        x = self.pool2(x)
        x = x.repeat(1, 2, 1, 1)
        x = self.relu(self.res4(x) + x)
        x = self.pool2(x)
        x = x.repeat(1, 2, 1, 1)
        x = self.relu(self.res5(x) + x)
        x = self.pool2(x)
        # 四維張量轉二維張量
        x = x.view(batch_size, -1)
        # 全連接層
        x = self.fc(x)
        return x


device = "cuda:0" if torch.cuda.is_available() else "cpu"  # 選擇運算處理器

CNN = CNN_CIFAR10(3, 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(CNN.parameters(), lr=lr)
print(CNN)

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
            optimizer.zero_grad()
            predict = model(image.to(device)).cpu()  # 把訓練資料送到GPU上，再把預測機率送回主記憶體
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
