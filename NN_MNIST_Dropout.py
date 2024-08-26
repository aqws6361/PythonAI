# 匯入套件
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib.pylab import plt


# 建構有Dropout的神經網路類別
class Network(nn.Module):
    def __init__(self, input_size, output_size, p):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p)
        self.fc3 = nn.Linear(128, output_size)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


# 超參數
batch_size = 64
lr = 0.001
epochs = 10

# 宣告模型
NN = Network(784, 10, 0.2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(NN.parameters(), lr=lr)

train_set = datasets.MNIST('./', download=True, train=True, transform=transforms.ToTensor())
test_set = datasets.MNIST('./', download=True, train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


def visualize(model, data_loader):
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    model.eval()
    print(f'Labels: {" ".join(str(int(label)) for label in labels[: 5])}')
    predict = model(images.view(images.shape[0], -1))
    pred_label = torch.max(predict.data, 1).indices
    print(f'Predictions: {" ".join(str(int(label)) for label in pred_label[: 5])}')
    fig, axes = plt.subplots(nrows=2, ncols=5)
    fig.set_size_inches(13, 8)
    plt.subplots_adjust(wspace=1, hspace=0.1)
    for i in range(5):
        axes[0][i].imshow(images[i].permute(1, 2, 0).numpy().squeeze(), cmap=plt.cm.gray)
        x = list(range(10))
        y = torch.softmax(predict.data[i], 0)
        axes[1][i].barh(x, y)
        for j, v in enumerate(y):
            axes[1][i].text(1.1 * max(y), j - 0.1, str("{:1.4f}".format(v)), color='black')
    plt.show()


visualize(NN, train_loader)


def train(model, epochs, train_loader, test_loader):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for e in range(epochs):
        model.train()
        loss_sum, correct_cnt = 0, 0
        for image, label in train_loader:
            optimizer.zero_grad()
            predict = model(image.view(image.shape[0], -1))
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
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.plot(train_loss, label='Train Set')
plt.plot(test_loss, label='Test Set')
plt.legend()
plt.show()
# 預測準確率圖表
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_acc, label='Train Set')
plt.plot(test_acc, label='Test Set')
plt.legend()
plt.show()

visualize(NN, test_loader)  # 結果視覺化
