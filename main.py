import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

# 加载MNIST数据
def load_data():
    # 将[0,1]的PILImage归一化为[-1,1]的Tensor
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True, num_workers=0)

    test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=False, num_workers=0)

    return train_loader, test_loader

# 创建神经网络
class Net(torch.nn.Module):
    # 定义神经网络结构 LeNet-5
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)     # 卷积层1
        self.pool = torch.nn.MaxPool2d(2, 2)      # 池化层
        self.conv2 = torch.nn.Conv2d(6, 16, 5)    # 卷积层2
        self.fc1 = torch.nn.Linear(16*5*5, 120)   # 全连接层1
        self.fc2 = torch.nn.Linear(120, 84)       # 全连接层2
        self.fc3 = torch.nn.Linear(84, 10)        # 全连接层3

    # 连接关系
    def forward(self, x):
        x = F.relu(self.conv1(x))     # 卷积层1接激活
        x = self.pool(x)              # 接池化层
        x = F.relu(self.conv2(x))     # 接卷积层2 接激活
        x = self.pool(x)              # 接池化层
        x = x.view(-1, 16*5*5)        # 改变tensor的size，准备进入全连接层
        x = F.relu(self.fc1(x))       # 全连接1
        x = F.relu(self.fc2(x))       # 全连接2
        x = self.fc3(x)               # 全连接3

        return x

learning_rate = 0.001
n_epochs = 3
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

'''------------训练-------------'''
# 加载数据
train_loader, test_loader = load_data()

# 创建神经网络
model = Net()
# 尝试在gpu上跑 但是由于cuda的版本一直没配好所以还是先用CPU跑了
print(torch.cuda.is_available())
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, start=0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)   # 转为Variable，可以反向传播
        optimizer.zero_grad()              # 梯度归零
        outputs = model(inputs)            # 传入神经网络,记录输出值 正向传播
        loss = criterion(outputs, labels)  # 计算误差
        loss.backward()                    # 反向传播
        optimizer.step()                   # 更新参数

        # 打印观察loss  每2000打印一次
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/2000))
            running_loss = 0.0

print('Finished Training！')

'''------------测试-------------'''
correct = 0      # 正确个数
total = 0        # 总数
for data in test_loader:
    images, labels = data
    outputs = model(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy on the test images: %d %%' % (100*correct/total))


