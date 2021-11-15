import torch
from torch import nn
import torchvision
from torch.utils.data import DataLoader


#  获取数据
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10("./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
#  数据长度
len_train_data = len(train_data)
len_test_data = len(test_data)

#  加载数据
train_dataloder = DataLoader(train_data, batch_size=64)
test_dataloder = DataLoader(test_data, batch_size=64)


#  构建模型
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )
    def forward(self,x):
        output = self.model(x)
        return output

#  模型实例化
model = MyNet()
model = model.cuda()

#  建立损失函数
loss_fc = nn.CrossEntropyLoss()
loss_fc = loss_fc.cuda()

#  建立优化器
optim = torch.optim.SGD(model.parameters(),lr=0.01)

#  准备参数
epoch = 50  # 训练轮数
train_step = 0

# tensorboard
writer = SummaryWriter("logs_train")

#  开始训练
for i in range(epoch):
    print("--------第{}轮训练开始----------".format(i+1))
    for data in train_dataloder:
        train_step += 1
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        input = model(imgs)
        loss_train = loss_fc(input, targets)
        if train_step % 100 == 0:
            print("训练次数：{} ||loss:{}".format(train_step, loss_train))
        optim.zero_grad()
        loss_train.backward()
        optim.step()
    loss_sum = 0.0
    accuracy_sum = 0
    with torch.no_grad():
        for data in test_dataloder:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            input = model(imgs)
            loss_test = loss_fc(input, targets)
            loss_sum += loss_test
            accuracy = (torch.argmax(input, 1) == targets).sum()
            accuracy_sum += accuracy
        writer.add_scalar("accuracy", accuracy_sum/len_test_data, i+1)
        print("第{}轮训练精度为：{}".format(i+1, accuracy_sum/len_test_data))
torch.save(model, "model_{}.pth".format(i+1))
print("模型保存完成")

writer.close()




