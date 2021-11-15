import torch
import torchvision
from PIL import Image
from torch import nn
import matplotlib.pyplot as plt

# 输入图片
img_path = input("please input img_name from imgs:\n")
img = Image.open('imgs/'+img_path)

# 改变图片格式
trans = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)), torchvision.transforms.ToTensor()])
img_trans = trans(img)


# 网络结构
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

# 进行预测
# myNet = torch.load("model_50.pth",map_location=torch.device('cpu')) # 使用cpu进行计算
myNet = torch.load("model_50.pth")
myNet = myNet.cuda()
img_trans = torch.reshape(img_trans, (1, 3, 32, 32))
myNet.eval()
img_trans = img_trans.cuda()
with torch.no_grad():
    input = myNet(img_trans)
target = torch.argmax(input, dim=1)
target = target[0]

# 判断类别
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.imshow(img)
plt.show()
print("This is a/an "+classes[target])
