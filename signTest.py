import os
import torch
import torchvision
from torchvision import transforms
from PIL import Image
from torch import nn
i=0 #识别图片计数
root_path="./model/test"         #待测试文件夹
names=os.listdir(root_path)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32*4*4, 100),
            nn.ReLU(),
            nn.Linear(100, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32*4*4)
        x = self.fc(x)
        # x = nn.functional.normalize(x)
        return x
    

for name in names:
    print(name)
    i=i+1
    data_class=['left','pause','right','straight']   #按文件索引顺序排列
    image_path=os.path.join(root_path,name)             
    image=Image.open(image_path)
    print(image)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
        transforms.Resize((28, 28)),                 # 调整图像大小为 28x28
        transforms.ToTensor(),                       # 将图像转换为张量，值缩放到 [0, 1]
        transforms.Normalize((0.5,), (0.5,))         # 归一化到 [-1, 1]，均值和标准差为 0.5
    ])
    image=transform(image)
    print(image.shape)

    model_ft=Net()     #需要使用训练时的相同模型
    # print(model_ft)

    model=torch.load("best_model.pth",map_location=torch.device("cpu")) #选择训练后得到的模型文件
    # print(model)
    image=torch.reshape(image,(1,1,28,28))      #修改待预测图片尺寸，需要与训练时一致
    model.eval()
    with torch.no_grad():
        output=model(image)
    print(output)               #输出预测结果
    # print(int(output.argmax(1)))
    print("第{}张图片预测为：{}".format(i,data_class[int(output.argmax(1))]))   #对结果进行处理，使直接显示出预测的种类





