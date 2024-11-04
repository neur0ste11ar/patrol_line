import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.models as models
import torch.nn as nn
import torch

print("是否使用GPU训练：{}".format(torch.cuda.is_available()))    #打印是否采用gpu训练
if torch.cuda.is_available:
    print("GPU名称为：{}".format(torch.cuda.get_device_name()))  #打印相应的gpu信息
#数据增强太多也可能造成训练出不好的结果，而且耗时长，宜增强两三倍即可。
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为单通道灰度图
    transforms.Resize((28, 28)),                 # 调整图像大小为 28x28
    transforms.ToTensor(),                       # 将图像转换为张量，值缩放到 [0, 1]
    transforms.Normalize((0.5,), (0.5,))         # 归一化到 [-1, 1]，均值和标准差为 0.5
])
dataset_train=ImageFolder('./model/data/train',transform=transform)     #训练数据集
# print(dataset_tran[0])
dataset_valid=ImageFolder('./model/data/val',transform=transform)     #验证或测试数据集
# print(dataset_train.classer)#返回类别
print(dataset_train.class_to_idx)                               #返回类别及其索引
# print(dataset_train.imgs)#返回图片路径
print(dataset_valid.class_to_idx)
train_data_size=len(dataset_train)                              #放回数据集长度
test_data_size=len(dataset_valid)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
#torch自带的标准数据集加载函数
dataloader_train=DataLoader(dataset_train,batch_size=4,shuffle=True,num_workers=0,drop_last=True)
dataloader_test=DataLoader(dataset_valid,batch_size=4,shuffle=True,num_workers=0,drop_last=True)

# Define model
# 定义model
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



#2.模型加载
model_ft=Net()#使用迁移学习，加载预训练权重
# print(model_ft)

#冻结卷积层函数
# for i,para in enumerate(model_ft.parameters()):
#     if i<18:
#         para.requires_grad=False

# print(model_ft)


# model_ft.half()#可改为半精度，加快训练速度，在这里不适用

model_ft=model_ft.cuda()#将模型迁移到gpu
#3.优化器
loss_fn=nn.CrossEntropyLoss()

loss_fn=loss_fn.cuda()  #将loss迁移到gpu
learn_rate=0.01         #设置学习率
optimizer=torch.optim.SGD(model_ft.parameters(),lr=learn_rate,momentum=0.01)#可调超参数

total_train_step=0
total_test_step=0
epoch=20                #迭代次数
writer=SummaryWriter("logs_train")
best_acc=-1
ss_time=time.time()

for i in range(epoch):
    start_time = time.time()
    print("--------第{}轮训练开始---------".format(i+1))
    model_ft.train()
    for data in dataloader_train:
        imgs,targets=data
        # if torch.cuda.is_available():
        # imgs.float()
        # imgs=imgs.float()#为上述改为半精度操作，在这里不适用
        imgs=imgs.cuda()
        targets=targets.cuda()
        # imgs=imgs.half()
        outputs=model_ft(imgs)
        loss=loss_fn(outputs,targets)

        optimizer.zero_grad()   #梯度归零
        loss.backward()         #反向传播计算梯度
        optimizer.step()        #梯度优化

        total_train_step=total_train_step+1
        if total_train_step%100==0:#一轮时间过长可以考虑加一个
            end_time=time.time()
            print("使用GPU训练100次的时间为：{}".format(end_time-start_time))
            print("训练次数：{},loss:{}".format(total_train_step,loss.item()))
            # writer.add_scalar("valid_loss",loss.item(),total_train_step)
    model_ft.eval()
    total_test_loss=0
    total_accuracy=0
    with torch.no_grad():       #验证数据集时禁止反向传播优化权重
        for data in dataloader_test:
            imgs,targets=data
            # if torch.cuda.is_available():
            # imgs.float()
            # imgs=imgs.float()
            imgs = imgs.cuda()
            targets = targets.cuda()
            # imgs=imgs.half()
            outputs=model_ft(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
            accuracy=(outputs.argmax(1)==targets).sum()
            total_accuracy=total_accuracy+accuracy
        print("整体测试集上的loss：{}(越小越好,与上面的loss无关此为测试集的总loss)".format(total_test_loss))
        print("整体测试集上的正确率：{}(越大越好)".format(total_accuracy / len(dataset_valid)))

        writer.add_scalar("valid_loss",(total_accuracy/len(dataset_valid)),(i+1))#选择性使用哪一个
        total_test_step = total_test_step + 1
        if total_accuracy > best_acc:   #保存迭代次数中最好的模型
            print("已修改模型")
            best_acc = total_accuracy
            torch.save(model_ft, "best_model.pth")
ee_time=time.time()
zong_time=ee_time-ss_time
print("训练总共用时:{}h:{}m:{}s".format(int(zong_time//3600),int((zong_time%3600)//60),int(zong_time%60))) #打印训练总耗时
writer.close()

