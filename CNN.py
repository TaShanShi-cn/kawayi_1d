import h5py
import numpy
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import sys


# 1.构建数据集
def create_dataset():
    # 读取数据集
    data=h5py.File("D:/Dataset/auto_sliced_dataset/SNR_30.hdf5")
    X = data['X']  # (2555904,1024,2),<class 'h5py.hl.dataset.Dataset'>
    # Y 2555904行，24列，列代表调制方式，每一行元素只有一个为1，代表是何种调制方式，其余为0
    Y = data['Y']  # (2555904,24),<class 'h5py.hl.dataset.Dataset'>
    #将hdf5的dataset转换威numpy
    x = np.array(X)
    y = np.array(Y)

    # 将特征值和目标值拆分
    # x表示特征值，y表示目标值
    # 转换类型
    x = x.astype(np.float32)
    y = y.astype(np.int64)
    # 数据集划分为训练集与测试集
    x_train, x_vaild, y_train, y_vaild = \
        train_test_split(x, y, test_size=0.3, random_state=88, stratify=y)  # random_state用于固定随机划分的结果
    # stratify分层采样，保证划分的数据中每个类别的数据都有

    # 数据标准化
    #transfer=StandardScaler()
    #x_train=transfer.fit_transform(x_train)
    #x_vaild=transfer.transform(x_vaild)

    # 构建Pytorch数据集对象
    train_dataset = TensorDataset(torch.from_numpy(x_train), torch.tensor(y_train))
    vaild_dataset = TensorDataset(torch.from_numpy(x_vaild), torch.tensor(y_vaild))
    # 返回数据：训练集对象，测试集对象，特征维度，类别数量
    return train_dataset, vaild_dataset, x_train.shape[1], 24


train_dataset, vaild_dataset, input_dim, output_dim = create_dataset()



# 2.构建分类网络模型

class AMC(nn.Module):  # 自定义的类要继承nn.Moudle

    def __init__(self):
        super(AMC, self).__init__()  # 调用父类的初始化函数
        # 定义卷积池化层
        self.conv1=nn.Conv2d(2,128,stride=1,kernel_size=(3,1),padding=0)
        self.pool1=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#输出(128,1022)

        self.conv2=nn.Conv2d(128,256,stride=2,kernel_size=(3,1),padding=0)
        self.pool2=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#输出(256,510)

        self.conv3=nn.Conv2d(256,512,stride=3,kernel_size=(3,1),padding=0)
        self.pool3=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#(512,170)

        self.conv4=nn.Conv2d(512,256,stride=2,kernel_size=(4,1),padding=0)
        self.pool4=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#(256,84)

        self.conv5=nn.Conv2d(256,128,stride=1,kernel_size=(5,1),padding=0)
        self.pool5=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#(128,80)

        self.conv6=nn.Conv2d(128,64,stride=1,kernel_size=(20,1),padding=0)
        self.pool6=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#(64,61)

        self.conv7=nn.Conv2d(64,64,stride=1,kernel_size=(10,1),padding=0)
        self.pool7=nn.MaxPool2d(kernel_size=3, stride=1,padding=1)#(64,52)

        self.linear1=nn.Linear(3328,24)


    def activation(self, x):
        return torch.relu(x)

    def forward(self, x):  # 构建前向计算的函数

        x=self.conv1(x)
        x=F.relu(x)
        x=self.pool1(x)
        #print(x.shape)

        x=self.conv2(x)
        x=F.relu(x)
        x=self.pool2(x)
        #print(x.shape)

        x=self.conv3(x)
        x=F.relu(x)
        x=self.pool3(x)
        #print(x.shape)

        x=self.conv4(x)
        x=F.relu(x)
        x=self.pool4(x)
        #print(x.shape)

        x=self.conv5(x)
        x=F.relu(x)
        x=self.pool5(x)
        #print(x.shape)

        x=self.conv6(x)
        x=F.relu(x)
        x=self.pool6(x)
        #print(x.shape)

        x=self.conv7(x)
        x=F.relu(x)
        x=self.pool7(x)
        #print(x.shape)

        #将特征图送入全连接层，此时需要进行维度的变换
        x=x.reshape(x.size(0),-1)

        output=self.linear1(x)


        return output


# 3.编写训练函数
# 将数据送入网络，进行正向传播计算，反向传播与更新网络参数
# 使用多分类交叉熵损失函数，使用SGD优化方法

def train():
    # 固定随机数种子
    torch.manual_seed(0)

    # 初始化网络模型
    model = AMC().cuda()
    #model.load_state_dict(torch.load('D:/Dataset/Experiment/trained_network/AMC_28.pth'))
    # 损失函数,使用交叉熵损失函数会首先对数据进行Softmax,再进行交叉熵损失计算
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001,betas=(0.9,0.99))
    # 训练轮数,表示将所有的训练数据完全送入到网络中多少次
    num_epochs = 10

    for epoch_idx in range(num_epochs):
        # 初始化数据加载器
        dataloader = DataLoader(dataset=train_dataset, shuffle=True, batch_size=128,drop_last=False)

        # 训练时间
        start = time.time()
        # 计算损失
        total_loss = 0.0
        total_num = 0
        # 准确率
        correct = 0

        for x, y in dataloader:
            # 将数据送入网络
            x = x.cuda()
            y = y.cuda()
            x=x.unsqueeze(1)
            x = x.permute(0, 3, 2, 1)
            output = model(x)
            # 计算损失
            loss = criterion(output.float(), y.float())
            # 梯度清零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()

            # 累计总样本数量
            total_num += len(y)
            # 累计总损失
            total_loss += loss.item() * len(y)

            y_pred = torch.argmax(output, dim=-1)
            correct += sum(y_pred == torch.argmax(y,dim=-1))

        print('epoch:%4s loss:%.4f time: %.2fs acc:%.4f' % (epoch_idx + 1, total_loss / total_num, time.time() - start,correct/total_num))

    # 模型保存
    torch.save(model.state_dict(), 'D:/Dataset/Experiment/trained_network/AMC_30.pth')


# 4.编写评估函数
# 使用训练好的模型，对未知的样本进行预测的过程
def test():
    # 4.1.加载模型
    model = AMC().cuda()
    # 将参数初始化为已经训练好的模型参数
    model.load_state_dict(torch.load('D:/Dataset/Experiment/trained_network/AMC_30.pth'))

    # 4.2.构建测试集数据加载器
    dataloader = DataLoader(vaild_dataset, batch_size=128, shuffle=False)

    # 4.3.计算在测试集上的准确率
    correct = 0
    for x, y in dataloader:
        # 将数据送入网络
        x=x.cuda()
        x = x.unsqueeze(1)
        x = x.permute(0, 3, 2, 1)
        y=y.cuda()
        test_output = model(x)
        # 得到预测标签
        test_y_pred = torch.argmax(test_output, dim=-1)
        correct += (sum(test_y_pred == torch.argmax(y,dim=-1)))
    print('acc:%.5f' % (correct / len(vaild_dataset)))

if __name__ == '__main__':
    train()
    test()
