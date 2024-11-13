import torch
from torch import nn
from torch.utils import data

# 生成数据集
def synthetic_data(w, b, num_examples):
    """
    生成y=Xw+b+噪声

    参数:
    w -- 权重向量
    b -- 偏置项
    num_examples -- 生成样本数量

    返回:
    X -- 输入特征矩阵
    y -- 添加了噪声的输出向量
    """
    # 生成均值为0，标准差为1的正态分布随机数矩阵X，大小为num_examples行，w的长度列
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 计算y，其中y=Xw+b，使用矩阵乘法计算
    y = torch.matmul(X, w) + b
    # 给y添加均值为0，标准差为0.01的正态分布噪声
    y += torch.normal(0, 0.01, y.shape)
    # 返回特征矩阵X和添加了噪声的输出向量y，y被重塑为(-1, 1)的形式
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 加载数据集
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """构造一个PyTorch数据迭代器

    参数:
    data_arrays (tuple): 包含输入特征和标签的数组元组。
    batch_size (int): 每个批次的数据大小。
    is_train (bool): 是否为训练数据。如果是训练数据，数据迭代器将会打乱数据，默认为True。

    返回:
    DataLoader: PyTorch的数据迭代器，用于在训练或测试过程中逐批次获取数据。
    """
    # 将数据数组转换为PyTorch数据集对象
    dataset = data.TensorDataset(*data_arrays)
    # 返回数据加载器对象，如果is_train为True，数据将会在每个epoch中被打乱
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
net = nn.Sequential(nn.Linear(2, 1))

# 初始化模型参数
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

# 定义损失函数
loss = nn.MSELoss()

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# 验证
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
