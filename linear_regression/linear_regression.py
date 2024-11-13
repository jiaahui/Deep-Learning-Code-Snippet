import random
import torch

# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声
    
    参数:
    w -- 权重向量
    b -- 偏置项
    num_examples -- 生成样本数量
    
    返回:
    X -- 特征矩阵
    y -- 目标值向量
    """
    X = torch.normal(0, 1, (num_examples, len(w)))  # 标准正态分布采样 X
    # @ 矩阵乘法计算 如果其中一个参数是向量的话 自动转化为列向量 无需手动转换
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape) # 增加高斯噪音
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 加载数据集
def data_iter(batch_size, features, labels):
    """
    创建一个数据迭代器，用于随机返回一批特征和标签。
    
    参数:
    batch_size (int): 每个批次的数据大小。
    features (Tensor): 包含所有特征的数据张量。
    labels (Tensor): 包含所有标签的数据张量。
    
    生成:
    Tensor, Tensor: 一个批次的特征和对应的标签。
    """
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# 初始化模型参数
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)	# 正态分布
b = torch.zeros(1, requires_grad=True)

# 定义模型
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b

# 定义损失函数
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 定义优化算法
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

# 训练
batch_size = 10
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# 验证
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
