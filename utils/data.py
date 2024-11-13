import torch
from torch.utils import data
from torchvision import transforms


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

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器

    参数:
    data_arrays: 包含数据和标签的数组，准备加载到数据迭代器中。
    batch_size: 每个批次的数据大小。
    is_train: 布尔值，指示是否在训练模式下加载数据。默认为True。

    返回:
    DataLoader: 一个PyTorch DataLoader实例，用于在训练或测试过程中迭代数据集。
    """
    # 创建一个TensorDataset实例，将数据和标签整合到一个数据集中
    dataset = data.TensorDataset(*data_arrays)
    # 使用DataLoader加载数据集，指定批次大小和是否打乱数据
    return data.DataLoader(dataset, batch_size, shuffle=is_train)
