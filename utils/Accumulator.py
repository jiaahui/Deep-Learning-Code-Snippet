class Accumulator:
    """
    在`n`个变量上累加。
    
    该类提供了一个简便的方式在多个变量上进行累加操作，适用于需要在程序中
    累加多个不同类型数据的场景。
    """
    def __init__(self, n):
        """
        初始化累加器，包含`n`个变量。
        
        参数:
        - n: 需要累加的变量的数量。
        """
        # 初始化数据列表，长度为n，元素均为0.0
        self.data = [0.0] * n

    def add(self, *args):
        """
        将给定的值分别加到每个变量上。
        
        参数:
        - *args: 一个或多个要累加到变量上的值，每个值对应一个变量。
        """
        # 将当前数据与传入的参数相加，支持浮点数累加
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        """
        重置所有变量的值为0。
        """
        # 重置数据列表，将其所有元素设为0.0
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        """
        获取指定变量的值。
        
        参数:
        - idx: 变量的索引。
        
        返回:
        - 指定索引处的变量值。
        """
        # 返回指定索引的累加值
        return self.data[idx]