from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


def use_svg_display():
    """
    使用svg格式在Jupyter中显示绘图
    
    该函数通过设置matplotlib的格式为svg，使得在Jupyter notebook中显示的图形更加清晰和美观
    """
    backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    """
    设置matplotlib的图标大小
    
    此函数通过修改matplotlib的配置参数来调整图表的大小，使其更适合展示或出版需求
    
    参数:
    figsize (tuple): 一个包含两个浮点数的元组，表示图表的宽度和高度，默认值为(3.5, 2.5)
    
    返回值:
    无
    """
    # 使用SVG格式显示图表，确保图表质量和兼容性
    use_svg_display()
    # 设置图表的默认大小
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """
    设置matplotlib的轴

    参数:
    - axes: matplotlib的Axes对象，用于设置图形的轴
    - xlabel: x轴的标签
    - ylabel: y轴的标签
    - xlim: x轴的范围
    - ylim: y轴的范围
    - xscale: x轴的比例类型，例如'linear'、'log'
    - yscale: y轴的比例类型，例如'linear'、'log'
    - legend: 图例的列表，如果不需要图例，则为None
    """
    # 设置x轴标签
    axes.set_xlabel(xlabel)
    # 设置y轴标签
    axes.set_ylabel(ylabel)
    # 设置x轴比例
    axes.set_xscale(xscale)
    # 设置y轴比例
    axes.set_yscale(yscale)
    # 设置x轴范围
    axes.set_xlim(xlim)
    # 设置y轴范围
    axes.set_ylim(ylim)
    # 如果提供了图例，则添加图例
    if legend:
        axes.legend(legend)
    # 添加网格线
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点

    参数:
    X: 数据点的X轴坐标，可以是一维或二维列表/数组
    Y: 数据点的Y轴坐标，如果为None，则使用X作为Y轴坐标，X轴坐标为默认序列
    xlabel: X轴标签
    ylabel: Y轴标签
    legend: 图例
    xlim: X轴范围
    ylim: Y轴范围
    xscale: X轴刻度模式，默认为线性
    yscale: Y轴刻度模式，默认为线性
    fmts: 绘制格式字符串列表，默认为实线、虚线等不同样式的线条
    figsize: 图形大小，默认为3.5x2.5英寸
    axes: 绘制轴对象，如果为None，则使用当前轴
    """
    # 如果未提供图例，初始化图例为空列表
    if legend is None:
        legend = []

    # 设置图形大小
    set_figsize(figsize)
    # 使用提供的轴对象或当前轴对象
    axes = axes if axes else plt.gca()

    # 定义一个内部函数来判断输入是否为单轴数据
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list) and not hasattr(X[0], "__len__"))

    # 根据输入数据的维度进行相应的处理，以确保X和Y是可迭代的轴数据
    if has_one_axis(X):
        X = X[0]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    # 清除当前轴的绘图内容，为新绘图做准备
    axes.cla()
    # 遍历每个数据序列和对应的格式，绘制数据点
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    # 设置轴的属性，包括标签、范围、刻度模式和图例
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
