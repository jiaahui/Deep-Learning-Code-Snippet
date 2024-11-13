from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline


def use_svg_display():
    """
    使用svg格式在Jupyter中显示绘图
    
    该函数通过设置matplotlib的格式为svg，使得在Jupyter notebook中显示的图形更加清晰和美观
    """
    backend_inline.set_matplotlib_formats('svg')

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

class Animator:
    """在动画中绘制数据。"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes,]
        self.config_axes = lambda: set_axes(self.axes[
            0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)
