import math

import pandas as pd
import torch
from torch import nn

# 掩蔽 softmax
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上遮蔽元素来执行 softmax 操作"""
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = d2l.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# 点积注意力
class DotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

# 多头注意力
class MultiHeadAttention(nn.Module):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values, valid_lens):
        pass

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(
                10000,
                torch.arange(0, num_hiddens, 2, dtype=torch.float32) /
                num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)

# 基于位置的前馈网络
class PositionWiseFFN(nn.Module):
    """
    初始化基于位置的前馈网络。
    
    参数:
    ffn_num_input (int): 输入特征的维度。
    ffn_num_hiddens (int): 隐藏层的维度。
    ffn_num_outputs (int): 输出层的维度。
    """
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        # 第一个全连接层，用于将输入特征映射到隐藏层
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        # ReLU激活函数，用于引入非线性
        self.relu = nn.ReLU()
        # 第二个全连接层，用于将隐藏层特征映射到输出层
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    """
    前馈网络的前向传播。
    
    参数:
    X (Tensor): 输入特征。
    
    返回:
    Tensor: 经过前馈网络处理后的输出。
    """
    def forward(self, X):
        # 先通过第一个全连接层，再通过ReLU激活函数，最后通过第二个全连接层
        return self.dense2(self.relu(self.dense1(X)))

# 残差链接和层规范化
class AddNorm(nn.Module):
    """
    AddNorm类用于实现残差连接和层归一化的组合操作。
    
    参数:
    - normalized_shape: LayerNorm中指定的输入形状，通常是一个整数或整数元组。
    - dropout: Dropout层中使用的丢弃概率。
    """
    def __init__(self, normalized_shape, dropout, **kwargs):
        # 初始化父类nn.Module
        super(AddNorm, self).__init__(**kwargs)
        # 创建Dropout层，用于在前向传播中应用dropout
        self.dropout = nn.Dropout(dropout)
        # 创建LayerNorm层，用于在前向传播中应用层归一化
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        前向传播函数，实现残差连接和层归一化的组合操作。
        
        参数:
        - X: 输入张量，与Y经过dropout后相加。
        - Y: 输入张量，将经过dropout处理。
        
        返回:
        - 经过dropout的Y与X相加后的结果，再经过层归一化的输出张量。
        """
        # 将Y经过dropout处理后与X相加，然后进行层归一化
        return self.ln(self.dropout(Y) + X)
    
# 编码器
# 该类用于构建Transformer模型中的编码器部分，包括多头注意力机制和位置前馈网络
class EncoderBlock(nn.Module):
    # 初始化函数
    # 参数:
    # key_size: 键的特征维度
    # query_size: 查询的特征维度
    # value_size: 值的特征维度
    # num_hiddens: 隐藏层单元数
    # norm_shape: 归一化形状
    # ffn_num_input: 位置前馈网络输入维度
    # ffn_num_hiddens: 位置前馈网络隐藏层维度
    # num_heads: 多头注意力机制中的头数
    # dropout: Dropout比率
    # use_bias: 是否使用偏差项，默认为False
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        # 多头注意力机制
        self.attention = MultiHeadAttention(key_size, query_size,
                                                value_size, num_hiddens,
                                                num_heads, dropout, use_bias)
        # 添加与归一化
        self.addnorm1 = AddNorm(norm_shape, dropout)
        # 位置前馈网络
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        # 添加与归一化
        self.addnorm2 = AddNorm(norm_shape, dropout)

    # 前向传播函数
    # 参数:
    # X: 输入数据
    # valid_lens: 有效长度，用于处理填充数据
    # 返回: 经过编码器块处理后的输出
    def forward(self, X, valid_lens):
        # 注意力机制与添加归一化的综合运用
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        # 位置前馈网络与添加归一化的综合运用
        return self.addnorm2(Y, self.ffn(Y))

class TransformerEncoder(d2l.Encoder):
    """
    Transformer编码器类，继承自d2l.Encoder。
    
    参数:
    - vocab_size: 词汇表大小。
    - key_size: 键的大小。
    - query_size: 查询的大小。
    - value_size: 值的大小。
    - num_hiddens: 隐藏单元的数量。
    - norm_shape: 归一化层的形状。
    - ffn_num_input: 前馈网络输入层的大小。
    - ffn_num_hiddens: 前馈网络隐藏层的大小。
    - num_heads: 注意力头的数量。
    - num_layers: 编码器块的数量。
    - dropout: Dropout的概率。
    - use_bias: 是否使用偏差，默认为False。
    """
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                EncoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        """
        前向传播函数。
        
        参数:
        - X: 输入数据。
        - valid_lens: 有效长度。
        
        返回:
        - 编码后的数据。
        """
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
    
# 解码器
class DecoderBlock(nn.Module):
    """解码器中第 i 个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(key_size, query_size,
                                                 value_size, num_hiddens,
                                                 num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(key_size, query_size,
                                                 value_size, num_hiddens,
                                                 num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            dec_valid_lens = torch.arange(1, num_steps + 1,
                                          device=X.device).repeat(
                                              batch_size, 1)
        else:
            dec_valid_lens = None

        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module(
                "block" + str(i),
                DecoderBlock(key_size, query_size, value_size, num_hiddens,
                             norm_shape, ffn_num_input, ffn_num_hiddens,
                             num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

# 训练
num_hiddens, num_layers, dropout, batch_size, num_steps = 32, 2, 0.1, 64, 10
lr, num_epochs, device = 0.005, 200, d2l.try_gpu()
ffn_num_input, ffn_num_hiddens, num_heads = 32, 64, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)

encoder = TransformerEncoder(len(src_vocab), key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, num_layers, dropout)
decoder = TransformerDecoder(len(tgt_vocab), key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, num_layers, dropout)
net = d2l.EncoderDecoder(encoder, decoder)
d2l.train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)