import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, num_groups):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              groups=num_groups)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SelfAttentionWithConv(nn.Module):
    def __init__(self, in_channels, num_groups, num_channels):
        super(SelfAttentionWithConv, self).__init__()
        self.conv1 = ConvBlock(in_channels, in_channels, kernel_size=3, stride=1, padding=1, num_groups=num_groups)
        self.query = nn.Linear(in_channels, num_channels)
        self.key = nn.Linear(in_channels, num_channels)
        self.value = nn.Linear(in_channels, num_channels)
        self.conv2 = nn.Conv1d(num_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 进行组卷积特征提取
        x = x.permute(0, 2, 1)  # 调整输入形状为(batch_size, in_channels, sequence_length)
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # 恢复原来的形状

        # 计算查询、键和值
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # 计算注意力权重矩阵
        attention_weights = torch.matmul(q, k.transpose(-1, -2))
        attention_weights = F.softmax(attention_weights, dim=-1)

        # 插入通道注意力计算矩阵
        channel_attention_weights = torch.sigmoid(self.conv2(attention_weights))
        attention_weights = attention_weights * channel_attention_weights

        # 根据注意力权重对值进行加权求和
        output = torch.matmul(attention_weights, v)

        return output


# 示例输入特征张量的形状：(batch_size, sequence_length, in_channels)
input_features = torch.randn(16, 128, 64)

# 创建SelfAttentionWithConv模型实例
self_attention = SelfAttentionWithConv(in_channels=128, num_groups=8, num_channels=32)

# 前向传播
output = self_attention(input_features)

print("自注意力输出的形状：", output.shape)
