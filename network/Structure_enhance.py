import torch
import torch.nn as nn

import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    # 获取各张量的尺寸
    sizes = [x1.shape[-2:], x2.shape[-2:], x3.shape[-2:], x4.shape[-2:]]
    max_size = tuple(max(dim) for dim in zip(*sizes))  # 计算最大尺寸

    # 对尺寸较小的张量进行填充
    x1 = F.pad(x1, (0, max_size[1] - x1.shape[-1], 0, max_size[0] - x1.shape[-2]))
    x2 = F.pad(x2, (0, max_size[1] - x2.shape[-1], 0, max_size[0] - x2.shape[-2]))
    x3 = F.pad(x3, (0, max_size[1] - x3.shape[-1], 0, max_size[0] - x3.shape[-2]))
    x4 = F.pad(x4, (0, max_size[1] - x4.shape[-1], 0, max_size[0] - x4.shape[-2]))

    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h
def iwt_init(x):
    r = 2
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(in_channel / (r ** 2)), r * in_height, r * in_width

    # 确保输入通道数是4的倍数
    if in_channel % 4 != 0:
        raise ValueError("Input channel number must be a multiple of 4")

    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2

    # 获取各张量的尺寸
    sizes = [x1.shape[-2:], x2.shape[-2:], x3.shape[-2:], x4.shape[-2:]]
    max_size = tuple(max(dim) for dim in zip(*sizes))  # 计算最大尺寸

    # 对尺寸较小的张量进行填充
    x1 = F.pad(x1, (0, max_size[1] - x1.shape[-1], 0, max_size[0] - x1.shape[-2]))
    x2 = F.pad(x2, (0, max_size[1] - x2.shape[-1], 0, max_size[0] - x2.shape[-2]))
    x3 = F.pad(x3, (0, max_size[1] - x3.shape[-1], 0, max_size[0] - x3.shape[-2]))
    x4 = F.pad(x4, (0, max_size[1] - x4.shape[-1], 0, max_size[0] - x4.shape[-2]))

    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)

    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)


# class StructureEnhenceEWTBlock(nn.Module):
#     def __init__(self):
#         super(StructureEnhenceEWTBlock, self).__init__()
#         self.dwt_module = DWT()
#         self.iwt_module = IWT()
#         self.model = TransformerBlock(dim=4, num_heads=4, ffn_expansion_factor=1.,
#                             bias=True, LayerNorm_type='WithBias')
#
#     def forward(self, P_before):
#         x1 = self.dwt_module(P_before)
#         # x2 = self.dwt_module(Svis)
#         # x = (x1 + x2)/2
#         residual = x1
#         x = self.model(x1)
#         # x = self.model(x)
#         x = x + residual
#         x = self.iwt_module(x)
#         return x

class StructureEnhenceEWTBlock(nn.Module):
    def __init__(self):
        super(StructureEnhenceEWTBlock, self).__init__()
        self.kernel = torch.ones(1, 1, 3, 3) / (3 ** 2)
        self.model = TransformerBlock(dim=4, num_heads=4, ffn_expansion_factor=1.,
                                      bias=True, LayerNorm_type='WithBias')

    def forward(self, P_before):
        residual = P_before
        # 使用conv2d进行卷积操作
        x1 = F.conv2d(P_before, self.kernel, padding=1)
        x = self.model(x1)
        x = x + residual

        return x


input_tensor1 = torch.randn(8, 1, 255, 255)  # 随机生成一个输入张量作为示例
input_tensor2 = torch.randn(8, 1, 255, 255)
structureEnhenceEWTBlock = StructureEnhenceEWTBlock()
# 调用函数处理输入张量
iwt_output = structureEnhenceEWTBlock(input_tensor1)

# 查看IWT输出的形状，应该与输入张量的形状相同
print(iwt_output.shape)
