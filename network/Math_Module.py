import torch
import torch.nn as nn


class P_calculate(nn.Module):
    """
        to solve min(P) = ||M-P-Q||^2 + γ||P-T||^2
        this is a least square problem
        how to solve?
        P = (gamma*T + M-Q) / (1 + gamma)
    """

    def __init__(self):
        super().__init__()

    def forward(self, M, Q, T, gamma):
        return ((M - Q + gamma * T) / (gamma + 1))


class Q_calculate(nn.Module):
    """
        to solve min(Q) = ||M-P-Q||^2 + λ||Q-S||^2
        Q = (M-P+lamda*S) / (1+lamda)
    """

    def __init__(self):
        super().__init__()

    def forward(self, M, P, S, lamda):
        return ((M - P + lamda * S) / (1 + lamda))


# 创建 P 和 Q 类的实例
p_instance = P_calculate()
q_instance = Q_calculate()

# 假设 M, Q, T, S 是已知的张量，gamma 和 lamda 是已知的参数
M = torch.randn(8, 1, 128, 128)  # 示例张量 M
Q = torch.randn(8, 1, 128, 128)  # 示例张量 Q
T = torch.randn(8, 1, 128, 128)  # 示例张量 T
S = torch.randn(8, 1, 128, 128)  # 示例张量 S
gamma = 0.5  # 示例参数 gamma
lamda = 0.5  # 示例参数 lamda

# 使用 P 类的实例来计算 P
P_value = p_instance(M, Q, T, gamma)

# 使用 Q 类的实例来计算 Q
Q_value = q_instance(M, P_value, S, lamda)
print(P_value.shape)
# 查看 Q 输出的形状，应该与输入张量的形状相同
print(Q_value.shape)
