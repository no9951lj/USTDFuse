from network.architecture import *


class Decom(nn.Module):
    def __init__(self):
        super().__init__()
        self.decom = nn.Sequential(
            get_conv2d_layer(in_c=1, out_c=8, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=8, out_c=16, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=16, out_c=32, k=3, s=1, p=1),
            nn.LeakyReLU(0.2, inplace=True),
            get_conv2d_layer(in_c=32, out_c=2, k=3, s=1, p=1),
            nn.ReLU()
        )

    def forward(self, input):
        output = self.decom(input)
        Base = output[:, 0:1, :, :]
        Detail = output[:, 1:2, :, :]
        return Base, Detail
