import argparse
import torch.nn.functional as F
import cv2
import time
import numpy as np
from PIL import Image
import torch
import os
import torch.nn as nn
from torchvision.transforms import ToTensor
from network.decom import Decom
from network.Structure_enhance import StructureEnhenceEWTBlock
from network.Texture_enhance import TextureEnhanceNet
from network.Math_Module import P_calculate, Q_calculate
from utils_1.img_read_save import img_save

def np_save_TensorImg(img_tensor, path):
    # 将Tensor转换为NumPy数组，并进行必要的变换
    image_np = img_tensor.detach().cpu().squeeze().numpy()  # 单通道图像，不需要permute
    # 如果图像张量的数据范围不是 [0, 1]，可能需要规范化
    img = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    # 将图像数据的值限制在0到255之间，并将数据类型转换为无符号8位整数
    im = Image.fromarray(np.clip(img * 255, 0, 255).astype('uint8'))
    # 保存图像到指定路径
    im.save(path, 'PNG')

class UnfoldingModel(nn.Module):
    def __init__(self, opts):
        super(UnfoldingModel, self).__init__()
        self.opts = opts
        self.P1 = P_calculate()
        self.Q1 = Q_calculate()
        # 加载模型
        self.model_Decom_vis = Decom().to(opts.device)
        self.model_Decom_ir = Decom().to(opts.device)
        self.model_S = StructureEnhenceEWTBlock().to(opts.device)
        self.model_T = TextureEnhanceNet().to(opts.device)
        # 加载训练好的模型权重
        checkpoint = torch.load(opts.unfolding_model_path, map_location=opts.device)
        self.model_Decom_vis.load_state_dict(checkpoint['state_dict']['model_Decom_vis'])
        for param in self.model_Decom_vis.parameters():
            param.requires_grad = False
        self.model_Decom_ir.load_state_dict(checkpoint['state_dict']['model_Decom_ir'])
        for param in self.model_Decom_ir.parameters():
            param.requires_grad = False
        self.model_S.load_state_dict(checkpoint['state_dict']['model_S'])
        for param in self.model_S.parameters():
            param.requires_grad = False
        self.model_T.load_state_dict(checkpoint['state_dict']['model_T'])
        for param in self.model_T.parameters():
            param.requires_grad = False

    def forward(self, IR, VIS):
        M = (IR + VIS) / 2
        Svis, Tvis = self.model_Decom_vis(VIS)
        Sir, Tir = self.model_Decom_ir(IR)
        for t in range(self.opts.round):
            if t == 0:
                P_para, Q_para = Svis, Tir  # 使用预训练模型初始化P和Q
            else:
                w_p = (self.opts.gamma + self.opts.Roffset * t)
                w_q = (self.opts.lamda + self.opts.Loffset * t)
                P_para = self.P1(M=M, Q=Q_para, T=Tvis, gamma=w_p)
                Q_para = self.Q1(M=M, P=P_para, S=Sir, lamda=w_q)
                T = self.model_T(Q_para)
                S = self.model_S(P_para)
        return S, T


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configure')
    parser.add_argument('--vis_dir', type=str, default="./train_data/MSRS/train/vi")
    parser.add_argument('--ir_dir', type=str, default="./train_data/MSRS/train/ir")
    parser.add_argument('--Mfinal_dir', type=str, default="./train_data/MSRS/train/MSRS_round2")
    parser.add_argument('--unfolding_model_path', type=str, default="./ckpt/all_model-4.pth")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--round', type=int, default=2, help='Number of rounds for unfolding')
    parser.add_argument('--gamma', type=float, default=20, help='Gamma parameter for unfolding')
    parser.add_argument('--Roffset', type=float, default=0.1, help='R offset parameter for unfolding')
    parser.add_argument('--lamda', type=float, default=10, help='Lambda parameter for unfolding')
    parser.add_argument('--Loffset', type=float, default=0.1, help='L offset parameter for unfolding')
    opts = parser.parse_args()

    device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
    opts.device = device  # 将设备添加到opts中

    unfoldingModel = UnfoldingModel(opts).to(device)  # 传递opts参数
    para = sum([np.prod(list(p.size())) for p in unfoldingModel.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(unfoldingModel._get_name(), para * type_size / 1000 / 1000))

    total = sum([param.nelement() for param in unfoldingModel.parameters()])
    print('Number of parameter: {:4f}M'.format(total / 1e6))



    transform = ToTensor()

    total_time = 0
    num_images = 0

    for img_name in os.listdir(opts.vis_dir):
        if img_name not in os.listdir(opts.ir_dir):
            continue

        vis_img_path = os.path.join(opts.vis_dir, img_name)
        ir_img_path = os.path.join(opts.ir_dir, img_name)

        vis_image = Image.open(vis_img_path).convert('L')  # 转换为灰度图
        ir_image = Image.open(ir_img_path).convert('L')  # 转换为灰度图
        vis_image_tensor = transform(vis_image).unsqueeze(0).to(device)  # 增加批次维度并移动到设备
        ir_image_tensor = transform(ir_image).unsqueeze(0).to(device)  # 增加批次维度并移动到设备

        start_time = time.time()

        with torch.no_grad():
            Sfinal, Tfinal = unfoldingModel(ir_image_tensor, vis_image_tensor)
            Mfinal = Sfinal + Tfinal

        end_time = time.time()
        total_time += (end_time - start_time)
        num_images += 1

        # 确保数据有限且在有效范围内
        data_Fuse = torch.clamp(Mfinal, 0, 1)
        fi = np.squeeze((data_Fuse * 255).cpu().detach().numpy())

        # 检查并替换 NaN 和 Inf 值
        fi = np.nan_to_num(fi, nan=0.0, posinf=0.0, neginf=0.0)

        # 确保像素值在 0-255 范围内
        fi = np.clip(fi, 0, 255).astype(np.uint8)

        # 构建保存路径
        save_dir = opts.Mfinal_dir  # 保存目录
        image_name = os.path.splitext(img_name)[0]  # 从 img_name 中提取不带扩展名的文件名作为 imagename

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 尝试保存图像并打印输出信息
        try:
            img_save(fi, image_name, save_dir)  # 按照 img_save 的定义传入三个参数
            print(f"Image saved successfully: {os.path.join(save_dir, image_name)}.png")
        except Exception as e:
            print(f"Failed to save image: {os.path.join(save_dir, image_name)}.png. Error: {e}")

    if num_images > 0:
        avg_time = total_time / num_images
        print(f"Average processing time per image: {avg_time:.4f} seconds")
    else:
        print("No images processed.")