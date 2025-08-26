import argparse
import torch
import torch.nn as nn
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from network.decom import Decom
from network.Structure_enhance import StructureEnhenceEWTBlock
from network.Texture_enhance import TextureEnhanceNet
from network.Math_Module import P_calculate, Q_calculate
from utils_1.utils import load_initialize, SSIM
import os
import torch
import random
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils_1.loss import Fusionloss, cc
from torch.cuda.amp import autocast, GradScaler  # 导入自动混合精度计算相关模块
import kornia.losses as K
import kornia
import time
from utils_1.img_read_save import img_save, image_read_cv2

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# 设置随机种子为一个固定的值，例如42
set_seed(0)


def show_image(image_tensor, title=None):
    # 确保图像张量在CPU上，并转换为NumPy数组
    image_np = image_tensor.detach().cpu().numpy()

    # 如果图像张量的数据范围不是 [0, 1]，可能需要规范化
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

    # 确保数据类型为 float
    image_np = image_np.astype(np.float32)

    # 调整维度顺序从 (C, H, W) 到 (H, W, C)
    if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
        image_np = np.transpose(image_np, (1, 2, 0))
        # 对于单通道图像，matplotlib需要额外的处理
        if image_np.shape[2] == 1:
            image_np = np.repeat(image_np, 3, axis=2)  # 将单通道复制到三个通道

    # 如果图像张量的数据类型是 float，可能需要转换为 uint8
    image_np = (image_np * 255).astype(np.uint8)

    plt.imshow(image_np)
    if title is not None:
        plt.title(title)  # 如果提供了标题，则显示标题
    plt.show()


class CustomDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, transform=None, num_patches=9):
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform
        self.num_patches = num_patches
        # 过滤掉非图像文件和.DS_Store文件
        self.image_files = [f for f in os.listdir(image_dir)
                            if os.path.isfile(os.path.join(image_dir, f))
                            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                            and f!= '.DS_Store']
        self.image_files.sort()  # 确保文件排序一致

    def __len__(self):
        return len(self.image_files) * self.num_patches

    def __getitem__(self, idx):
        img_idx = idx // self.num_patches
        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, f"{img_name}")
        # image = image_read_cv2(img_path, mode='GRAY')[np.newaxis, np.newaxis, ...] / 255.0
        image = Image.open(img_path).convert('L')

        patch_idx = idx % self.num_patches
        x = (patch_idx % 3) * self.patch_size
        y = (patch_idx // 3) * self.patch_size
        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.transform:
            patch = self.transform(patch)

        return patch
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)
class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.sobelconv=Sobelxy()

    def forward(self,image_vis,image_ir,generate_img):
        image_y=image_vis
        x_in_max=torch.max(image_y,image_ir)
        loss_in=F.l1_loss(x_in_max,generate_img)
        y_grad=self.sobelconv(image_y)
        ir_grad=self.sobelconv(image_ir)
        generate_img_grad=self.sobelconv(generate_img)
        x_grad_joint=torch.max(y_grad,ir_grad)
        loss_grad=F.l1_loss(x_grad_joint,generate_img_grad)
        loss_total=1*loss_in+10*loss_grad
        return loss_total
# 设置数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

parser = argparse.ArgumentParser(description='Configure')
parser.add_argument('--vis_dir', type=str, default="./train_data/MSRS/train/vi")
parser.add_argument('--ir_dir', type=str, default="./train_data/MSRS/train/ir")
parser.add_argument('--Decom_model_path', type=str, default="./ckpt/decom-enhence1.pth")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--round', type=int, default=3, help='Number of rounds for unfolding')
parser.add_argument('--gamma', type=float, default=20, help='Gamma parameter for unfolding')
parser.add_argument('--Roffset', type=float, default=0.1, help='R offset parameter for unfolding')
parser.add_argument('--lamda', type=float, default=10, help='Lambda parameter for unfolding')
parser.add_argument('--Loffset', type=float, default=0.1, help='L offset parameter for unfolding')
opts = parser.parse_args()

device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
opts.device = device  # 将设备添加到opts中

train_dataset_vis = CustomDataset(opts.vis_dir, transform=transform)
train_dataset_ir = CustomDataset(opts.ir_dir, transform=transform)
lr = 1e-4
weight_decay = 0
train_loader_vis = DataLoader(train_dataset_vis, batch_size=16, shuffle=False)
train_loader_ir = DataLoader(train_dataset_ir, batch_size=16, shuffle=False)
model_Decom_vis = Decom().to(device)
model_Decom_ir = Decom().to(device)
checkpoint = torch.load(opts.Decom_model_path, map_location=opts.device)
model_Decom_vis.load_state_dict(checkpoint['state_dict']['model_Decom_vis_state_dict'])
model_Decom_ir.load_state_dict(checkpoint['state_dict']['model_Decom_ir_state_dict'])
model_S = StructureEnhenceEWTBlock().to(device)
model_T =TextureEnhanceNet().to(device)
# 创建优化器和调度器
optimizer_decom_vis = torch.optim.Adam(model_Decom_vis.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_decom_ir = torch.optim.Adam(model_Decom_ir.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_s = torch.optim.Adam(model_S.parameters(), lr=lr, weight_decay=weight_decay)
optimizer_t = torch.optim.Adam(model_T.parameters(), lr=lr, weight_decay=weight_decay)

scheduler_decom_vis = torch.optim.lr_scheduler.StepLR(optimizer_decom_vis, step_size=20, gamma=0.5)
scheduler_decom_ir = torch.optim.lr_scheduler.StepLR(optimizer_decom_ir, step_size=20, gamma=0.5)
scheduler_s = torch.optim.lr_scheduler.StepLR(optimizer_s, step_size=20, gamma=0.5)
scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, step_size=20, gamma=0.5)
# 记录开始时间
start_time = time.time()
num_epochs = 50  #80变成了120
Loss_ssim = K.SSIMLoss(window_size=11, reduction='mean')
alpha = 1
beta = 2
# epoch_gap = 10  # epoches of Phase I
clip_grad_norm_value = 0.01
ssim_metric = SSIM(window_size=11, size_average=True, val_range=None)
fusionloss = Fusionloss().cuda()  # 将模型移动到GPU
P1 = P_calculate()
Q1 = Q_calculate()
L1Loss = nn.L1Loss()
MSELoss = nn.MSELoss()
# 初始化GradScaler，用于混合精度训练时缩放梯度
scaler = GradScaler()

for epoch in range(num_epochs):
    for ir_imgs, vis_imgs in zip(train_loader_ir, train_loader_vis):
        ir_imgs, vis_imgs = ir_imgs.to(device), vis_imgs.to(device)

        # 清零梯度
        optimizer_decom_vis.zero_grad()
        optimizer_decom_ir.zero_grad()
        optimizer_s.zero_grad()
        optimizer_t.zero_grad()

        Svis, Tvis = model_Decom_vis(vis_imgs)
        Sir, Tir = model_Decom_ir(ir_imgs)
        frobenius_loss1 = 0
        frobenius_loss2 = 0
        M = (vis_imgs + vis_imgs) / 2
        for t in range(opts.round):
            if t == 0:
                P_para, Q_para = (Tvis + Tir) / 2 , (Sir + Svis) / 2  # 使用预训练模型初始化P和Q
            else:
                w_p = (opts.gamma + opts.Roffset * t)
                w_q = (opts.lamda + opts.Loffset * t)
                P_para = P1(M=M, Q=Q_para, T=Tvis, gamma=w_p)
                Q_para = Q1(M=M, P=P_para, S=Sir, lamda=w_q)
                T = model_T(Q_para)
                S = model_S(P_para)
        #         frobenius_loss1 += F.l1_loss(P_para, S)
        #         frobenius_loss2 += F.l1_loss(Q_para, T)
        #
        # recon_loss1 = F.l1_loss(Svis + Tvis, vis_imgs)
        # recon_loss2 = F.l1_loss(Sir + Tir, ir_imgs)
        # cc_loss_S = cc(Svis, Sir)
        # cc_loss_T = cc(Tvis, Tir)
        # loss_decomp = (cc_loss_T) ** 2 / (1.01 + cc_loss_S)
        # mse_loss_V = 5 * Loss_ssim(Svis + Tvis, vis_imgs)
        # mse_loss_I = 5 * Loss_ssim(Sir + Tir, ir_imgs)
        # # Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(Svis + Tvis),
        # #                        kornia.filters.SpatialGradient()(vis_imgs))
        # total_loss1 = mse_loss_V + mse_loss_I + loss_decomp + recon_loss1 + recon_loss2

        sobel_fused = kornia.filters.Sobel()(S + T)
        sobel_ir = kornia.filters.Sobel()(ir_imgs)
        sobel_vis = kornia.filters.Sobel()(vis_imgs)
        gradient_loss = nn.MSELoss()(sobel_fused, sobel_ir) + nn.MSELoss()(sobel_fused, sobel_vis)
        mse_loss_V = 5 * Loss_ssim(vis_imgs, S + T) + MSELoss(vis_imgs, S + T)
        mse_loss_I = 5 * Loss_ssim(ir_imgs, S + T) + MSELoss(ir_imgs, S + T)
        recon_loss = fusionloss(vis_imgs, ir_imgs, S + T)
        total_loss = 2* recon_loss +(mse_loss_V + mse_loss_I) + 5*gradient_loss
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            model_Decom_vis.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            model_Decom_ir.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            model_S.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(
            model_T.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer_decom_vis.step()
        optimizer_decom_ir.step()
        optimizer_s.step()
        optimizer_t.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')
    if torch.cuda.is_available():
        print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
        print(f"GPU Max Memory Allocated: {torch.cuda.max_memory_allocated(device) / (1024 * 1024):.2f} MB")
        # 每隔两个epoch显示最后一组数据
# 记录结束时间
end_time = time.time()
# 计算训练时间
train_time = end_time - start_time
print(f"Training Time: {train_time:.2f} seconds")

# 保存模型
checkpoint = {
    'opts': opts,
    'state_dict': {
        'model_Decom_vis': model_Decom_vis.state_dict(),
        'model_Decom_ir': model_Decom_ir.state_dict(),
        'model_S': model_S.state_dict(),
        'model_T': model_T.state_dict()
    }
}
torch.save(checkpoint, './ckpt/all_model-4.pth')