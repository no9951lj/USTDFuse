import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from network.decom import Decom
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torchvision import models
from utils_1.loss import cc
import kornia.losses as K
import kornia
# 显示图像样本
def show_image(image_tensor, title=None):
    # 确保图像张量在CPU上，并转换为NumPy数组
    image_np = image_tensor.cpu().numpy()

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

class CustomDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, transform=None, num_patches=9):
        """
        image_dir: 包含图像的目录路径
        patch_size: 每个patch的大小
        transform: 应用于每个patch的转换
        num_patches: 每张图像生成的patch数量
        """
        self.image_dir = image_dir
        self.patch_size = patch_size
        self.transform = transform
        self.num_patches = num_patches
        # 过滤掉非图像文件和.DS_Store文件
        self.image_files = [f for f in os.listdir(image_dir)
                           if os.path.isfile(os.path.join(image_dir, f))
                           and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                           and f != '.DS_Store']
        self.image_files.sort()  # 确保文件排序一致
        print("Sorted image files:", self.image_files)  # 打印排序好的文件列表

    def __len__(self):
        # 每张图像生成num_patches个patch
        return len(self.image_files) * self.num_patches

    def __getitem__(self, idx):
        # 计算属于哪张图像
        img_idx = idx // self.num_patches
        img_name = self.image_files[img_idx]
        img_path = os.path.join(self.image_dir, img_name)
        # image = Image.open(img_path).convert('RGB')
        image = Image.open(img_path).convert('L')  # 转换为灰度图

        # 计算patch的起始坐标
        patch_idx = idx % self.num_patches
        x = (patch_idx % 3) * self.patch_size
        y = (patch_idx // 3) * self.patch_size
        patch = image.crop((x, y, x + self.patch_size, y + self.patch_size))

        if self.transform:
            patch = self.transform(patch)
        # 打印数据样本的最小值和最大值
        # print(f"Patch {idx} min/max: {patch.min().item()} / {patch.max().item()}")

        return patch

def gradient_loss(image):
    """
    计算图像的梯度。

    参数:
    image (torch.Tensor): 输入图像张量，形状为 (B, C, H, W)

    返回:
    torch.Tensor: 图像梯度张量，形状为 (B, C*2, H, W)
    """
    # 定义水平和垂直方向的卷积核
    kernel_x = torch.tensor([[[[-1., 1.]]]], dtype=torch.float32).repeat(image.shape[1], 1, 1, 1).to(image.device)
    kernel_y = torch.tensor([[[[-1., 1.]]]], dtype=torch.float32).repeat(image.shape[1], 1, 1, 1).to(image.device)

    # 对输入张量应用填充
    padded_image = F.pad(image, (1, 0, 0, 1))  # 对高度和宽度的两端各填充1个像素

    # 计算梯度
    grad_x = F.conv2d(padded_image, kernel_x, groups=image.shape[1])
    grad_y = F.conv2d(padded_image, kernel_y, groups=image.shape[1])

    # 合并梯度
    grad = torch.cat((grad_x, grad_y), dim=1)
    norm = torch.norm(grad, p=2)

    return norm
class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super(VGGFeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True)
        self.vgg_layers = vgg19.features
        self.vgg_layers = nn.Sequential(*list(self.vgg_layers.children())[:22])  # 选择前22层，即到conv5_4

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.vgg_layers):
            x = layer(x)
            if i in [1, 7, 14, 21, 28]:  # 选择特定的层
                outputs.append(x)
        return outputs

# 定义损失函数
# 定义损失函数
def vgg_perceptual_loss(Bvis, Dvis, Bir, Dir, layer_weights):
    vgg_model = VGGFeatureExtractor()
    vgg_model = vgg_model.to(device)
    vgg_model.eval()
    for param in vgg_model.parameters():
        param.requires_grad = False
    Bvis = Bvis.repeat(8, 3, 1, 1)  # 复制单通道到三个通道
    Dvis = Dvis.repeat(8, 3, 1, 1)
    Bir = Bir.repeat(8, 3, 1, 1)
    Dir = Dir.repeat(8, 3, 1, 1)
    Bvis = Bvis.to(device)
    Dvis = Dvis.to(device)
    Bir = Bir.to(device)
    Dir = Dir.to(device)
    Bvis_features = vgg_model(Bvis)
    Dvis_features = vgg_model(Dvis)
    Bir_features = vgg_model(Bir)
    Dir_features = vgg_model(Dir)

    total_loss = 0
    for i, (Bvis_feature, Dvis_feature, Bir_feature, Dir_feature) in enumerate(zip(Bvis_features, Dvis_features, Bir_features, Dir_features)):
        if i == 0:  # 第2层，只计算loss_D
            loss_D = nn.MSELoss()(Dvis_feature, Dir_feature.detach())
            total_loss += loss_D * layer_weights[0]
        elif i == 1:  # 第7层，只计算loss_D
            loss_D = nn.MSELoss()(Dvis_feature, Dir_feature.detach())
            total_loss += loss_D * layer_weights[1]
        elif i == 2:  # 第14层，计算loss_B和loss_D，各占0.5
            loss_B = nn.MSELoss()(Bvis_feature, Bir_feature.detach())
            loss_D = nn.MSELoss()(Dvis_feature, Dir_feature.detach())
            total_loss += (loss_B + loss_D) * layer_weights[2] * 0.5
        elif i == 3:  # 第21层，计算loss_B和loss_D，各占0.5
            loss_B = nn.MSELoss()(Bvis_feature, Bir_feature.detach())
            loss_D = nn.MSELoss()(Dvis_feature, Dir_feature.detach())
            total_loss += (loss_B + loss_D) * layer_weights[3] * 0.5
        elif i == 4:  # 第28层，只计算loss_B
            loss_B = nn.MSELoss()(Bvis_feature, Bir_feature.detach())
            total_loss += loss_B * layer_weights[4]

    return total_loss

# 定义每组层的权重
layer_weights = [1, 1, 1, 1, 1]  # 可以根据需要调整权重


# 设置数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
    # 可以添加更多的转换操作，如归一化等
])
parser = argparse.ArgumentParser(description='Configure')
# specify your data path here!
parser.add_argument('--vis_dir', type=str, default="./train_data/DATFuse/train/vi")
parser.add_argument('--ir_dir', type=str, default="./train_data/DATFuse/train/ir")
parser.add_argument('--model_path', type=str, default='decom-DATFuse.pth')
parser.add_argument('--gpu_id', type=int, default=0)
opts = parser.parse_args()

device = torch.device(f"cuda:{opts.gpu_id}" if torch.cuda.is_available() else "cpu")
opts.device = device  # 将设备添加到opts中
train_dataset_vis = CustomDataset(opts.vis_dir, transform=transform)
train_dataset_ir = CustomDataset(opts.ir_dir, transform=transform)

train_loader_vis = DataLoader(train_dataset_vis, batch_size=16, shuffle=False)
train_loader_ir = DataLoader(train_dataset_ir, batch_size=16, shuffle=False)
Loss_ssim = K.SSIMLoss(window_size=11, reduction='mean')
MSELoss = nn.MSELoss()
# 初始化模型
model_Decom_vis = Decom().to(device)
model_Decom_ir = Decom().to(device)
# 加载模型和优化器状态
# 加载模型和优化器状态
checkpoint = torch.load(opts.model_path, map_location=device)
model_Decom_vis.load_state_dict(checkpoint['state_dict']['model_Decom_vis_state_dict'])
model_Decom_ir.load_state_dict(checkpoint['state_dict']['model_Decom_ir_state_dict'])
L1Loss = nn.L1Loss()
optimizer = torch.optim.Adam(list(model_Decom_vis.parameters()) + list(model_Decom_ir .parameters()), lr=0.0001)
# 训练循环
num_epochs = 10  # 设置训练的轮数
for epoch in range(num_epochs):
    # adjust_learning_rate(optimizer, epoch, 0.0001)  # 更新学习率
    for vis_imgs, ir_imgs in zip(train_loader_vis , train_loader_ir ):
        optimizer.zero_grad()
        vis_imgs, ir_imgs = vis_imgs.to(device), ir_imgs.to(device)
        Bvis,Dvis = model_Decom_vis(vis_imgs)  # 获取损失值
        Bir, Dir = model_Decom_ir(ir_imgs)
        cc_loss_S = cc(Bvis, Bir)
        cc_loss_T = cc(Dvis, Dir)
        loss_decomp = (cc_loss_T) ** 2 / (1.01 + cc_loss_S)
        vgg_loss = vgg_perceptual_loss(Bvis, Dvis, Bir, Dir, layer_weights)
        gradient_loss1=gradient_loss(Dvis)
        gradient_loss2 = gradient_loss(Dir)
        recon_loss1 = F.l1_loss(Bvis+Dvis, vis_imgs)
        recon_loss2 = F.l1_loss(Bir+Dir, ir_imgs)
        l1_loss1 = torch.sum(torch.abs(Bvis))
        l1_loss2 = torch.sum(torch.abs(Bir))
        mse_loss_V = 5 * Loss_ssim(Bvis + Dvis, vis_imgs)
        mse_loss_I = 5 * Loss_ssim(Bir + Dir, ir_imgs)
        recon_loss1 = recon_loss1 + 5 * Loss_ssim(Bvis + Dvis, vis_imgs)
        recon_loss2 = recon_loss2 + 0.1*5 * Loss_ssim(Bir + Dir, ir_imgs)
        total_loss = 10*vgg_loss + 0.0001*loss_decomp + recon_loss1 + recon_loss2  + 0.1*(0.000001 * (l1_loss1 )+ 0.0000004 *l1_loss2+0.0000001*(gradient_loss1+gradient_loss2))

        # total_loss = mse_loss_V + mse_loss_I + loss_decomp  + recon_loss1 + recon_loss2

        total_loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss.item():.4f}')
        # 每隔两个epoch显示最后一组数据
    if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
        # 选择要显示的patch索引，这里我们选择每个epoch显示不同的patch
        # batch_data_vis = next(iter(train_loader_vis))
        # batch_data_ir = next(iter(train_loader_ir))
        sample_vis = vis_imgs[0].cpu()  # 获取第一个样本的低光照图像
        sample_ir = ir_imgs[0].cpu()  # 获取第一个样本的高光照图像
        sample_Bvis = Bvis[0].cpu().detach()  # 获取第一个样本的R
        sample_Dvis = Dvis[0].cpu().detach()  # 获取第一个样本的L
        sample_Bir = Bir[0].cpu().detach()  # 获取第一个样本的hat_R
        sample_Dir = Dir[0].cpu().detach()  # 获取第一个样本的hat_L

        reconstructed_vis = sample_Bvis + sample_Dvis  # 假设没有偏置项
        reconstructed_ir = sample_Bir + sample_Dir  # 假设没有偏置项


        # 使用示例
        show_image(sample_vis, title='vis')  # 显示高光照图像并设置标题
        show_image(sample_ir, title='ir')  # 显示低光照图像并设置标题
        show_image(sample_Bvis, title='Bvis')  # 显示R并设置标题
        show_image(sample_Dvis, title='Dvis')  # 显示L并设置标题
        show_image(sample_Bir, title='Bir')  # 显示重构的I并设置标题
        show_image(sample_Dir, title='Dir')  # 显示hat_R和
        show_image(reconstructed_vis, title='reconstructed_vis')  # 显示重构的I并设置标题
        show_image(reconstructed_ir, title='reconstructed_ir')  # 显示hat_R和


# 保存模型
checkpoint = {
    'opts': opts,
    'state_dict': {
        'model_Decom_vis_state_dict': model_Decom_vis.state_dict(),
        'model_Decom_ir_state_dict': model_Decom_ir.state_dict(),
    }
}
torch.save(checkpoint, './ckpt/decom-enhence1.pth')