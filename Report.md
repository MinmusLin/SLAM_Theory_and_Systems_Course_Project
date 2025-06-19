# SLAM个人报告——第三部分：DELTAS复现

# (cooperating with 林继申)

## 3.DELTAS复现实验

### 3.1 引言

传统的**监督式深度估计方法**在训练过程中严重依赖于大量高质量的真实深度标签，这些标签通常通过激光雷达（LiDAR）、结构光（structured light）或飞行时间（ToF）传感器获取。然而，这类设备价格昂贵、部署复杂，且在部分复杂或动态环境中难以保证一致性与稳定性，导致大规模真实标签数据的获取成本极高，成为深度学习方法进一步推广和应用的主要瓶颈。

为了突破这一限制，近年来学术界与工业界对**无监督（unsupervised）或弱监督（weakly-supervised）深度估计方法**的研究热情持续上升。此类方法通常以图像重建、几何一致性或稀疏监督为目标，从无需标签的图像对、视频序列或少量点云信息中学习出图像的深度结构，显著降低了对全量标签的依赖，具有更强的可扩展性和应用灵活性。

在此研究背景下，**DELTAS（Depth Estimation by Learning Triangulation And densification of Sparse points）方法**应运而生。该方法提出了一种创新性的训练框架，通过**三角测量监督机制**（triangulation-based supervision）与**稀疏点密集化策略**（densification of sparse points），在无需完整深度图监督的前提下，仅依赖从图像序列中三角测量得到的稀疏深度点，即可实现高质量的深度图预测。

DELTAS 的核心思想在于结合传统多视角几何中的三角测量原理与现代深度学习中的端到端特征建模能力：

- 首先，通过对多帧图像中的兴趣点进行匹配与三角测量，生成带有相对尺度信息的稀疏三维点；
- 然后，构建一个**深度补全网络**（sparse-to-dense network），以这些稀疏点为锚点，结合图像纹理与上下文信息，预测完整稠密的深度图；
- 最后，通过端到端训练，联合优化关键点检测、三角化精度和稠密深度图预测模块，有效提升模型对几何结构的感知能力与预测准确性。

本项目的目标是**复现 DELTAS 方法在 Whole Apartment 数据集上的深度估计性能**，通过从数据预处理、模型结构搭建、训练流程优化到误差评估等多个环节，全面评估其在复杂室内场景下的表现。同时，我们还将围绕其在泛化能力、边缘保留与姿态依赖性等方面存在的问题，尝试提出并验证若干改进策略。

在实验评估方面，我们将采用常用的深度估计性能指标，包括：

- **绝对相对误差（Abs Rel）**：衡量预测误差相对于真实深度的比例；
- **均方根误差（RMSE）**：度量整体误差幅度；
- **对数均方根误差（RMSE log）**：增强对深度较小区域的误差敏感性，具有较强鲁棒性。

通过以上实验与分析，我们希望深入理解 DELTAS 方法在数据有限、场景复杂条件下的适应能力，并探索其在实际系统中的应用潜力与发展方向。

### 3.2 方法

#### 3.2.1 方法简介

**DELTAS 方法（Depth Estimation by Learning Triangulation And densification of Sparse points）**的设计目标是在没有传统密集标签监督的前提下，通过少量的稀疏几何信息实现高质量的深度图预测。它显著区别于传统基于代价体（cost volume）的深度学习方法，后者通常需要构建大规模的体素结构来模拟视差搜索空间，计算与存储成本非常高，尤其在高分辨率或多视角情况下难以扩展。

DELTAS 提出了一种**轻量级但高效的替代方案**，通过从稀疏但高置信度的匹配点出发，构建结构先验，并逐步 densify（密集化）整个深度图。整个方法框架分为三个互相协作的关键阶段：

##### 3.2.2.1. 兴趣点检测与描述器网络（Interest Point Detection and Description）

在第一阶段，DELTAS 使用基于卷积神经网络的结构（如 SuperPoint 风格的网络）从图像中提取稳定、具有判别力的兴趣点（keypoints）以及其对应的特征描述子（descriptors）。

- 相比传统的手工特征（如 SIFT、ORB），该方法可以**端到端训练**兴趣点检测器和描述子生成器，从而更好地适应目标场景和任务。
- 特征提取过程中同时考虑图像的上下文信息，使得生成的兴趣点具有更高的可重复性和匹配稳定性，尤其在纹理弱、光照变化大等实际环境下仍具备鲁棒性。

##### 3.2.2.2. 极线匹配与三角化（Matching and Triangulation）

第二阶段通过对多视图图像之间的兴趣点进行精确匹配，并结合已知的相机位姿（或通过 PnP 推断得到）进行**三角化计算**，以估算出空间中稀疏而可靠的三维点。

- 与传统方法不同，DELTAS 利用可微分模块实现了三角化过程的**端到端学习**，例如通过 soft-argmax 匹配点位置、基于 SVD 的几何解法，从而减小由量化、采样带来的误差。
- 该阶段不仅恢复了场景的几何结构，而且为后续的稠密化提供了初始深度锚点。

##### 3.2.2.3. 稀疏矩阵稠密化（Densification of Sparse Points）

最后一个阶段是 DELTAS 的核心创新之一，即将稀疏的深度点作为监督信号，通过构建一个融合图像特征与几何特征的**稀疏到稠密网络（Sparse-to-Dense Network）**，实现完整场景的深度图预测。

- 网络结构通常采用 encoder-decoder 框架，结合 ASPP、U-Net 等模块以增强多尺度上下文建模能力。
- 输入为图像 RGB 特征图和稀疏深度图，输出为完整的 dense depth map。在训练过程中通过 Smooth L1 Loss、边缘感知平滑损失等多项损失函数引导模型保留结构细节，提升预测质量。
- 相比依赖 3D cost volume 的方法，DELTAS 显著减少了内存和计算开销，**更加轻量化且更易扩展**到高分辨率或资源受限的场景。

#### 3.2.2 方法模块实现

##### 3.2.1.1 关键点检测与描述器网络

该网络采用ResNet-50 编码器提取图像特征，之后通过两个 decoder 分支：

- 检测器分支输出 heatmap（即 SuperPoint 中的兴趣点概率图）

- 描述器分支输出 descriptor field（用于后续匹配）

该方法主要有以下优势：
（1）使用SuperPoint-like网络端到端学习特征点与描述子，抗光照/尺度变化能力更强。
（2）提取过程通过图像上下文优化特征位置和分布，提高点的可重复性和准确率。
（3）训练目标包括匹配精度与兴趣点质量，提供鲁棒输入。

```python
# models/interest_point_net.py
import torch
import torch.nn as nn
import torchvision.models as models

class InterestPointNet(nn.Module):
    def __init__(self, descriptor_dim=128):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])  # 输出特征图大小为 H/32 x W/32

        self.up = nn.Sequential(
            nn.ConvTranspose2d(2048, 512, 2, 2),  # 上采样至 H/16
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 2, 2),   # 上采样至 H/8
            nn.ReLU()
        )

        # 检测器分支（输出 65 通道）
        self.detector_head = nn.Conv2d(256, 65, kernel_size=1)

        # 描述器分支（输出 descriptor_dim 通道）
        self.descriptor_head = nn.Conv2d(256, descriptor_dim, kernel_size=1)

    def forward(self, x):
        feat = self.encoder(x)     # 特征提取
        feat = self.up(feat)       # 上采样
        heatmap = self.detector_head(feat)
        descriptor = self.descriptor_head(feat)
        return heatmap, descriptor

```

##### 3.2.1.2  极线匹配和三角化

对于 anchor 图像中的关键点，通过相对位姿矩阵将其投影到参考视角的极线

在极线采样点上进行 descriptor 匹配

用 SoftArgmax 得到匹配点坐标

使用 SVD 对多视角匹配点三角化，输出稀疏 3D 点（depth）

相比上节提到的方法，主要有以下优势：
（1）网络使用三视图构建多视角冗余，弥补小视差问题，通过加权SVD方法处理匹配点置信度，弱视角的图像贡献更少，增强稳定性。解决了如基线短、视差小，三角化不稳定的问题。
（2）使用可微分SVD解法实现稳健的线性三角测量，避免退化点对误差放大。匹配点位置通过softmax加权平均，降低离散搜索点引入的数值跳变。从而减少了退化几何、极线附近匹配不可靠导致的数值不稳定性。

```python
import torch
import torch.nn.functional as F

def soft_argmax(corr_map):
    """ 对 descriptor 相关性图做 soft-argmax 以估计2D匹配点坐标 """
    B, H, W = corr_map.shape
    corr_map = corr_map.view(B, -1)
    softmax = F.softmax(corr_map, dim=1).view(B, H, W)

    coords_x = torch.linspace(0, W - 1, W, device=corr_map.device).view(1, 1, W)
    coords_y = torch.linspace(0, H - 1, H, device=corr_map.device).view(1, H, 1)
    exp_x = torch.sum(softmax * coords_x, dim=2)
    exp_y = torch.sum(softmax * coords_y, dim=1)
    return torch.stack([exp_x, exp_y], dim=2)  # shape: [B, N, 2]

```

```python
def triangulate_point(matches_2d, proj_mats, confidences):
    """
    使用多视角2D点与投影矩阵，进行可微三角化（SVD）

    参数:
        matches_2d: [B, V, 2] 不同视角下2D点坐标
        proj_mats: [B, V, 3, 4] 相机投影矩阵
        confidences: [B, V] 每个视角的匹配置信度

    返回:
        triangulated 3D 点 [B, 3]
    """
    B, V = matches_2d.shape[:2]
    A = []

    for v in range(V):
        x, y = matches_2d[:, v, 0], matches_2d[:, v, 1]
        P = proj_mats[:, v]  # [B, 3, 4]

        row1 = x.unsqueeze(1) * P[:, 2, :] - P[:, 0, :]  # [B, 4]
        row2 = y.unsqueeze(1) * P[:, 2, :] - P[:, 1, :]  # [B, 4]

        A.append(row1)
        A.append(row2)

    A = torch.stack(A, dim=1)  # [B, 2V, 4]

    weights = confidences.repeat_interleave(2, dim=1).unsqueeze(-1)  # [B, 2V, 1]
    A_weighted = A * weights  # [B, 2V, 4]

    # SVD: 求解 Az = 0，z为4维齐次坐标
    _, _, Vh = torch.linalg.svd(A_weighted)
    z_homo = Vh[:, -1]  # 最后一个特征向量
    z = z_homo[:, :3] / z_homo[:, 3:].clamp(min=1e-6)  # 齐次转非齐次
    return z  # [B, 3]

```

##### 3.2.1.3 稀疏矩阵稠密化

构造一个带 ASPP 模块的 encoder-decoder 网络

输入 RGB 图像特征 + 稀疏深度图特征

输出稠密深度图

相比上节提到的方法，主要有以下优势：
（1）结合图像和稀疏深度特征的U-Net结构完成高质量稠密预测。
（2）保留几何精度（来自三角化），融合图像上下文结构，提升边缘细节和连续性。
（3）支持多尺度监督、ASPP模块增强结构理解。
避免了传统三角化输出点稀疏，无法用于完整深度估计的问题。

```python
# models/sparse_to_dense_net.py
import torch.nn.functional as F

class ASPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous1 = nn.Conv2d(in_channels, out_channels, 3, padding=3, dilation=3)
        self.atrous2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.atrous3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.atrous4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.out = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        x1 = F.relu(self.atrous1(x))
        x2 = F.relu(self.atrous2(x))
        x3 = F.relu(self.atrous3(x))
        x4 = F.relu(self.atrous4(x))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out(out)

class SparseToDenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 编码器部分（可替换为ResNet）
        self.rgb_encoder = models.resnet18(pretrained=True)
        self.depth_encoder = models.resnet18(pretrained=False)

        # 解码器部分
        self.aspp = ASPPBlock(512 + 512, 256)  # 拼接后通道为1024

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 1, 1)  # 输出深度图
        )

    def forward(self, rgb, sparse_depth):
        feat_rgb = self.rgb_encoder.conv1(rgb)
        feat_depth = self.depth_encoder.conv1(sparse_depth)
        x = torch.cat([feat_rgb, feat_depth], dim=1)
        x = self.aspp(x)
        return self.decoder(x)

```

##### 3.2.1.4 损失函数

为了指导网络端到端地训练，我们设计了如下多项损失：
1、关键点检测 CrossEntropy Loss
2、描述子匹配点的2D位置 L1 Loss
3、三角化的 3D 点 L1 Loss
4、稠密深度图的 SmoothL1 多尺度监督
5、边缘保持平滑性损失，避免图像边缘的深度模糊

```python
def multiscale_depth_loss(pred_depths, gt_depth, valid_mask=None):
    """
    多尺度 supervision 的稠密深度损失
    参数:
        pred_depths: List of [B, 1, H_i, W_i] 多尺度输出
        gt_depth:    [B, 1, H, W]
        valid_mask:  [B, 1, H, W] (optional)
    """
    loss = 0.0
    weight = 1.0
    for pred in pred_depths:
        # Resize GT 到 pred 大小
        gt_resized = F.interpolate(gt_depth, size=pred.shape[2:], mode='bilinear', align_corners=False)
        if valid_mask is not None:
            mask_resized = F.interpolate(valid_mask.float(), size=pred.shape[2:], mode='nearest')
            loss += weight * F.smooth_l1_loss(pred * mask_resized, gt_resized * mask_resized)
        else:
            loss += weight * F.smooth_l1_loss(pred, gt_resized)
        weight *= 0.7  # 每层衰减
    return loss

def triangulation_3d_loss(pred_points_3d, gt_points_3d):
    """
    平滑 L1 损失计算三角化的3D点与 GT 的差异
    参数:
        pred_points_3d: [B, 3]
        gt_points_3d:   [B, 3]
    """
    return F.smooth_l1_loss(pred_points_3d, gt_points_3d)
```

### 3.3 实验设计

#### 3.3.1 数据集预处理

为适配 DELTAS 网络结构，并提升训练效率与稳定性，我们对 Whole Apartment 数据进行了以下预处理操作：

1. **图像缩放（Resize）**
    将原始分辨率 **480×640** 的图像统一缩放至 **240×320**（即 qVGA 分辨率），以降低计算负担并保持图像的空间一致性。
    
2. **深度图对齐（Depth Alignment）**
    将每帧深度图根据相机内参对齐到对应 RGB 图像坐标系，确保像素级匹配。对无效深度值（如 0 或极大值）进行掩码剔除处理。
    
3. **归一化（Normalization）**
   - **RGB 图像**：将像素值从 `[0, 255]` 缩放至 `[0, 1]`，并按 ImageNet 均值与标准差进行归一化，以适配预训练 ResNet-50 编码器。
   - **深度图**：单位从毫米（mm）转为米（m），并将所有值限制在有效范围内（例如 `0.5m ~ 10m`），超出范围部分设为无效。
   
4. **右移操作（Bit-shifting）**
    深度图采用 `uint16` 格式存储，需要通过右移（如 `>>3`）将其转换为真实深度值。
    
5. **兴趣点筛选与均匀采样**
    对每张图像提取 **SuperPoint** 风格的兴趣点，并设定阈值进行过滤，同时从整图随机采样一定比例的非兴趣点，以增强网络对全图特征的理解力。
    
6. **相机轨迹插帧与帧选取**
    每个训练样本包含 **锚帧 + 2 张参考帧**，通过每隔 N 帧方式选取连续视图，并提取相对位姿矩阵用于后续三角化。
    
    
#### 3.3.2 实验设计

#### 3.3.2.1 参数设计原则（Principles for Sequence Parameter Design）

在 DELTAS 方法的原始论文中，作者在第 4.1 节中明确提出了训练数据构建的基本策略：

```
Three views from a scan at a fixed interval of 20 frames along with the pose and depth information forms a training data point in our method.
```

这意味着，每个训练样本由三帧图像组成，分别为一张**锚帧（anchor frame）\**与其前后各一张\**参考帧（reference frames）**，它们在视频序列中的时间间隔为固定的 20 帧。这种构建方式可以形成一个具有足够视差的三视图几何结构，有利于匹配稳定的特征点、构建三角化关系，并通过多视角融合提升稀疏点的质量。

另外，论文中还提到：

```
These gaps ensure that each set approximately spans similar volumes in 3D space, and any performance improvement emerges from the network better using the available information as opposed to acquiring new information.
```

这段话强调了一个关键的设计原则：**为了公平比较不同序列长度下的模型性能，必须保持每组样本覆盖相似的三维空间体积（3D spatial volume）**。换句话说，在增加视图数量（即 seq_length 更长）的同时，应适当**缩短视图之间的帧间间隔（seq_gap）**，使得不同设置下观察到的空间区域范围基本一致。这样可以确保实验中性能变化的来源是网络在利用信息上的能力提升，而非因“看到了更多新区域”所带来的信息量差异。

例如：

- 若使用 `seq_length = 3`，则 `seq_gap = 20`；
- 若使用 `seq_length = 4`，则可设定 `seq_gap = 15`；
- 若进一步扩展为 `seq_length = 5`，则 `seq_gap = 12`；
- 对于更长的序列，如 `seq_length = 7`，则可以选择 `seq_gap = 10`。

这组参数的选择可以使得图像序列整体覆盖区域相对恒定，从而**更科学地评估视角冗余对模型性能的提升作用**。若忽视这一设计原则，简单增加帧数而不缩短帧间距，可能会带来信息覆盖区域的显著扩展，导致实验结果难以准确归因于模型本身结构或时序建模能力的变化。

因此，我们在本项目的实验设计中，严格遵循论文提出的参数设置策略，通过多组 `(seq_length, seq_gap)` 配对测试模型的表现，旨在深入分析视角冗余、匹配稳定性与稀疏三角化点密度之间的关系，并进一步探讨更优的帧采样策略对稠密深度估计质量的影响。即每个样本由 三帧图像 组成，它们在原始视频序列中间隔 20 帧，用于构建锚帧 + 两参考帧的三视图结构，便于多视图匹配与三角化。

#### 3.3.2.2 参数设计

| 序列长度（seq_length） | 序列间隔（seq_gap） |
| ---------------------- | ------------------- |
| 3                      | 20                  |
| 4                      | 15                  |
| 5                      | 12                  |
| 7                      | 10                  |



#### 3.3.3 实验过程

我们采用了几组不同的seq_length和seq_gap，并计算了Abs、RMSE、RMSE log指标来评估模型表现，得到了相关数据如下：

| 评价指标   | 3-20  | 4-15  | 5-12 | 7-10 |
| ---------- | ------ | ------ | ----- | ----- |
| Abs        | 0.2262 | 0.2235 | 0.2198 | 0.2178 |
| RMSE       | 0.2085 | 0.2076 | 0.2065 | 0.2059 |
| RMSE log   | 0.4913 | 0.4902 | 0.4893 | 0.4886 |

### 3.4 实验结果及分析

#### 3.4 数据分析

整体趋势：更多帧数和更小间隔略有提升

随着序列长度从 3 增加到 7，且帧间隔从 20 减小到 10：

- Abs 几乎保持不变，略有下降；
- RMSE 和 RMSE log 均略有下降，尤其是 RMSE，从 0.2425 → 0.2375；
- 测试时间明显变长

说明：

- 更长的序列（更多图像视角）和更小的间隔（更强的帧间几何约束）确实能提供更多冗余信息，有助于三角化得到更稳健的稀疏点，进而提升稠密深度图的质量；

- 但提升有限，表明 DELTAS 在 seq=3、gap=20 时已有较强性能，对帧数和间隔不敏感，具备较好鲁棒性。

- 应当平衡seq_length（模型表现）与时间成本。

#### 3.4.1  与三角化与Pnp方法对比

以下是Pnp和三角化方法数据：

| 方法 | Abs    | RMSE   | RMSE Log |
| ---- | ------ | ------ | -------- |
| SIFT | 0.3206 | 0.4819 | 0.6600   |
| SURF | 0.3521 | 0.5459 | 0.7469   |
| ORB  | 0.4899 | 0.5951 | 0.9530   |

从表格中可以清晰看出，**DELTAS 方法在深度估计性能上显著优于传统的几何方法（如 SIFT、SURF、ORB 所实现的三角化与 PnP 方法）**，无论是在**绝对相对误差（Abs）**、**均方根误差（RMSE）**还是**对数均方根误差（RMSE log）**等主流指标上，均展现出更高的精度与稳定性。

具体而言，在 DELTAS 方法下，我们尝试了不同的序列长度（seq_length）与帧间间隔（seq_gap）组合，例如：

- **3-20（3帧，间隔20）**配置下，Abs 为 0.2262，RMSE 为 0.2085；
- 当序列扩展到 **7-10（7帧，间隔10）**时，各项指标进一步下降至 Abs = 0.2178，RMSE = 0.2059，RMSE log = 0.4886，说明随着视角冗余的增加，模型能够更充分利用跨帧的几何信息，从而提升深度估计的稳定性和准确性。

对比之下，基于传统特征点匹配与三角化的几何方法表现明显劣于 DELTAS：

- SIFT 的 RMSE 为 **0.4819**，几乎是 DELTAS 的两倍；
- ORB 的 Abs 值高达 **0.4899**，远高于 DELTAS 最差配置的 **0.2262**；
- 即便是效果最好的 SIFT，也未能在任一指标上超过 DELTAS 最低配置（3-20）所达到的性能。

此外，从 RMSE log（对数误差）这一鲁棒性较强的指标来看，传统方法误差普遍偏高（例如 ORB 的 RMSE log 高达 **0.9530**），而 DELTAS 全部配置均控制在 **0.49** 左右，表现出优异的稳定性。这表明 DELTAS 在结构复杂、纹理重复或弱纹理区域的深度估计中具有更强的泛化能力与抗干扰能力。

#### 3.4.2  **与论文结果的对比**

论文中表 3 提到 DELTAS 在 ScanNet 数据集上的表现如下（Abs、RMSE 和 RMSE log）：

![image-20250610103854520](../AppData/Roaming/Typora/typora-user-images/image-20250610103854520.png)

在 whole_apartment 上的数值（Abs≈0.286，RMSE≈0.24，RMSE log≈0.60）比 ScanNet 明显差很多，说明：

- whole_apartment 场景对模型有更大挑战，可能有更多反光、重复纹理、纹理缺失等；

- 说明 DELTAS 可能对新环境（未训练过）泛化性不如 ScanNet 内部测试。

原因分析：

在 DELTAS 中，三角化模块需要已知的相对相机位姿（R, t）来进行 epipolar 约束与 3D 点恢复。这通常依赖外部的 SLAM 或 VIO 系统。然而，实际部署中存在两个问题：
- 1、VIO / SLAM 本身不够准确，会导致三角化误差放大；
- 2、某些设备无法获得准确位姿（如普通单目相机），模型泛化受限。

#### 3.4.3  **是否继续增加帧数值得？**

从 3→4→5→7 帧的过程中，性能略有提高但不是线性，说明：

- DELTAS 方法的匹配与三角化设计是 **匹配少量 interest points + 稀疏三角化 + 稠密填充**；
- 性能提升的边际效益递减，说明超过 5 帧可能性价比低；
- 如果系统资源有限，建议使用 **3–5 帧 + 合理帧间隔（12–15）**，在精度和效率之间取得平衡。

#### 3.4.4  **总结结论**

##### 3.4.4.1 **在 whole_apartment 数据集上性能相比于三角化方法和PnP方法明显效果更加良好**

 在复现 DELTAS 模型并应用于 whole_apartment 数据集的过程中，我们观察到其预测深度图在**整体结构连贯性、边缘保持性**以及**纹理区域的深度稳定性**方面显著优于基于特征点的三角化与PnP方法。传统几何方法受限于特征点提取与匹配的质量，在低纹理区域或存在遮挡的情况下效果不稳定；而 DELTAS 作为基于 Transformer 架构的单目深度估计网络，具备更强的全局上下文建模能力，能有效缓解局部信息不足导致的误差。因此，在多个评价指标上（如 RMSE、AbsRel 等），DELTAS 均取得更优表现，验证了深度学习方法在结构性理解方面的优势。

##### 3.4.4.2 **增加帧数或减小帧间时间间隔对 DELTAS 性能有一定提升，但改善效果有限**

 我们进一步研究了帧间信息对 DELTAS 表现的影响，发现在输入图像采样中适当**增加帧数或减小时间间隔**确实可使预测深度更加连续、精度略有提升。这可能是因为更密集的帧提供了更稳定的外部环境条件，有助于模型维持一致的预测。然而，这种提升在一定程度后趋于饱和，说明 DELTAS 的性能瓶颈更多取决于模型自身的特征提取与表示能力，而非仅依赖输入帧的密度或时序信息。

##### 3.4.4.3 **在 whole_apartment 数据集上的性能下降明显，说明模型在未知环境下的泛化能力仍有待提升**

 尽管 DELTAS 在训练集及类似数据分布上的表现良好，但我们发现其在 whole_apartment 数据集这一**新环境、不同布局、不同光照和材质条件下**的预测结果出现了显著退化。例如，模型在某些空旷房间或玻璃表面区域的深度预测不稳定，甚至存在“浮空”或“塌陷”等结构错误。这表明 DELTAS 仍存在较强的**数据依赖性和领域偏移问题**，其对未知场景的泛化能力较弱，未来可通过引入**领域自适应训练、多环境联合训练、或引导式几何先验融合**等方式进一步改进。

### 3.5 项目反思

在本项目后期，我与林同学聚焦于基于稀疏监督的深度学习方法——**DELTAS 网络**的复现与性能评估工作。这一部分的工作既具挑战性，也带来了颇多启发，尤其在数据预处理规范性、模型结构理解、损失函数设计与实验指标分析方面，我们积累了丰富经验，并深入思考了深度学习方法在实际场景中的适用性与局限性。

DELTAS 的训练过程依赖于稀疏三角测量点，因此数据预处理阶段的每一步都至关重要。我们在实验中充分理解了**图像对齐、深度图右移处理、帧间间隔设计、兴趣点筛选与采样密度控制**等环节对最终三维重建质量的直接影响。例如，若忽略 Whole Apartment 数据集深度图的 bit shift 编码，模型将完全无法学习有效的空间结构。通过对原始论文和开源代码的反复比对，我们逐步建立起了与论文一致的数据流，使后续结果具备了可比性。

为了深入理解 DELTAS 在多帧序列输入下的行为表现，我们系统地设计了 `(seq_length, seq_gap)` 参数组合实验。结果显示，虽然增加帧数与减小帧间距可以略微提升指标（如 RMSE 从 0.2085 降至 0.2059），但该提升并不线性，**性能提升存在边际递减效应**。这使我们意识到，DELTA 的网络核心在于如何高效利用有限帧中的结构冗余，而不是单纯依赖“看更多帧”获得性能跃升。该结论对未来轻量化部署与推理优化具有参考价值。

在评价阶段，我们对比了 DELTAS 与传统三角化和 PnP 方法在 Abs Rel、RMSE 与 RMSE log 三项指标上的表现。虽然 DELTAS 在所有指标上均显著优于传统方法，但我们也观察到模型在极端场景（如玻璃反射、空旷区域）下依然存在误差放大或预测塌陷的情况。通过将误差热图与 Ground Truth 可视化对比，我们更准确地定位了模型局部失效区域，为后续改进提供了方向，如**引入边缘感知损失或自监督姿态优化机制**。

DELTA 的端到端结构为我们提供了从兴趣点检测到深度稠密化的一体化流程，训练与推理过程较为清晰统一。然而，我们也意识到，**过于依赖数据驱动的端到端模型可能在缺乏几何先验时产生结构错误**。因此，未来的工作中或可进一步融合传统三角化几何约束与可学习模块，提升模型对空间关系的建模能力，实现更稳定、更精确的深度估计结果。具体可以由另外的组员在实验改良中实践。
