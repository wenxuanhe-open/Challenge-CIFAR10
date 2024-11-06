# Challenge CIFAR-10 🚀

本项目旨在挑战 CIFAR-10 数据集，复现经典深度残差网络（ResNet）及其改进（如多尺度特征提取）的实验，同时引入现代 Vision Transformer (ViT) 模型对比实验，验证不同模型在 CIFAR-10 数据集上的性能，并尝试 SOTA 模型以探索其极限表现。通过系统的实验对比，探讨深度学习模型在小型数据集上的适用性及优化策略。

---

## 项目结构 📂

```
.
├── checkpoints                # 训练过程中保存的模型权重
├── configs                    # 配置文件
├── data                       # 数据集文件夹
├── log                        # 训练日志
│   ├── custom_vit_log.txt         # 自定义 ViT 模型日志
│   ├── Plain18_log.txt            # Plain18 日志
│   ├── Plain34_log.txt            # Plain34 日志
│   ├── ResNet18_log.txt           # ResNet18 日志
│   ├── ResNet18WithMS_log.txt     # ResNet18 with MS 日志
│   ├── ResNet34_log.txt           # ResNet34 日志
│   ├── sota_log.txt               # SOTA 模型日志
│   ├── vit_tiny_log.txt           # ViT-tiny (预训练) 日志
│   └── vitn_tiny_log.txt          # ViT-tiny (无预训练) 日志
├── models                       # 模型定义文件夹
├── paper                        # 相关论文
├── plots                        # 实验生成的对比图
│   ├── training_and_validation_loss_comparison_Plain18_Plain34.png
│   ├── training_and_validation_loss_comparison_ResNet18_ResNet34.png
│   ├── training_and_validation_loss_comparison_vit-tiny-p_vit-tiny-n_vit-tiny-c.png
│   ├── training_and_validation_loss_ResNet18WithMS.png
│   ├── validation_error_comparison_Plain18_Plain34.png
│   ├── validation_error_comparison_ResNet18_ResNet34.png
│   ├── validation_error_comparison_vit-tiny-p_vit-tiny-n_vit-tiny-c.png
│   ├── validation_error_ResNet18WithMS.png
│   └── validation_error_Vit-tiny-Pre-Training.png
├── utils                        # 辅助工具
├── train.py                     # ResNet 和 Plain 网络训练脚本
├── train_vit.py                 # ViT 网络训练脚本
├── custom_train_vit.py          # 自定义 ViT 模型训练脚本
└── SOTA.py                      # SOTA 模型尝试脚本
```

---

## 配置文件 🛠️

在 `configs/` 目录下包含以下配置文件：

1. **ResNet 和 Plain 网络** (`config.yaml`)：
   ```yaml
   epochs: 150
   batch_size: 256
   learning_rate: 0.1
   ```

2. **ViT 模型** (`config_vit.yaml`)：
   ```yaml
   model: vit_tiny_patch16_224
   pretrained: true
   epochs: 150
   batch_size: 512
   learning_rate: 0.0001
   ```

3. **SOTA 模型** (`config_sota.yaml`)：
   ```yaml
   model: eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
   pretrained: true
   epochs: 50
   batch_size: 256
   learning_rate: 0.0001
   ```

---

## 实验启动 🚦

### ResNet 和 Plain 网络

```bash
# 训练 Plain18 网络
python train.py --model Plain18 --device cuda:3

# 训练 ResNet18 网络
python train.py --model ResNet18 --device cuda:6

# 训练 ResNet18 with MS 网络
python train_ms.py --model ResNet18WithMS --device cuda:7
```

### ViT 网络

```bash
# ViT (预训练权重)
python train_vit.py --config configs/config_vit.yaml --pretrained True

# ViT (无预训练权重)
python train_vit.py --config configs/config_vit.yaml --pretrained False

# 自定义 ViT 模型
python custom_train_vit.py
```

### SOTA 模型尝试

```bash
python SOTA.py --config configs/config_sota.yaml
```

---

## 实验结果与分析 📊

### 1. ResNet 和 Plain 网络实验

#### 验证错误率排名：
1. **ResNet18 with MS** - 引入多尺度特征提取后的网络效果最优，验证错误率最低。
2. **ResNet34** - 深层 ResNet 表现优于浅层 ResNet18。
3. **ResNet18** - 浅层 ResNet 网络表现较好，但低于 ResNet34 和 ResNet18 with MS。
4. **Plain18** - 浅层 Plain 网络的验证错误率低于 Plain34。
5. **Plain34** - 深层 Plain 网络表现最差，验证了深度网络的退化问题。

#### 图表与分析：
- **Plain18 与 Plain34**：
  - **训练与验证损失曲线**：
    ![Plain18 vs Plain34 Loss](plots/training_and_validation_loss_comparison_Plain18_Plain34.png)
  - **验证错误率曲线**：
    ![Plain18 vs Plain34 Validation Error](plots/validation_error_comparison_Plain18_Plain34.png)

    **分析**：
    - Plain 网络在深层结构中（Plain34）表现出明显的退化问题，即验证错误率随深度增加而升高，训练误差逐步降低但验证误差未改善。
    - 这验证了 ResNet 论文中提到的深度退化问题，强调了残差连接的重要性。

- **ResNet18 与 ResNet34**：
  - **训练与验证损失曲线**：
    ![ResNet18 vs ResNet34 Loss](plots/training_and_validation_loss_comparison_ResNet18_ResNet34.png)
  - **验证错误率曲线**：
    ![ResNet18 vs ResNet34 Validation Error](plots/validation_error_comparison_ResNet18_ResNet34.png)

    **分析**：
    - ResNet 的引入有效缓解了深度退化问题，ResNet34 的性能优于 ResNet18，验证了更深层网络（结合残差连接）的优势。

- **ResNet18 with MS**：
  - **训练与验证损失曲线**：
    ![ResNet18 with MS Loss](plots/training_and_validation_loss_ResNet18WithMS.png)
  - **验证错误率曲线**：
    ![ResNet18 with MS Validation Error](plots/validation_error_ResNet18WithMS.png)

    **分析**：
    - 引入多尺度特征提取策略后，ResNet18 的验证错误率进一步降低，其性能接近甚至超过了 ResNet34。
    - 多尺度特征提取增强了浅层网络的表达能力，使其在 CIFAR-10 等小型数据集上表现尤为出色。

---

### 2. Vision Transformer (ViT) 实验

#### 验证错误率排名：
1. **ViT-tiny (预训练)** - 收敛速度最快，验证错误率仅 **2.68%**，性能最佳。
2. **ViT-tiny (无预训练)** - 验证错误率 **24.39%**，比预训练模型显著更高，训练时间长且收敛较慢。
3. **自定义 ViT** - 验证错误率为 **31.03%**，性能不及标准 ViT，但为验证新架构提供了实验平台。

#### 图表与详细分析：

- **ViT-tiny (预训练) 与 ViT-tiny (无预训练)**：
  - **训练与验证损失曲线**：
    ![ViT Loss Comparison](plots/training_and_validation_loss_comparison_vit-tiny-p_vit-tiny-n_vit-tiny-c.png)
  - **验证错误率曲线**：
    ![Validation Error Comparison](plots/validation_error_comparison_vit-tiny-p_vit-tiny-n_vit-tiny-c.png)

  **详细分析**：
  1. **预训练模型 (ViT-tiny-p)**：
     - **验证错误率最低**：快速下降到 **2.68%**，验证了迁移学习的有效性。
     - **收敛速度快**：在不到 10 个 epoch 内，模型达到稳定状态。
     - **现象与结论**：预训练权重在小型数据集（CIFAR-10）上显著缩短了训练时间并提升了模型性能。

  2. **无预训练模型 (ViT-tiny-n)**：
     - **收敛速度较慢**：训练和验证损失的下降曲线均较平缓，表明模型对数据集适应需更长时间。
     - **过拟合问题显著**：训练误差快速下降，但验证误差维持较高水平，表明模型在数据规模有限的情况下难以充分泛化。

  3. **自定义 ViT 模型 (ViT-tiny-c)**：
     - **训练和验证损失曲线**：
       ![Custom ViT Loss](plots/training_and_validation_loss_comparison_vit-tiny-p_vit-tiny-n_vit-tiny-c.png)
     - **分析**：验证错误率维持在 **31.03%**，性能不及标准 ViT，但由于架构简洁，适合快速验证实验假设。

---

#### **ViT-tiny (预训练) 的单独分析**

- **训练与验证损失曲线**：
  ![ViT Pre-Trained Loss](plots/training_and_validation_loss_Vit-tiny-Pre-Training.png)
  
  **分析**：
  - **损失曲线的变化**：在训练初期，训练和验证损失迅速下降，并在短时间内趋于平稳，表明模型训练稳定且收敛速度快。
  - **迁移学习的效果**：预训练权重在小型数据集上实现了良好的迁移效果，使得训练误差与验证误差非常接近，无明显的过拟合现象。
  - **总结**：预训练显著降低了 ViT-tiny 的训练时间，同时提高了模型的泛化能力，是小型数据集上的关键优化手段。

---

### 3. SOTA 模型实验

#### 模型性能与复杂度对比：

| 模型名称                                        | 输入尺寸 | ImageNet Top-1 | 参数量（M） | 推理速度（样本/秒） |
|------------------------------------------------|----------|----------------|-------------|---------------------|
| **eva02_large_patch14_448.mim_m38m_ft_in22k_in1k** | 448      | 90.05%         | 305.08      | 102.50              |
| **eva_giant_patch14_336.clip_ft_in1k**          | 336      | 89.46%         | 1013.01     | 100.67              |
| **convnext_xxlarge.clip_laion2b_soup_ft_in1k**  | 256      | 88.62%         | 846.47      | **281.32**          |

#### 推理复杂度与效率对比：

| 模型名称                                        | 输入尺寸 | 推理步骤时间（ms） | 推理批大小 | GMACs | 推理 TFLOPs | 单样本推理时间（μs） |
|------------------------------------------------|----------|--------------------|------------|-------|-------------|-----------------------|
| **eva02_large_patch14_448.mim_m38m_ft_in22k_in1k** | 448      | 4995.17           | 512        | 362.33| 74.28       | 9756.10              |
| **eva_giant_patch14_336.clip_ft_in1k**          | 336      | 5085.92           | 512        | 620.64| 124.96      | 9933.45              |
| **convnext_xxlarge.clip_laion2b_soup_ft_in1k**  | 256      | 909.97            | 256        | 198.09| 111.45      | **3554.67**          |

#### **详细分析**：
1. **架构优势与限制**：
   - **`eva02_large_patch14_448.mim_m38m_ft_in22k_in1k`** 是最先进的模型之一，凭借 **90.05%** 的 ImageNet Top-1 准确率展示了卓越性能。然而，模型参数量虽适中（305.08M），但训练和推理资源需求较高，单样本推理时间为 **9756.10 μs**。
   - **`convnext_xxlarge.clip_laion2b_soup_ft_in1k`** 在推理效率上最优，单样本推理时间仅 **3554.67 μs**，适用于需要高吞吐量的场景。

2. **性能与效率的平衡**：
   - 输入尺寸越大（如 448x448），模型的特征提取能力越强，验证准确率更高；但推理复杂度显著增加。
   - SOTA 模型在小型数据集（如 CIFAR-10）上性能未必完全展现，但在大规模数据（如 ImageNet）上具有明显优势。

3. **未来方向**：
   - **轻量化技术**：通过稀疏注意力等方法，优化 SOTA 模型在小型数据集上的效率。
   - **迁移学习与多任务训练**：提升模型在不同分布上的鲁棒性与适应性。

---

## Recap ✨

### KeyPoints：

1. **残差网络的有效性**：
   - ResNet 系列（特别是引入多尺度特征提取后的 ResNet18）在缓解深层网络退化问题上具有显著优势。
   - **多尺度特征提取** 提升了浅层网络的性能，是一种高效的优化策略。

2. **Vision Transformer 的表现**：
   - **预训练权重是小型数据集上的关键**：显著缩短了训练时间，同时极大提升了模型性能。
   - 无预训练的 ViT 收敛较慢，泛化性能有限。

3. **SOTA 模型的潜力**：
   - 在大规模数据集上表现卓越，但训练和推理资源需求高。
   - 通过优化计算复杂度，结合迁移学习技术，有望在小型数据集上展现更好的性能。

---


