# 深度残差网络实验及多尺度改进 🚀

本项目旨在复现何凯明等人在 ResNet 论文中的实验，验证残差网络的有效性，以及通过在 ResNet18 中引入多尺度特征提取（MS）进行创新性改进，以提升模型性能。实验验证了残差网络在深度网络中的有效性，并展示了创新的多尺度特征提取如何进一步优化模型表现。

## 项目结构 📂

```
.
├── checkpoints                # 训练过程中保存的模型权重
│   ├── Plain18_checkpoint.pth
│   ├── Plain34_checkpoint.pth
│   ├── ResNet18_checkpoint.pth
│   ├── ResNet34_checkpoint.pth
│   └── ResNet18WithMS_checkpoint.pth
├── configs
│   └── config.yaml            # 配置文件，包含训练超参数设置
├── data                       # 数据集文件夹
│   ├── cifar-10-batches-py
│   └── cifar-10-python.tar.gz
├── log                        # 训练日志，记录每个 epoch 的损失和验证错误率
│   ├── Plain18_log.txt
│   ├── Plain34_log.txt
│   ├── ResNet18_log.txt
│   ├── ResNet34_log.txt
│   └── ResNet18WithMS_log.txt
├── models                     # 模型定义文件夹
│   ├── __init__.py
│   ├── plainnet.py            # Plain 网络定义
│   ├── resnet.py              # ResNet 网络定义
│   └── resnet_multiscale.py   # ResNet18 with MS 网络定义
├── paper
│   └── 1512.03385v1.pdf       # Reference Paper
├── plots                      # 实验生成的对比图
│   ├── training_and_validation_loss_comparison_Plain18_Plain34.png
│   ├── training_and_validation_loss_comparison_ResNet18_ResNet34.png
│   ├── training_and_validation_loss_ResNet18WithMS.png
│   ├── validation_error_comparison_Plain18_Plain34.png
│   ├── validation_error_comparison_ResNet18_ResNet34.png
│   └── validation_error_ResNet18WithMS.png
├── utils                      # 辅助函数文件夹
│   └── utils.py
├── train.py                   # ResNet 和 Plain 网络训练脚本
├── train_ms.py                # ResNet18 with MS 网络训练脚本
└── .gitignore
```

## 配置文件 🛠️

在 `configs/config.yaml` 中配置了训练的超参数：

```yaml
epochs: 150
batch_size: 256
learning_rate: 0.1
```

## 运行环境 ⚙️

- **硬件平台**：4 台 V800 服务器
- **软件依赖**：
  - Python 3.x
  - PyTorch
  - torchvision
  - tqdm
  - matplotlib
  - PyYAML

## 实验启动 🚦

在不同设备上运行以下命令来启动训练，测算得到每一个训练环节需3G显存：

```bash
# 训练 Plain18 网络
python train.py --model Plain18 --device cuda:3

# 训练 Plain34 网络
python train.py --model Plain34 --device cuda:4

# 训练 ResNet18 网络
python train.py --model ResNet18 --device cuda:6

# 训练 ResNet34 网络
python train.py --model ResNet34 --device cuda:7

# 训练 ResNet18 with MS 网络
python train_ms.py --model ResNet18WithMS --device cuda:7
```

## 实验结果与分析 📊

### 1. 实验结果排名与分析
根据验证错误率和损失曲线，模型的排名如下：

**模型验证错误率排名（从优到差）**：
1. **ResNet18 with MS**
2. **ResNet34**
3. **ResNet18**
4. **Plain18**
5. **Plain34**

**详细分析**：
- **Plain 网络 vs ResNet 网络**：
  - **Plain18 与 Plain34**：
    ![Plain18 vs Plain34 Loss](plots/training_and_validation_loss_comparison_Plain18_Plain34.png)
    ![Plain18 vs Plain34 Validation Error](plots/validation_error_comparison_Plain18_Plain34.png)
    
    - **分析**：Plain 网络（无残差连接）在深层网络中表现出明显的“退化问题”，即 Plain34 的验证错误率高于 Plain18。这验证了论文中提到的，当网络深度增加时，网络会面临训练困难和性能下降。

  - **ResNet18 与 ResNet34**：
    ![ResNet18 vs ResNet34 Loss](plots/training_and_validation_loss_comparison_ResNet18_ResNet34.png)
    ![ResNet18 vs ResNet34 Validation Error](plots/validation_error_comparison_ResNet18_ResNet34.png)
    
    - **分析**：ResNet 网络的引入解决了深度网络的退化问题。ResNet34 比 ResNet18 的验证错误率低，符合 ResNet 论文的结论，即增加网络深度（结合残差连接）可以有效提高性能。

### 2. 多尺度改进的效果 🚀
- **ResNet18 with MS**：
  ![ResNet18 with MS Loss](plots/training_and_validation_loss_ResNet18WithMS.png)
  ![ResNet18 with MS Validation Error](plots/validation_error_ResNet18WithMS.png)
  
  **引入Multiple Scale改进**：
  - 引入多尺度特征提取模块后，**ResNet18 with MS** 显示出明显的性能提升，其验证错误率低于 ResNet34。
  - **多尺度特征提取**增强了浅层网络的表达能力，使得 ResNet18 的性能接近甚至超过了 ResNet34，验证了特征提取策略对模型性能的影响。
  - 这种改进展示了在保持网络结构简洁的前提下，如何通过引入新的特征提取策略来进一步优化模型。

### 3. 退化问题与残差网络的有效性 🔧
- **退化问题**：在深层网络中（如 Plain34），即使训练损失下降，验证错误率也会增加。这种现象在使用无残差连接的网络中尤为明显。
- **残差网络的有效性**：通过在 ResNet 中使用跳跃连接，网络能够学习恒等映射，从而缓解退化问题。ResNet18 和 ResNet34 的结果证明了残差连接可以提高训练和验证的稳定性。

## 总结 ✨
我们的实验验证了何凯明等人在 ResNet 论文中提出的核心观点，即**残差连接能够缓解深层网络的退化问题**。通过引入创新性的多尺度特征提取模块，我们展示了 ResNet18 在浅层结构下也能达到甚至超过 ResNet34 的性能，证明了新的特征提取策略对提高模型性能的有效性。

本项目为网络设计提供了新的思路，强调了在保持网络简洁的同时，通过改进特征提取策略来提高性能的重要性。🔬

