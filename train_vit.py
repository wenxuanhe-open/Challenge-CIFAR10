import os
import yaml  # 用于加载 YAML 配置文件
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import time
from tqdm import tqdm  # 导入 tqdm 显示进度条
import warnings  # 用于忽略警告

warnings.filterwarnings("ignore", category=UserWarning)

# 读取 YAML 配置文件
def load_config(config_path='configs/config_vit.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 主函数
def main():
    # 加载配置
    config = load_config()

    # 提取配置参数
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    # 日志文件路径
    log_file_path = 'log/vitn_tiny_log.txt'

    # 确保日志目录存在
    if not os.path.exists('log'):
        os.makedirs('log')

    # 设备配置
    device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整到 ViT 需要的输入尺寸
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集加载
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 加载预训练 ViT 模型
    print("开始加载预训练模型...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False)
    print("预训练模型加载成功!")
    
    model.head = nn.Linear(model.head.in_features, 10)  # 修改为 CIFAR-10 的类别数
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证过程
    with open(log_file_path, 'w') as log_file:
        log_file.write('Epoch, Training Loss, Validation Loss, Validation Error (%)\n')

    for epoch in range(1, num_epochs + 1):
        # 训练模式
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Training Epoch {epoch}/{num_epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_training_loss = running_loss / len(train_loader)

        # 验证模式
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Validation Epoch {epoch}/{num_epochs}'):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_error = 100 * (1 - correct / total)

        # 打印和记录日志
        epoch_log = f'{epoch}, {avg_training_loss:.4f}, {avg_val_loss:.4f}, {val_error:.2f}\n'
        print(f'Epoch {epoch}, Training Loss: {avg_training_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Error: {val_error:.2f}%')
        
        # 写入日志文件
        with open(log_file_path, 'a') as log_file:
            log_file.write(epoch_log)

    print(f'训练结束，日志保存在 {log_file_path}')

# 运行主函数
if __name__ == "__main__":
    main()

