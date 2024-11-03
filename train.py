import os
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models import Plain18, Plain34, ResNet18, ResNet34
from utils.utils import calculate_accuracy
from tqdm import tqdm
import argparse

# 加载配置文件
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# 从配置文件读取参数
EPOCHS = config['epochs']
BATCH_SIZE = config['batch_size']
LEARNING_RATE = config['learning_rate']
LOG_DIR = "log"  # 直接指定日志目录
CHECKPOINT_DIR = "checkpoints"  # 模型检查点保存目录

# 数据加载
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# 确保日志和检查点文件夹存在
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 加载检查点
def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")
    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from scratch.")
    return start_epoch

# 保存检查点
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved for epoch {epoch + 1}.")

# 训练函数
def train(model, criterion, optimizer, scheduler, num_epochs, model_name, start_epoch):
    log_file_path = os.path.join(LOG_DIR, f"{model_name}_log.txt")
    print(f"Logging training results to {log_file_path}")

    with open(log_file_path, "w" if start_epoch == 0 else "a") as log_file:
        if start_epoch == 0:
            log_file.write("Epoch, Training Loss, Validation Loss, Validation Error (%)\n")

        for epoch in range(start_epoch, num_epochs):
            model.train()
            train_loss = 0

            # 使用 tqdm 显示训练进度
            with tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Training") as pbar:
                for inputs, targets in pbar:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
            train_loss /= len(train_loader)

            # 更新学习率
            scheduler.step()

            # 验证过程
            val_loss = 0
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                with tqdm(test_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Validation") as pbar:
                    for inputs, targets in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                        pbar.set_postfix(loss=loss.item())
            val_loss /= len(test_loader)
            val_error = 100 * (1 - correct / total)

            # 打印当前轮次的结果
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Error: {val_error:.2f}%")

            # 记录到日志文件
            log_file.write(f"{epoch+1}, {train_loss:.4f}, {val_loss:.4f}, {val_error:.2f}\n")

            # 保存检查点
            save_checkpoint(model, optimizer, epoch, os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth"))


# 设置命令行参数
parser = argparse.ArgumentParser(description="Training Script for Multiple Models")
parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (default: cuda:0)')
parser.add_argument('--model', type=str, default="Plain18", choices=["Plain18", "Plain34", "ResNet18", "ResNet34"],
                    help='Model to train (default: Plain18)')
args = parser.parse_args()

# 主程序
if __name__ == '__main__':
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dict = {
        "Plain18": Plain18,
        "Plain34": Plain34,
        "ResNet18": ResNet18,
        "ResNet34": ResNet34
    }

    model_name = args.model
    model_class = model_dict.get(model_name)
    if model_class is None:
        raise ValueError(f"Model {model_name} is not recognized.")
    
    model = model_class().to(device)
    print(f"\nTraining model: {model_name}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    
    # 学习率调度器（每 50 个 epoch 将学习率减少为原来的 1/10）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    # 加载检查点并开始训练
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{model_name}_checkpoint.pth")
    start_epoch = load_checkpoint(model, optimizer, checkpoint_path)

    # 开始训练
    train(model, criterion, optimizer, scheduler, EPOCHS, model_name, start_epoch)
