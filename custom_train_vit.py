import os
import yaml  # 用于加载 YAML 配置文件
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # 用于显示进度条
import warnings  # 用于忽略警告

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# 读取 YAML 配置文件
def load_config(config_path='configs/config_vit.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# 自定义 Patch Embedding
class CustomPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super(CustomPatchEmbedding, self).__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# 自定义多头自注意力层
class CustomMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CustomMultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, num_tokens, embed_dim = x.shape
        qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attention = (q @ k.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        return self.fc_out(out)

# 自定义 Transformer Block
class CustomTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(CustomTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = CustomMultiHeadSelfAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

# 自定义 Vision Transformer
class CustomVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_dim=768, dropout=0.1):
        super(CustomVisionTransformer, self).__init__()
        self.patch_embed = CustomPatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList([
            CustomTransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        return self.head(cls_token_final)

# 主函数
def main():
    # 加载配置
    config = load_config()

    # 提取配置参数
    num_epochs = config['epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']

    # 日志文件路径
    log_file_path = 'log/custom_vit_log.txt'

    # 确保日志目录存在
    if not os.path.exists('log'):
        os.makedirs('log')

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 数据集加载
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 初始化自定义 ViT 模型
    model = CustomVisionTransformer(img_size=224, patch_size=16, num_classes=10, embed_dim=192, depth=12, num_heads=3, mlp_dim=768)
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
