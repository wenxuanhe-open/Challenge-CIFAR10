import os
import yaml
import torch
import matplotlib.pyplot as plt

def calculate_accuracy(model, data_loader, device="cpu"):
    """Calculate the accuracy of the model on the given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    model.train()
    return accuracy

def save_model(model, path):
    """Save the model to a specified path."""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load the model from a specified path."""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
    else:
        print(f"No model found at {path}")

def parse_log_file(log_file_path):
    """Parse a log file and return epochs, training loss, and validation error."""
    epochs = []
    train_losses = []
    val_losses = []
    val_errors = []

    with open(log_file_path, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            epoch, train_loss, val_loss, val_error = line.strip().split(',')
            epochs.append(int(epoch))
            train_losses.append(float(train_loss))
            val_losses.append(float(val_loss))
            val_errors.append(float(val_error))

    return epochs, train_losses, val_losses, val_errors

def load_config(config_path="configs/config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config['batch_size'], config['learning_rate']

def plot_training_and_validation(log_files, model_names, config_path="configs/config.yaml"):
    """Plot training loss and validation error curves for multiple models and save the plots."""
    # 使用接近的颜色方案
    color_palette = ["#00BFFF", "#DC143C", "#32CD32", "#FFA500"]  # 浅蓝色, 红色, 绿色, 橙色

    batch_size, learning_rate = load_config(config_path)

    # 确保保存图像的目录存在
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # 生成文件名后缀
    model_suffix = "_".join([name.replace(" ", "") for name in model_names])

    # 绘制验证错误曲线
    plt.figure(figsize=(12, 6))
    for i, log_file in enumerate(log_files):
        if os.path.exists(log_file):
            epochs, _, _, val_errors = parse_log_file(log_file)
            plt.plot(epochs, val_errors, label=f"{model_names[i]} Validation Error", color=color_palette[i % len(color_palette)])
        else:
            print(f"Log file {log_file} not found.")

    # 标注超参数信息
    plt.text(0.5, 0.95, f"Batch Size: {batch_size}, Learning Rate: {learning_rate}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.xlabel("Epochs")
    plt.ylabel("Validation Error (%)")
    plt.title("Validation Error Comparison")
    plt.legend()
    plt.grid(True)
    # 保存图像
    validation_plot_path = os.path.join(plots_dir, f"validation_error_comparison_{model_suffix}.png")
    plt.savefig(validation_plot_path)
    print(f"Validation error plot saved to {validation_plot_path}")
    plt.show()

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(12, 6))
    for i, log_file in enumerate(log_files):
        if os.path.exists(log_file):
            epochs, train_losses, val_losses, _ = parse_log_file(log_file)
            plt.plot(epochs, train_losses, label=f"{model_names[i]} Training Loss", linestyle='--', color=color_palette[i % len(color_palette)])
            plt.plot(epochs, val_losses, label=f"{model_names[i]} Validation Loss", color=color_palette[i % len(color_palette)])
        else:
            print(f"Log file {log_file} not found.")

    # 标注超参数信息
    plt.text(0.5, 0.95, f"Batch Size: {batch_size}, Learning Rate: {learning_rate}",
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Comparison")
    plt.legend()
    plt.grid(True)
    # 保存图像
    loss_plot_path = os.path.join(plots_dir, f"training_and_validation_loss_comparison_{model_suffix}.png")
    plt.savefig(loss_plot_path)
    print(f"Training and validation loss plot saved to {loss_plot_path}")
    plt.show()

# 1: 绘制 ResNet18 和 ResNet34 的图表
plot_training_and_validation(["log/ResNet18_log.txt", "log/ResNet34_log.txt"], ["ResNet18", "ResNet34"], config_path="configs/config.yaml")

# 2: 绘制 Plain18 和 Plain34 的图表
plot_training_and_validation(["log/Plain18_log.txt", "log/Plain34_log.txt"], ["Plain18", "Plain34"], config_path="configs/config.yaml")
