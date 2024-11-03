import os
import torch
import matplotlib.pyplot as plt

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

def plot_training_curves(train_errors, val_errors, iterations, model_name="Model"):
    """Plot training and validation error curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, train_errors, label=f"{model_name} Training Error", color="blue")
    plt.plot(iterations, val_errors, label=f"{model_name} Validation Error", color="red")
    plt.xlabel("Iterations")
    plt.ylabel("Error (%)")
    plt.title(f"{model_name} Training and Validation Error")
    plt.legend()
    plt.grid(True)
    plt.show()

def record_error(error_list, current_error):
    """Append the current error to the error list."""
    error_list.append(current_error)

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
