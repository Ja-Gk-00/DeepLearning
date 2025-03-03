from Architectures.SimpleCNN import SimpleCNN
import pickle
import torch
from torch import nn

def load_from_pickle(file_path: str) -> SimpleCNN:
    with open(file_path, 'rb') as f:
        model: SimpleCNN = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model

def save_model(model: nn.Module, path: str) -> None:
    torch.save(model, path)
    print(f"Model saved successfully at {path}")

def load_model(path: str) -> nn.Module:
    model = torch.load(path)
    print(f"Model loaded successfully from {path}")
    return model
