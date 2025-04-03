import os
import torch
import torch.optim as optim
import torch.nn as nn

from DataObjects import DataLoader
from typing import Optional, Tuple

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                num_epochs: int = 10, lr: float = 0.001,
                device: torch.device = None) -> None:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss: float = 0.0
        train_correct: int = 0
        total_train: int = 0
        
        for batch in train_loader:
            inputs = batch.data.to(device)
            labels = batch.labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels).item()
            total_train += labels.size(0)
        
        avg_train_loss = train_loss / total_train
        train_acc = train_correct / total_train
        
        model.eval()
        val_loss: float = 0.0
        val_correct: int = 0
        total_val: int = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch.data.to(device)
                labels = batch.labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels).item()
                total_val += labels.size(0)
        
        avg_val_loss = val_loss / total_val
        val_acc = val_correct / total_val
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {avg_train_loss:.4f}, Train acc: {train_acc:.4f} | Val loss: {avg_val_loss:.4f}, Val acc: {val_acc:.4f}")

def infer(model: nn.Module, data_loader: DataLoader,
          device: torch.device = None) -> list:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions: list = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch.data.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().tolist())
    
    return predictions


def evaluate(model: nn.Module,
             test_loader: Optional[DataLoader] = None,
             device: Optional[torch.device] = None) -> Tuple[float, float]:

    if test_loader is None:
        test_dir = os.path.join("Data", "Data_converted", "test")
        test_loader = DataLoader(test_dir, batch_size=64, shuffle=True)
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    total_test = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.data.to(device)
            labels = batch.labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            test_correct += torch.sum(preds == labels).item()
            total_test += labels.size(0)
    
    avg_test_loss = test_loss / total_test
    test_acc = test_correct / total_test
    
    print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    return avg_test_loss, test_acc
