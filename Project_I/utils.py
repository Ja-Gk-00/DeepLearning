from Architectures.SimpleCNN import SimpleCNN
import pickle


def load_from_pickle(file_path: str) -> SimpleCNN:
    with open(file_path, 'rb') as f:
        model: SimpleCNN = pickle.load(f)
    print(f"Model loaded from {file_path}")
    return model