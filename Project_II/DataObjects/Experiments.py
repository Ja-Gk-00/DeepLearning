from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from DataLoader import DataLoader
from ..Architectures.Transformer.CustomTransformer import CustomTransformerModel
from ..Architectures.Transformer.GPT2 import GPT2FineTuner
from ..Architectures.CNN.SimpleCNN import SimpleCNN
from ..Architectures.Statistical.GMM import GMMClassifier
from ContextSaver import Capturing

class Experiment(ABC):
    @abstractmethod
    def load_parameters(self, params: Dict[str, Any]) -> None:
        """Load experiment settings."""
        pass

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """Execute the experiment end-to-end and return results."""
        pass


class BaseExperiment(Experiment):
    def __init__(self) -> None:
        self.params: Dict[str, Any] = {}

    def load_parameters(self, params: Dict[str, Any]) -> None:
        self.params = params

    def run(self) -> Dict[str, Any]:
        raise NotImplementedError("Please implement run() in subclasses.")


class TransformerExperiment(BaseExperiment):
    """
    Experiment for CustomTransformerModel.

    Parameters in `params` dict:
    - train_dir (str): path to training data directory.
    - val_dir (Optional[str]): path to validation data directory (optional).
    - batch_size (int): batch size for DataLoader (default: 32).
    - epochs (int): number of training epochs (default: 10).
    - sr (int): sampling rate for DataLoader (default: 16000).
    - data_type (str): type of data loader, e.g., 'spectrogram', 'mfcc' (default: 'spectrogram').
    - save_path ([str, Path]): where to save text output of training and evaluation.
    - model_kwargs (dict): keyword arguments forwarded to CustomTransformerModel.
    """
    def run(self) -> Dict[str, Any]:
        # Load parameters
        train_dir = self.params['train_dir']
        val_dir = self.params.get('val_dir')
        test_dir = self.params.get('test_dir')
        batch_size = self.params.get('batch_size', 32)
        epochs = self.params.get('epochs', 10)
        sr = self.params.get('sr', 16000)
        data_type = self.params.get('data_type', 'spectrogram')
        model_kwargs = self.params.get('model_kwargs', {})

        # Prepare loaders
        train_loader = DataLoader(train_dir, data_type, batch_size=batch_size, shuffle=True, sr=sr)
        test_loader = DataLoader(test_dir, data_type, batch_size=batch_size, shuffle=True, sr=sr)
        val_loader = None
        if val_dir:
            val_loader = DataLoader(val_dir, data_type, batch_size=batch_size, shuffle=False, sr=sr)

        sample = next(iter(train_loader)).data[0]
        C, *rest = sample.shape
        input_dim = C * rest[0] if len(rest) > 1 else C
        model = CustomTransformerModel(input_dim=input_dim, **model_kwargs)

        with Capturing(self.params.get('save_path', "log.txt")) as logs:
            model.train(train_loader, epochs=epochs, val_loader=val_loader)
            results = model.evaluate(test_loader) if test_loader else {}
            if 'summary' in results:
                results['score'] = results['summary'].get('accuracy')
        return results


class GPT2Experiment(BaseExperiment):
    """
    Experiment for GPT2FineTuner.

    Parameters in `params` dict:
    - train_dir (str): path to training data directory.
    - val_dir (Optional[str]): path to validation data directory (optional).
    - batch_size (int): batch size for DataLoader (default: 32).
    - epochs (int): number of training epochs (default: 5).
    - audio_dim (int): dimension of audio feature vectors fed to GPT2.
    - save_path ([str, Path]): where to save text output of training and evaluation.
    - model_kwargs (dict): keyword arguments forwarded to GPT2FineTuner.
    """
    def run(self) -> Dict[str, Any]:
        # Load parameters
        train_dir = self.params['train_dir']
        val_dir = self.params.get('val_dir')
        test_dir = self.params.get('test_dir')
        batch_size = self.params.get('batch_size', 32)
        epochs = self.params.get('epochs', 5)
        audio_dim = self.params['audio_dim']
        model_kwargs = self.params.get('model_kwargs', {})

        # Prepare loaders
        train_loader = DataLoader(train_dir, 'raw', batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dir, 'raw', batch_size=batch_size, shuffle=True)
        val_loader = None
        if val_dir:
            val_loader = DataLoader(val_dir, 'raw', batch_size=batch_size, shuffle=False)

        # Instantiate model
        model = GPT2FineTuner(audio_dim=audio_dim, **model_kwargs)
        # Train & optionally validate
        with Capturing(self.params.get('save_path', "log.txt")) as logs:
            model.train(train_loader, epochs=epochs, val_loader=val_loader)
            results = model.evaluate(test_loader) if test_loader else {}
            # Ray Tune score
            if 'summary' in results:
                results['score'] = results['summary'].get('accuracy')
        return results


class CNNExperiment(BaseExperiment):
    """
    Experiment for SimpleCNN on spectrogram data.
    Parameters in `params` dict:
    - train_dir (str): path to spectrogram training directory.
    - val_dir (Optional[str]): path to spectrogram validation directory (optional).
    - batch_size (int): batch size for DataLoader (default: 32).
    - epochs (int): number of training epochs (default: 10).
    - in_channels (int): number of input channels for the CNN (default: 1).
    - save_path ([str, Path]): where to save text output of training and evaluation.
    - model_kwargs (dict): keyword arguments forwarded to SimpleCNN.
    """
    def run(self) -> Dict[str, Any]:
        # Load parameters
        train_dir = self.params['train_dir']
        val_dir = self.params.get('val_dir')
        test_dir = self.params.get('test_dir')
        batch_size = self.params.get('batch_size', 32)
        epochs = self.params.get('epochs', 10)
        in_channels = self.params.get('in_channels', 1)
        model_kwargs = self.params.get('model_kwargs', {})

        # Prepare loaders
        train_loader = DataLoader(train_dir, 'spectrogram', batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dir, 'spectrogram', batch_size=batch_size, shuffle=True)
        val_loader = None
        if val_dir:
            val_loader = DataLoader(val_dir, 'spectrogram', batch_size=batch_size, shuffle=False)

        # Instantiate model
        model = SimpleCNN(in_channels=in_channels, **model_kwargs)
        # Train & optionally validate
        with Capturing(self.params.get('save_path', "log.txt")) as logs:
            model.train(train_loader, epochs=epochs, val_loader=val_loader)
            results = model.evaluate(test_loader) if test_loader else {}
            # Ray Tune score
            if 'summary' in results:
                results['score'] = results['summary'].get('accuracy')
        return results


class GMMExperiment(BaseExperiment):
    """
    Experiment for GMMClassifier on FFT features.

    Parameters in `params` dict:
    - train_dir (str): path to FFT training directory.
    - val_dir (Optional[str]): path to FFT validation directory (optional).
    - batch_size (int): batch size for DataLoader (default: 64).
    - save_path ([str, Path]): where to save text output of training and evaluation.
    - model_kwargs (dict): keyword arguments forwarded to GMMClassifier.
    """
    def run(self) -> Dict[str, Any]:
        # Load parameters
        train_dir = self.params['train_dir']
        val_dir = self.params.get('val_dir')
        test_dir = self.params.get('test_dir')
        batch_size = self.params.get('batch_size', 64)
        model_kwargs = self.params.get('model_kwargs', {})

        # Prepare loaders
        train_loader = DataLoader(train_dir, 'fft', batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dir, 'fft', batch_size=batch_size, shuffle=True)
        val_loader = None
        if val_dir:
            val_loader = DataLoader(val_dir, 'fft', batch_size=batch_size, shuffle=False)

        # Instantiate and train
        model = GMMClassifier(**model_kwargs)
        with Capturing(self.params.get('save_path', "log.txt")) as logs:
            model.train(train_loader, val_loader=val_loader)
            results = model.evaluate(test_loader) if test_loader else {}
            # Ray Tune score
            if 'summary' in results:
                results['score'] = results['summary'].get('accuracy')
        return results


class ExperimentFactory:
    """
    Factory to create Experiment instances by name.

    Usage:
        exp = ExperimentFactory.create_experiment(
            arch_name='transformer',
            train_dir='/path/train',
            val_dir='/path/val',
            test_dir='/path/test',
            batch_size=32,
            epochs=10,
            save_path='/path/save',
            model_kwargs={...}
        )

    Arguments:
    - arch_name (str): one of 'transformer', 'gpt2', 'cnn', 'gmm'.
    - kwargs: all parameters accepted by the target experiment's `params` dict.
    """
    @staticmethod
    def create_experiment(
        arch_name: str,
        **kwargs: Any
    ) -> Experiment:
        mapping = {
            'transformer': TransformerExperiment,
            'gpt2': GPT2Experiment,
            'cnn': CNNExperiment,
            'gmm': GMMExperiment
        }
        arch_name = arch_name.lower()
        if arch_name not in mapping:
            raise ValueError(f"Unknown architecture '{arch_name}'. Available: {list(mapping.keys())}")
        exp_class = mapping[arch_name]
        exp = exp_class()
        exp.load_parameters(kwargs)
        return exp
