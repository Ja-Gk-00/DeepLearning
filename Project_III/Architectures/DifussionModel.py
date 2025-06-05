from typing import Any, Dict, Optional
import torch
from torch import nn, optim
from diffusers import UNet2DModel, DDPMScheduler
from DataObjects.DataLoader import DataLoader
from Architectures.ArchitectureModel import GeneratorBase


class DiffusionModel(GeneratorBase):
    def __init__(self, device: torch.device, image_size: int = 256) -> None:
        super().__init__(device)
        self.model: Optional[UNet2DModel] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.image_size = image_size

    def build_model(self) -> None:
        # Load pretrained UNet without modifying its architecture
        self.model = UNet2DModel.from_pretrained("google/ddpm-celebahq-256").to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained("google/ddpm-celebahq-256")
        self._move_to_device()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.image_size, self.image_size):
            x = torch.nn.functional.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )
        return self.model(x)["sample"]  # type: ignore

    def sample_latent(self, num_samples: int) -> torch.Tensor:
        channels = self.model.in_channels  # type: ignore
        # match pretrained UNet’s default sample_size (256)
        return torch.randn(
            (num_samples, channels, self.image_size, self.image_size),
            device=self.device
        )

    def generate(
        self, num_samples: int, generator: Optional[torch.Generator] = None
    ) -> torch.Tensor:
        latent = self.sample_latent(num_samples)
        for t in reversed(range(self.scheduler.num_train_timesteps)):  # type: ignore
            timestep = torch.full(
                (num_samples,), t, device=self.device, dtype=torch.long
            )
            noise_pred = self.model(latent, timestep)["sample"]  # type: ignore
            latent = self.scheduler.step(noise_pred, t, latent)["prev_sample"]  # type: ignore
        return latent

    def train_step(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        imgs = batch
        if imgs.shape[-2:] != (self.image_size, self.image_size):
            imgs = torch.nn.functional.interpolate(
                imgs,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False
            )
        noise = torch.randn_like(imgs)
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps,
            (imgs.size(0),), device=self.device
        )  # type: ignore
        noisy = self.scheduler.add_noise(imgs, noise, timesteps)  # type: ignore
        pred = self.model(noisy, timesteps)["sample"]  # type: ignore
        loss = nn.MSELoss()(pred, noise)
        return {"loss": loss}

    def evaluate(self, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.train_step(batch)

    def configure_optimizers(self) -> Any:
        return optim.Adam(self.model.parameters(), lr=1e-4)  # type: ignore

    def save_model(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)  # type: ignore

    def load_model(self, path: str, map_location: Optional[str] = None) -> None:
        state = torch.load(path, map_location=map_location)
        self.model.load_state_dict(state)  # type: ignore

    def _move_to_device(self) -> None:
        if self.model is not None:
            self.model.to(self.device)  # type: ignore

    def train_architecture(self, dataloader: DataLoader, epochs: int) -> None:
        optimizer = self.configure_optimizers()  # type: ignore
        for _ in range(epochs):
            for batch_obj in dataloader:
                imgs = torch.stack([img for img, _ in batch_obj]).to(self.device)
                out = self.train_step(imgs)
                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()

    @classmethod
    def from_pretrained(cls, device: torch.device, image_size: int = 256) -> "DiffusionModel":
        dm = cls(device, image_size)
        dm.build_model()
        return dm
