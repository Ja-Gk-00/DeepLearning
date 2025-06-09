from typing import Any, Dict, Optional
import torch
from torch import nn, optim
import torch.nn.functional as F
from diffusers import UNet2DModel, DDPMScheduler
from DataObjects.DataLoader import DataLoader
from Architectures.ArchitectureModel import GeneratorBase

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

from tqdm.auto import tqdm
import os

from Utils.utils import make_grid, evaluate

from diffusers import DDPMScheduler
from diffusers import DDPMPipeline

import math

MODEL_ID = "google/ddpm-celebahq-256"

class DiffusionModel(GeneratorBase):

    def __init__(self, device: torch.device, image_size: int = 256) -> None:
        super().__init__(device)
        self.model: Optional[UNet2DModel] = None
        self.scheduler: Optional[DDPMScheduler] = None
        self.image_size = image_size

    def build_model(self) -> None:
        self.model = UNet2DModel.from_pretrained(MODEL_ID).to(self.device)
        self.scheduler = DDPMScheduler.from_pretrained(MODEL_ID)
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
    def return_custom_arch(cls, config:Dict[str, Any]) -> UNet2DModel:
        model = UNet2DModel(
            sample_size=config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channes for each UNet block
            down_block_types=( 
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D", 
                "DownBlock2D", 
                "DownBlock2D", 
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D"  
            ))
        return model
    

    @classmethod
    def from_pretrained(cls, device: torch.device, image_size: int = 256) -> "DiffusionModel":
        dm = cls(device, image_size)
        dm.build_model()
        return dm
    
@staticmethod
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device):
    logging_dir = os.path.join(config.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )
    device = accelerator.device

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
    accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
    )
        
    global_step = 0
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch.data.to(device)
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=device)
            bs = clean_images.shape[0]

            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
                
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir) 

@staticmethod
def generate(
        config,
        pipeline,
        seed=2137,
        grid_rows=None,
        grid_cols=None,
    ):
        bs = config.eval_batch_size
        gen_seed = seed or config.seed
        generator = torch.manual_seed(gen_seed)
        images = pipeline(batch_size=bs, generator=generator).images

        out_dir = os.path.join(config.output_dir, "generated")
        os.makedirs(out_dir, exist_ok=True)
        for idx, img in enumerate(images):
            img.save(os.path.join(out_dir, f"image_{idx:04d}.png"))

        rows = grid_rows or int(math.sqrt(bs))
        cols = grid_cols or rows
        grid = make_grid(images, rows=rows, cols=cols)
        grid.save(os.path.join(out_dir, "grid.png"))

        print(f"Saved {len(images)} images + grid at: {out_dir}")

