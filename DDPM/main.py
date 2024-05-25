import glob
import os

import torch
import torchvision
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import FrechetInceptionDistance
from torchinfo import summary
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import torch.nn.functional as F
from ddpm.modules import UNet, EMA
from vqgan.utils import cosine_scheduler, get_params_groups


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        return x


class ImageDataset(Dataset):
    def __init__(self, path: list, transform):
        self.names = sum([glob.glob(p + "/**/*.jp*", recursive=True)+glob.glob(p + "/**/*.png", recursive=True)+glob.glob(p + "/**/*.webp", recursive=True) for p in path], [])
        self.transform = transform

    def __getitem__(self, index):
        image = self.transform(Image.open(self.names[index]).convert("RGB"))
        return image

    def __len__(self):
        return len(self.names)


class Config:
    run_name = "DDPM_Uncondtional"
    image_size = 256
    feature_size = 16
    c_max = 512
    noise_steps = 1000
    sample_number = 100
    steps = int(500e3)
    batch_size = 8
    accumulate_size = 8
    train_path = [r"E:\Datasets\lsun\lsun_church\church_outdoor_train_lmdb"]
    valid_path = [r"E:\Datasets\lsun\lsun_church\church_outdoor_val_lmdb"]
    device = "cuda"
    base_lr = 3e-5
    final_lr = 0.
    weight_decay = 1e-4
    warmup_steps = 1000
    ema_rate = 0.9999


def train(config):
    device = config.device
    unet = UNet(input_size=config.image_size, feature_size=config.feature_size, c_max=config.c_max).to(device)
    ema_unet = EMA(unet, decay=config.ema_rate)
    optimizer = torch.optim.AdamW(get_params_groups(unet), lr=config.base_lr)
    diffusion = Diffusion(img_size=config.image_size, noise_steps=config.noise_steps, device=device)
    logger = SummaryWriter()
    # data
    transform_train = transforms.Compose([transforms.RandomResizedCrop(size=(config.image_size, config.image_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = ImageDataset(path=config.train_path, transform=transform_train)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    transform_valid = transforms.Compose([transforms.Resize(size=(config.image_size, config.image_size)),
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    valid_dataset = ImageDataset(path=config.valid_path, transform=transform_valid)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=True)
    steps_per_epoch = len(train_loader)
    epochs = config.steps // steps_per_epoch
    print(f"samples: {len(train_dataset)}, steps per epoch: {steps_per_epoch}, epochs: {epochs}, total steps: {config.steps}")
    lr_schedule = cosine_scheduler(base_value=config.base_lr, final_steps=config.steps,
                                   warmup_steps=config.warmup_steps, final_value=config.final_lr)
    scaler = GradScaler()
    if not os.path.isdir("out_images"):
        os.makedirs("out_images")
    if not os.path.isdir(os.path.join("ckpt", config.run_name)):
        os.makedirs(os.path.join("ckpt", config.run_name))

    start_epoch = 5
    if start_epoch > 0:
        checkpoint = torch.load(os.path.join("ckpt", config.run_name, f"ckpt.pt"))
        unet.load_state_dict(checkpoint["unet"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema_unet.shadow = checkpoint["shadow"]
    for epoch in range(start_epoch, epochs):
        pbar = tqdm(train_loader)
        for i, images in enumerate(pbar):
            with autocast():
                global_step = epoch * steps_per_epoch + i
                unet.train()
                images = images.to(device)
                t = diffusion.sample_timesteps(images.shape[0]).to(device)
                if global_step == 0:
                    summary(model=unet, input_data=(images, t), depth=2)
                for j, _ in enumerate(optimizer.param_groups):
                    optimizer.param_groups[j]["lr"] = lr_schedule[global_step]
                    if j == 0:  # only the first group is regularized
                        optimizer.param_groups[j]["weight_decay"] = config.weight_decay

                x_t, noise = diffusion.noise_images(images, t)
                predicted_noise = unet(x_t, t)
                loss = F.mse_loss(noise, predicted_noise) / config.accumulate_size

                scaler.scale(loss).backward()
                if ((i + 1) % config.accumulate_size == 0) or ((i + 1) == steps_per_epoch):
                    scaler.step(optimizer)
                    optimizer.zero_grad()
                    scaler.update()
                ema_unet.update()

                pbar.set_postfix(MSE=loss.item())
                logger.add_scalar("mse", loss.item(), global_step=global_step)

        if (epoch+1) % 1 == 0:
            with torch.no_grad():
                ema_unet.apply_shadow()
                unet.eval()
                sampled_images = diffusion.sample(unet, n=config.batch_size*4)
                torchvision.utils.save_image(tensor=make_grid(
                    torch.cat(
                        [sampled_images], dim=0), nrow=8, padding=0), fp=os.path.join('out_images/', f'image_{epoch}.png'))
                ema_unet.restore()
                torch.save({"unet": unet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "shadow": ema_unet.shadow}, os.path.join("ckpt", config.run_name, f"ckpt.pt"))
                print(f"saved checkpoint at {epoch}")

        if (epoch+1) % 5 == 0 or epoch == epochs:
            with torch.no_grad():
                ema_unet.apply_shadow()
                unet.eval()

                fid = FrechetInceptionDistance(device=torch.device(config.device))
                for i, images in enumerate(tqdm(train_loader)):
                    if i >= (config.sample_number // config.batch_size):
                        break
                    images = images.to(device)
                    sampled_images = diffusion.sample(unet, n=config.batch_size)
                    fid.update((images.clamp(-1, 1) + 1) / 2, is_real=True)
                    fid.update(sampled_images, is_real=False)
                fid_score = fid.compute()
                logger.add_scalar("train_fid", fid_score.item(), global_step=global_step)

                fid = FrechetInceptionDistance(device=torch.device(config.device))
                for i, images in enumerate(tqdm(valid_loader)):
                    if i >= (config.sample_number):
                        break
                    images = images.to(device)
                    sampled_images = diffusion.sample(unet, n=1)
                    fid.update((images.clamp(-1, 1) + 1) / 2, is_real=True)
                    fid.update(sampled_images, is_real=False)
                fid_score = fid.compute()
                logger.add_scalar("valid_fid", fid_score.item(), global_step=global_step)

                ema_unet.restore()


if __name__ == "__main__":
    config = Config()
    train(config)
