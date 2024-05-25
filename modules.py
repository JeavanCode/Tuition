import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                assert self.shadow[name].requires_grad == False

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                assert param.requires_grad == True

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        assert channels % 64 == 0
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=channels // 64, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        _, _, size, _ = x.shape
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, input_size=256, feature_size=16, c_max=256, c_in=3, c_out=3, time_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.num_blocks = int(math.log2(input_size // feature_size))
        self.channels = [min(64 * 2 ** i, 512) for i in range(self.num_blocks+1)]
        self.attention_resolution = 16
        print(f"channels: {self.channels}")

        self.inc = DoubleConv(c_in, self.channels[0])

        self.encoder = []
        current_feature_size = input_size
        for i in range(self.num_blocks):
            self.encoder.append(Down(self.channels[i], self.channels[i + 1]))
            current_feature_size = current_feature_size // 2
            if current_feature_size <= self.attention_resolution:
                self.encoder.append(SelfAttention(self.channels[i + 1]))
        self.encoder = nn.ModuleList(self.encoder)

        self.bottles = []
        intermediate_channels = self.channels[-1] * 2
        n_bottles = 3
        for i in range(n_bottles):
            if i == 0:
                self.bottles.append(DoubleConv(self.channels[-1], intermediate_channels))
                self.bottles.append(SelfAttention(intermediate_channels))
            elif i == (n_bottles-1):
                self.bottles.append(DoubleConv(intermediate_channels, self.channels[-1]))
                self.bottles.append(SelfAttention(self.channels[-1]))
            else:
                self.bottles.append(DoubleConv(intermediate_channels, intermediate_channels))
                self.bottles.append(SelfAttention(intermediate_channels))
        self.bottles = nn.ModuleList(self.bottles)

        self.decoder = []
        for i in reversed(range(self.num_blocks)):
            self.decoder.append(Up(self.channels[i+1]+self.channels[i], self.channels[i]))
            current_feature_size = current_feature_size * 2
            if current_feature_size <= self.attention_resolution:
                self.decoder.append(SelfAttention(self.channels[i]))
        self.decoder = nn.ModuleList(self.decoder)

        self.outc = nn.Conv2d(self.channels[0], c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        x = self.inc(x)
        x_skips = [x]
        for module in self.encoder:
            if module.__class__.__name__ == "Down":
                x = module(x, t)
                if x.shape[-1] > self.attention_resolution:
                    x_skips.append(x)
            elif module.__class__.__name__ == "SelfAttention":
                x = module(x)
                if x.shape[-1] <= self.attention_resolution:
                    x_skips.append(x)
            else:
                raise ValueError("Unsupported Modules")

        for module in self.bottles:
            x = module(x)

        i = -2
        for module in self.decoder:
            if module.__class__.__name__ == "Up":
                x = module(x, x_skips[i], t)
                i += -1
            elif module.__class__.__name__ == "SelfAttention":
                x = module(x)
            else:
                raise ValueError("Unsupported Modules")

        output = self.outc(x)
        return output


if __name__ == '__main__':
    # net = UNet(device="cpu")
    net = UNet(input_size=64, feature_size=8, device="cpu")
    x = torch.randn(3, 3, 64, 64)
    t = torch.tensor([100, 200, 300], dtype=torch.int64)
    summary(model=net, input_data=(x, t), depth=2)

