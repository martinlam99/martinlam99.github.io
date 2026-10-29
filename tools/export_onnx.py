"""Export the LSGAN generator from a Lightning checkpoint to ONNX for the website demo.

Usage: python tools/export_onnx.py path/to/gan-epoch=99-v1.ckpt models/generator.onnx
Needs: torch, onnx, onnxscript
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tanh


# Same architecture as Generator.py used for training
class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()

        self.lin1 = nn.Linear(latent_dim, 512 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(512)

        self.ct1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.ct2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.ct3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.ct4 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        x = self.lin1(z).view(z.size(0), 512, 4, 4)
        x = F.relu(self.bn0(x))            # [1, 512, 4, 4]
        x = F.relu(self.bn1(self.ct1(x)))  # [1, 256, 8, 8]
        x = F.relu(self.bn2(self.ct2(x)))  # [1, 128, 16, 16]
        x = F.relu(self.bn3(self.ct3(x)))  # [1, 64, 32, 32]
        return tanh(self.ct4(x))           # [1, 3, 64, 64], values in [-1, 1]


def main(ckpt_path, out_path):
    # weights_only=True loads tensors without running any pickled code
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    latent_dim = ckpt["hyper_parameters"]["latent_dim"]

    # Keep only the generator weights and drop the "generator." prefix
    state = {k.removeprefix("generator."): v
             for k, v in ckpt["state_dict"].items() if k.startswith("generator.")}

    generator = Generator(latent_dim)
    generator.load_state_dict(state)
    generator.eval()  # BatchNorm uses its running stats, same as plot_images()

    z = torch.randn(1, latent_dim)
    torch.onnx.export(generator, (z,), out_path,
                      input_names=["z"], output_names=["image"],
                      opset_version=18, dynamo=True, external_data=False)
    print(f"Exported {out_path} (latent_dim={latent_dim})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
