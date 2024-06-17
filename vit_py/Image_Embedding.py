from collections import OrderedDict
from typing import Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
from torch import nn
from torch import Tensor
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Image_Embedding(nn.Module):
    """
    input: image [Tensor]
    putout: patch wise embedded vector [Tensor]
    """
    def __init__(self, 
                image_size: tuple[int, int, int, int], # N, C, H, W
                patch_size: int,
                hidden_dim: Optional[int] = None
            ):
        super(Image_Embedding, self).__init__()

        self.patch_size = patch_size
        self.n, self.c, self.h, self.w = image_size
        self.n_h = self.h // self.patch_size
        self.n_w = self.w // self.patch_size
        self.seq_length = self.n_h * self.n_w 

        if hidden_dim == None:
                self.hidden_dim = self.patch_size * self.patch_size * self.c
        else:
             self.hidden_dim = hidden_dim

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, self.seq_length+1, self.hidden_dim).normal_(std=0.02))  # from BERT

        self.conv_proj = nn.Sequential(
                        nn.LayerNorm(
                                [self.c, self.h, self.w]
                        ),
                        nn.Conv2d(
                        in_channels=3, out_channels=self.hidden_dim, kernel_size=patch_size, stride=patch_size
                        ),
                        nn.LayerNorm(
                                [self.hidden_dim, self.n_h, self.n_w]
                        )
                )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        torch._assert(h == self.h, f"Wrong image height! Expected {self.h} but got {h}!")
        torch._assert(w == self.w, f"Wrong image width! Expected {self.w} but got {w}!")
        
        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, self.n_h * self.n_w)
        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        x = x.permute(0, 2, 1)        
        
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        return x + self.pos_embedding