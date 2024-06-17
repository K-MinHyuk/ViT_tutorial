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
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

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
    
class MultiHeadAttention(nn.Module):
    def __init__(self, 
                 embedding_size: int = 768, 
                 num_heads: int = 8, 
                 dropout: float = 0):
        super(MultiHeadAttention, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        
        # keys, queries, values
        self.K = nn.Linear(embedding_size, embedding_size)
        self.Q = nn.Linear(embedding_size, embedding_size)
        self.V = nn.Linear(embedding_size, embedding_size)

        # drop out
        self.att_drop = nn.Dropout(dropout)
        
        self.projection = nn.Linear(embedding_size, embedding_size)
        
    def forward(self, x : Tensor) -> Tensor:
        # keys, queries, values
        keys    = rearrange(self.K(embedded_tensor), "b n (h d) -> b h n d", h=self.num_heads)
        queries = rearrange(self.Q(embedded_tensor), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.V(embedded_tensor), "b n (h d) -> b h n d", h=self.num_heads)

        # Attention Score
        QK = torch.einsum('b h q d, b h k d -> b h q k', queries, keys)
        scaling = self.embedding_size ** (1/2)
        attention_score = F.softmax(QK/scaling, dim=-1)
        representation = torch.einsum('b h p d, b h d v -> b h p v ', attention_score, values)

        # Concat and projection
        concated = rearrange(representation, "b h p d -> b p (h d)")
        out = self.projection(concated)

        return out