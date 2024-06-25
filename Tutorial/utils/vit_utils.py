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
                image_size: list[int, int, int], # C, H, W
                patch_size: int,
                hidden_dim: Optional[int] = None
            ):
        super(Image_Embedding, self).__init__()

        self.patch_size = patch_size
        self.c, self.h, self.w = image_size
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
    
class Multi_Head_Attention(nn.Module):
    def __init__(self, 
                 embedding_size: int = 768, 
                 num_heads: int = 8, 
                 dropout: float = 0):
        super(Multi_Head_Attention, self).__init__()
        
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
        keys    = rearrange(self.K(x), "b n (h d) -> b h n d", h=self.num_heads)
        queries = rearrange(self.Q(x), "b n (h d) -> b h n d", h=self.num_heads)
        values  = rearrange(self.V(x), "b n (h d) -> b h n d", h=self.num_heads)

        # Attention Score
        QK = torch.einsum('b h q d, b h k d -> b h q k', queries, keys)
        scaling = self.embedding_size ** (1/2)
        attention_score = F.softmax(QK/scaling, dim=-1)
        representation = torch.einsum('b h p d, b h d v -> b h p v ', attention_score, values)

        # Concat and projection
        concated = rearrange(representation, "b h p d -> b p (h d)")
        out = self.projection(concated)

        return out
    
class ResidualConnection(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, x):
        temp_x = x
        x = self.layer(x)
        return x + temp_x
    
class FeedForward(nn.Module):
    def __init__(self, 
                 embedding_size: int,
                 expansion: int = 4, 
                 dropout: float = 0.):
        super(FeedForward, self).__init__()

        self.ff_layer = nn.Sequential(
            nn.Linear(embedding_size, expansion * embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * embedding_size, embedding_size),
        )
    def forward(self, x):
        return self.ff_layer(x)
    
class Transformer_Block(nn.Module):
    def __init__(self, 
                 embedding_size: int = 768,
                 dropout: float = 0.,
                 forward_expansion: int = 4,
                 forward_dropout: float = 0,
                 **kwargs):
        super(Transformer_Block, self).__init__()
        self.norm_mha = nn.Sequential(
            ResidualConnection(
                nn.Sequential(
                    nn.LayerNorm(embedding_size),
                    Multi_Head_Attention(embedding_size, **kwargs),
                    nn.Dropout(dropout)
                    )
                )
            )
        self.norm_ff = nn.Sequential(
            ResidualConnection(
                nn.Sequential(
                    nn.LayerNorm(embedding_size),
                    FeedForward(embedding_size, forward_expansion, forward_dropout),
                    nn.Dropout(dropout)
                )
            )
        )

    def forward(self, x):
        x = self.norm_mha(x)
        return self.norm_ff(x)
    
class TransformerEncoder(nn.Module):
    def __init__(self, depth: int = 12, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.multi_encoder_layer = nn.Sequential(*[Transformer_Block(**kwargs) for _ in range(depth)])    
        
    def forward(self, x):
        return self.multi_encoder_layer(x)


class MLP_Head(nn.Module):
    def __init__(self, embedding_size: int = 768, n_classes: int = 1000, reduce_type: Optional[str] = None):
        super(MLP_Head, self).__init__()
        self.reduce_type = reduce_type
        self.r_layer = self.reducelayer()
        self.layers = nn.Sequential(
                nn.LayerNorm(embedding_size), 
                nn.Linear(embedding_size, n_classes)
                )
        
    def reducelayer(self):
        if self.reduce_type == None:
            return lambda x: x[:, 0]
        elif self.reduce_type == 'mean':
            return Reduce('b p e -> b e', reduction='mean')
    
    def forward(self, x):
        x = self.r_layer(x)
        return self.layers(x)
    

class ViT(nn.Module):
    def __init__(self,     
                img_size: list[int, int, int],
                patch_size: int = 16,
                embedding_size: int = 768,
                depth: int = 12,
                n_classes: int = 1000,
                reduce_type: Optional[str] = None,
                **kwargs):
        super(ViT, self).__init__()
        
        self.layers = nn.Sequential(
            Image_Embedding(img_size, patch_size, embedding_size),
            TransformerEncoder(depth, **kwargs),
            MLP_Head(n_classes=n_classes, reduce_type=reduce_type),
            torch.nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        return self.layers(x)

