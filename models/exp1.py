"""
Single Modality
"""
import torch
import torch.nn as nn
from models.transformer import Transformer
from models.misc import get_1d_sincos_pos_embed, PatchEmbed


class Exp1(nn.Module):
    def __init__(
        self,
        signal_length=3000,
        patch_size=30,
        in_chans=1,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.2,
        num_classes=5,
    ):
        super().__init__()
        self.num_patches = signal_length // patch_size

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        # --------------------------------------------------------------------------

        self.classifier = nn.Linear(embed_dim, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)

        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add position embedding
        x = x + self.pos_embed

        # apply Transformer blocks
        x = self.blocks(x)

        # token
        x = x[:, 0]

        # classify
        pred = self.classifier(x)

        return pred
