"""
Late Fusion
"""
import torch
import torch.nn as nn
from models.transformer import Transformer
from models.misc import get_1d_sincos_pos_embed, PatchEmbed


class Exp3(nn.Module):
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

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim),
            requires_grad=False,
        )

        # --------------------------------------------------------------------------
        # eeg specifics
        self.eeg_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eeg_encoder = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # eog specifics
        self.eog_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.eog_encoder = Transformer(embed_dim, depth, num_heads, mlp_ratio, dropout)
        # --------------------------------------------------------------------------

        self.classifier1 = nn.Linear(embed_dim, num_classes)
        self.classifier2 = nn.Linear(embed_dim, num_classes)
        self.classifier3 = nn.Linear(embed_dim * 2, num_classes)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.num_patches), cls_token=True
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward(self, x, y):
        # eeg encoding
        x = self.eeg_embed(x)
        cls_tokens = self.cls_token1.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.eeg_encoder(x)

        # eog encoding
        y = self.eog_embed(y)
        cls_tokens = self.cls_token2.expand(y.shape[0], -1, -1)
        y = torch.cat((cls_tokens, y), dim=1)
        y = y + self.pos_embed
        y = self.eog_encoder(y)

        # classify
        x = x[:, 0]
        pred1 = self.classifier1(x)
        y = y[:, 0]
        pred2 = self.classifier2(y)

        z = torch.cat((x, y), dim=-1)
        pred3 = self.classifier3(z)

        return pred1, pred2, pred3
