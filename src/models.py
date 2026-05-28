import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    """Configurable MLP that returns intermediate feature maps on request."""

    def __init__(self, input_dim=28 * 28, hidden_dims=(20, 10), output_dim=2, activation=nn.Sigmoid):
        super().__init__()
        layers = [nn.Flatten()]
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers += [nn.Linear(prev, output_dim), nn.Softmax(dim=-1)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x, inference=False):
        feature_maps = []
        for block in self.layers:
            x = block(x)
            if inference:
                feature_maps.append(x)
        return x, feature_maps

    def linear_layer_indices(self):
        return [i for i, m in enumerate(self.layers) if isinstance(m, nn.Linear)]

    def activation_layer_indices(self):
        return [i for i, m in enumerate(self.layers)
                if not isinstance(m, (nn.Flatten, nn.Linear))]


class SmallCNN(nn.Module):
    """Configurable N-conv-block CNN for image classification.

    Default: Deep4s config — 4 blocks [32, 64, 128, 256] with GlobalAvgPool.
    Achieved 90 % on CIFAR-10 in 150 epochs with baseline augmentation.

    Backbone: N × (Conv2d + BatchNorm + ReLU + MaxPool(2)).
    Head (global_pool=True):  AdaptiveAvgPool(1) → Linear(ch[-1], n_classes)
    Head (global_pool=False): AdaptiveAvgPool(4) → Linear(ch[-1]*16, fc_dim) → Linear(fc_dim, n_classes)

    forward(x) returns raw logits. Use nn.CrossEntropyLoss.
    Compatible with hook-based collect_layer_dicts() in bft.py.
    """

    def __init__(self, channels=(32, 64, 128, 256), fc_dim=128, n_classes=10, global_pool=True):
        super().__init__()
        in_ch = 3
        blocks = []
        for out_ch in channels:
            blocks += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_ch = out_ch
        self.features = nn.Sequential(*blocks)

        if global_pool:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Linear(channels[-1], n_classes)
        else:
            self.pool = nn.AdaptiveAvgPool2d(4)
            self.classifier = nn.Sequential(
                nn.Linear(channels[-1] * 16, fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(fc_dim, n_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x.flatten(1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Vision Transformer (Tiny) ─────────────────────────────────────────────────

class PatchEmbedder(nn.Module):
    def __init__(self, img_size=28, patch_size=7, embed_dim=32):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2   # 16
        patch_dim = patch_size * patch_size                # 49
        self.proj = nn.Linear(patch_dim, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)            # (B,1,4,4,7,7)
        x = x.contiguous().view(B, -1, p * p)             # (B,16,49)
        return self.proj(x)                                # (B,16,32)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=32, n_heads=2, ffn_dim=64):
        super().__init__()
        self.ln1  = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln2  = nn.LayerNorm(embed_dim)
        self.ffn1 = nn.Linear(embed_dim, ffn_dim)
        self.ffn2 = nn.Linear(ffn_dim, embed_dim)
        self._capture = False
        # Captured activations (set during forward when _capture=True)
        self._attn_in   = None   # (B, T, d) post-LN1, all tokens entering MHA
        self._attn_w    = None   # (B, heads, T, T) raw attention weights
        self._attn_out  = None   # (B, T, d) MHA output before residual
        self._ffn1_in   = None   # (B, T, d) post-LN2
        self._ffn2_in   = None   # (B, T, ffn_dim) post-GELU

    def forward(self, x):
        h = self.ln1(x)
        if self._capture:
            self._attn_in = h.detach()
        attn_out, attn_w = self.attn(h, h, h,
                                     need_weights=self._capture,
                                     average_attn_weights=False)
        if self._capture:
            self._attn_w   = attn_w.detach()    # (B, heads, T, T)
            self._attn_out = attn_out.detach()
        x = x + attn_out
        h = self.ln2(x)
        if self._capture:
            self._ffn1_in = h.detach()
        h2 = F.gelu(self.ffn1(h))
        if self._capture:
            self._ffn2_in = h2.detach()
        x = x + self.ffn2(h2)
        return x


class TinyViT(nn.Module):
    """Tiny Vision Transformer for small images (default: 28×28 MNIST, 2 classes).

    Single transformer block with learned CLS token and positional embeddings.
    forward(x, capture=False) — set capture=True to record activations in block._*.
    Returns log_softmax logits.
    Compatible with save_experiment / load_experiment via MODEL_REGISTRY.
    """

    def __init__(self, embed_dim=32, n_heads=2, ffn_dim=64, n_classes=2):
        super().__init__()
        self.patch_embed = PatchEmbedder(embed_dim=embed_dim)
        T = self.patch_embed.n_patches + 1   # 17
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, T, embed_dim))
        self.block = TransformerBlock(embed_dim, n_heads, ffn_dim)
        self.ln    = nn.LayerNorm(embed_dim)
        self.head  = nn.Linear(embed_dim, n_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, capture=False):
        self.block._capture = capture
        B = x.shape[0]
        tokens = torch.cat([self.cls_token.expand(B, -1, -1),
                             self.patch_embed(x)], dim=1) + self.pos_embed
        tokens = self.block(tokens)
        return F.log_softmax(self.head(self.ln(tokens[:, 0])), dim=1)
