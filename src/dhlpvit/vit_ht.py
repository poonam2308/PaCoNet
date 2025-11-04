# hough_on_tokens.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.dhlp.lcnn.models.HT import hough_transform, CAT_HTIHT


# These come from your HT.py file
# Make sure HT.py is on your PYTHONPATH or in the same folder.
# from HT import hough_transform, CAT_HTIHT


class HoughOnTokens(nn.Module):
    """
    Injects a (learnable) Hough Transform block into a ViT by
    reshaping tokens back to a 2D grid, applying CAT_HTIHT, and returning to tokens.

    Args:
        vit_grid: Optional (Ht, Wt). If None, we try to infer a square grid from N tokens.
        c: Working channel dim for the Hough branch (after 1x1 projection).
        theta_res: Angle step in degrees for Hough space (e.g., 3 -> 60 bins).
        rho_res: Rho step in pixels for Hough space (typically 1).
        attach_cls: Whether this module expects and preserves a class token at tokens[:, :1].
    """
    def __init__(
        self,
        vit_grid: Optional[Tuple[int, int]] = None,
        c: int = 64,
        theta_res: int = 3,
        rho_res: int = 1,
        attach_cls: bool = False,
    ):
        super().__init__()
        self.vit_grid = vit_grid
        self.c = int(c)
        self.theta_res = int(theta_res)
        self.rho_res = int(rho_res)
        self.attach_cls = bool(attach_cls)

        # Lazy-initialized layers/buffers (created once we know D, Ht, Wt, device)
        self.to_feats: Optional[nn.Conv2d] = None
        self.from_feats: Optional[nn.Conv2d] = None
        self.hough_block: Optional[nn.Module] = None
        self.register_buffer("vote_index", None, persistent=False)  # (Ht, Wt, h, w) after created

    # ---- helpers ----
    def _infer_grid(self, N: int) -> Tuple[int, int]:
        if self.vit_grid is not None:
            Ht, Wt = self.vit_grid
            if Ht * Wt != N:
                raise ValueError(f"N={N} does not match vit_grid={self.vit_grid}.")
            return Ht, Wt
        s = int(round(math.sqrt(N)))
        if s * s != N:
            raise ValueError(
                f"Can't infer square grid from N={N}. Pass vit_grid=(Ht, Wt)."
            )
        return s, s

    def _lazy_build(self, Ht: int, Wt: int, D: int, device, dtype):
        # 1) build projection layers if missing
        if self.to_feats is None:
            self.to_feats = nn.Conv2d(D, self.c, kernel_size=1, bias=True).to(device=device, dtype=dtype)
            self.from_feats = nn.Conv2d(self.c, D, kernel_size=1, bias=True).to(device=device, dtype=dtype)

        # 2) build vote_index buffer if missing
        if self.vote_index is None:
            vote_np = hough_transform(Ht, Wt, theta_res=self.theta_res, rho_res=self.rho_res)  # (Ht,Wt,h,w)
            vote = torch.from_numpy(vote_np).to(device=device, dtype=torch.float32)  # keep in float32
            self.vote_index = vote  # register_buffer already exists

        # 3) build the CAT_HTIHT block once we have vote_index
        if self.hough_block is None:
            # The block expects the same dtype/device as the feature map it will receive.
            # vote_index is used internally by HT/IHT and will be cast on the fly if needed.
            self.hough_block = CAT_HTIHT(self.vote_index, inplanes=self.c, outplanes=self.c).to(device=device, dtype=dtype)

    # ---- forward ----
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, N, D] if no CLS token,
                or [B, 1+N, D] if attach_cls=True
        """
        if self.attach_cls:
            cls_tok, x = tokens[:, :1], tokens[:, 1:]
        else:
            x = tokens

        B, N, D = x.shape
        Ht, Wt = self._infer_grid(N)

        device = x.device
        dtype = x.dtype

        self._lazy_build(Ht, Wt, D, device, dtype)

        # [B, N, D] -> [B, D, Ht, Wt]
        x = x.transpose(1, 2).contiguous().reshape(B, D, Ht, Wt)

        # 1x1 -> Hough block -> 1x1
        x = self.to_feats(x)
        x = self.hough_block(x)
        x = self.from_feats(x)

        # [B, D, Ht, Wt] -> [B, N, D]
        x = x.reshape(B, D, N).transpose(1, 2).contiguous()

        if self.attach_cls:
            x = torch.cat([cls_tok, x], dim=1)

        return x


# ----------- example usage -----------
if __name__ == "__main__":
    # Example: plug into a ViT-like pipeline
    B = 2
    H = W = 512
    patch = 16           # typical ViT patch size
    Ht = H // patch      # 32
    Wt = W // patch      # 32
    N = Ht * Wt          # 1024
    D = 768              # ViT embedding dim

    # Fake tokens (without CLS)
    tokens = torch.randn(B, N, D)

    # Create the module; we know the grid so pass it explicitly.
    hot = HoughOnTokens(vit_grid=(Ht, Wt), c=64, theta_res=3, rho_res=1, attach_cls=False)

    # Forward: returns same shape [B, N, D]
    out = hot(tokens)
    print("in :", tokens.shape)
    print("out:", out.shape)

    # If you *do* have a class token at the front:
    tokens_with_cls = torch.randn(B, 1 + N, D)
    hot_cls = HoughOnTokens(vit_grid=(Ht, Wt), c=64, attach_cls=True)
    out2 = hot_cls(tokens_with_cls)
    print("with CLS in :", tokens_with_cls.shape)
    print("with CLS out:", out2.shape)
