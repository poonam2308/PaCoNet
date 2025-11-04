# hough_in_swin.py
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# from your HT.py
from HT import hough_transform, CAT_HTIHT


class HoughOnSwinMap(nn.Module):
    """
    Line-prior adapter for Swin: operates on a 2D feature map (B, C, H, W),
    injects CAT_HTIHT as a residual.
    """
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        c_work: int = 64,
        theta_res: int = 3,
        rho_res: int = 1,
        alpha: float = 1.0,   # residual scaling
    ):
        super().__init__()
        self.H, self.W, self.C = int(H), int(W), int(C)
        self.c_work = int(c_work)
        self.alpha = float(alpha)

        # 1x1 projections
        self.to_feats = nn.Conv2d(C, c_work, kernel_size=1, bias=True)
        self.from_feats = nn.Conv2d(c_work, C, kernel_size=1, bias=True)

        # Precompute vote index for this grid, register as buffer
        vote_np = hough_transform(H, W, theta_res=theta_res, rho_res=rho_res)  # (H,W,h,w)
        self.register_buffer("vote_index", torch.from_numpy(vote_np).float(), persistent=False)

        # Hough block (expects NCHW)
        self.hough = CAT_HTIHT(self.vote_index, inplanes=c_work, outplanes=c_work)

        # Light norm to keep the residual stable
        self.norm = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) feature map
        returns: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert (H, W) == (self.H, self.W) and C == self.C, \
            f"Got {(B,C,H,W)}, expected C={self.C}, H={self.H}, W={self.W}"

        y = self.to_feats(x)
        y = self.hough(y)          # CAT_HTIHT
        y = self.from_feats(y)
        # residual fusion
        out = self.norm(x + self.alpha * y)
        return out


class HoughOnSwinTokens(nn.Module):
    """
    Same as above, but accepts Swin's (B, N, C) tokens and an (H, W) grid.
    Handy to drop right inside a Swin block without editing its internals much.
    """
    def __init__(
        self,
        H: int,
        W: int,
        C: int,
        c_work: int = 64,
        theta_res: int = 3,
        rho_res: int = 1,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.H, self.W, self.C = int(H), int(W), int(C)
        self.adapter = HoughOnSwinMap(H, W, C, c_work, theta_res, rho_res, alpha)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, C) where N = H*W (Swin uses NHWC internally but many wrappers expose tokens)
        """
        B, N, C = tokens.shape
        assert N == self.H * self.W and C == self.C, \
            f"Tokens have shape (B,{N},{C}); expected N=H*W={self.H*self.W}, C={self.C}"

        # (B,N,C) -> (B,C,H,W)
        x = tokens.transpose(1, 2).contiguous().view(B, C, self.H, self.W)
        x = self.adapter(x)
        # (B,C,H,W) -> (B,N,C)
        tokens = x.view(B, C, N).transpose(1, 2).contiguous()
        return tokens


# ---------------- Example: attaching to a timm Swin ----------------
if __name__ == "__main__":
    import timm

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Example model and input size
    img_size = 512
    model_name = "swin_tiny_patch4_window7_224"   # works with any Swin; grid scales with input
    model = timm.create_model(model_name, pretrained=False, img_size=img_size).to(device)
    model.eval()

    # Swin with patch size 4 produces an initial grid of H0=W0=img_size/4
    # Then PatchMerging halves H/W at each stage.
    # For 512x512:
    #   Stage 0: 128 x 128, C=96   (for swin_tiny)
    #   Stage 1: 64  x 64 , C=192
    #   Stage 2: 32  x 32 , C=384
    #   Stage 3: 16  x 16 , C=768

    # We’ll insert a Hough adapter after Stage 1 (64x64, C=192) as an example.
    # Timm's Swin exposes: model.layers[stage].blocks[block]
    H1, W1, C1 = img_size // 8, img_size // 8, 192  # 512/8=64

    hough_stage1 = HoughOnSwinMap(H=H1, W=W1, C=C1, c_work=64).to(device)

    # A thin wrapper module to splice in after a given block's forward. We’ll wrap the last
    # block of stage 1: x comes in as (B, H*W, C). We reshape to map, apply Hough, reshape back.
    class InjectAfterBlock(nn.Module):
        def __init__(self, block: nn.Module, adapter_map: HoughOnSwinMap, H: int, W: int, C: int):
            super().__init__()
            self.block = block
            self.adapter = adapter_map
            self.H, self.W, self.C = H, W, C

        def forward(self, x):
            # x is (B, H*W, C) at this point in Swin’s stage pipeline
            B, N, C = x.shape
            assert N == self.H * self.W and C == self.C
            x = self.block(x)  # original Swin block
            # convert to NCHW, inject Hough, back to tokens
            fmap = x.transpose(1, 2).contiguous().view(B, C, self.H, self.W)
            fmap = self.adapter(fmap)
            x = fmap.view(B, C, N).transpose(1, 2).contiguous()
            return x

    # Replace the last block in stage 1 with the wrapped version
    stage_idx = 1
    block_idx = len(model.layers[stage_idx].blocks) - 1
    orig_block = model.layers[stage_idx].blocks[block_idx]
    model.layers[stage_idx].blocks[block_idx] = InjectAfterBlock(orig_block, hough_stage1, H1, W1, C1).to(device)

    # Smoke test
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    with torch.no_grad():
        out = model(dummy)   # forward should work as usual
    print("Forward OK. Output shape:", tuple(out.shape))
