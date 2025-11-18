import torch
import torch.nn as nn
import torchvision.transforms as T
import timm
import numpy as np
from PIL import Image

class DinoPatchFeatureExtractor(nn.Module):
    """
    Wraps a DINO ViT into a feature extractor that returns patch embeddings.

    Output: np.ndarray of shape (N, D) where N is number of tokens (patches),
    and D is the embedding dimension.
    """
    def __init__(self, model_name="vit_small_patch16_224.dino", device=None):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Create DINO model; we use it as a backbone / feature model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.model.to(self.device)

        # Some timm models have forward_features; if not, fall back to forward
        self.has_forward_features = hasattr(self.model, "forward_features")

        # Standard DINO preprocessing to 224x224
        self.transform = T.Compose([
            T.ToTensor(),                          # [0,1], CxHxW
            T.Resize(224, antialias=True),
            T.CenterCrop(224),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    @torch.no_grad()
    def forward(self, img_rgb_np: np.ndarray) -> np.ndarray:
        """
        img_rgb_np: H x W x 3, uint8 or float
        returns: (N, D) numpy array of patch/token embeddings
        """
        # Convert to tensor: HWC -> CHW, [0,1]
        pil_img = Image.fromarray(img_rgb_np.astype(np.uint8))
        x = self.transform(pil_img).unsqueeze(0).to(self.device)  # 1x3x224x224

        # Get features
        if self.has_forward_features:
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)

        # timm ViTs often return:
        #   - Tensor of shape (B, num_tokens, D)
        #   - or dict with "x" / "features"
        if isinstance(feats, dict):
            # try common keys, adapt if needed
            if "x" in feats:
                feats = feats["x"]
            elif "features" in feats:
                feats = feats["features"]
            else:
                # last item in dict as fallback
                feats = list(feats.values())[-1]

        # Now feats shape should be (1, num_tokens, D) or (1, D, H', W')
        if feats.dim() == 3:
            # [B, tokens, D] -> [tokens, D]
            feats_np = feats[0].detach().cpu().numpy()
            # Optionally drop CLS token (token 0)
            # CLS token often captures global info; for dense segmentation you might want only patches
            feats_np = feats_np[1:]  # remove CLS
            return feats_np  # (N, D)

        elif feats.dim() == 4:
            # [B, D, H', W'] -> [H'*W', D]
            feats_np = feats[0].detach().cpu().numpy()  # D, H', W'
            feats_np = np.transpose(feats_np, (1, 2, 0))  # H', W', D
            return feats_np.reshape(-1, feats_np.shape[-1])

        else:
            # Fallback: just flatten
            feats_np = feats[0].detach().cpu().numpy()
            return feats_np.reshape(-1, feats_np.shape[-1])


class DinoEmbeddingWrapper:
    """
    Simple callable wrapper around DinoPatchFeatureExtractor to fit the
    EmbeddingKMeansEvaluator API (takes img_rgb np.ndarray, returns np.ndarray).
    """
    def __init__(self, model_name="vit_small_patch16_224.dino", device=None):
        self.model = DinoPatchFeatureExtractor(model_name=model_name, device=device)

    def __call__(self, img_rgb_np: np.ndarray) -> np.ndarray:
        return self.model(img_rgb_np)  # (N, D)

