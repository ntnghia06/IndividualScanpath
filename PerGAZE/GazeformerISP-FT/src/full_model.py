"""Unfreeze the online ResNet without changing the base/ISP-S behavior."""
import torch
from models.gazeformer.model import GazeformerISP
from geometry import FEATURE_GRID
from torchvision.ops.misc import FrozenBatchNorm2d


def unfreeze_batch_norm(module):
    for name, child in list(module.named_children()):
        if isinstance(child, FrozenBatchNorm2d):
            normal = torch.nn.BatchNorm2d(child.weight.numel(), eps=child.eps)
            with torch.no_grad():
                normal.weight.copy_(child.weight)
                normal.bias.copy_(child.bias)
                normal.running_mean.copy_(child.running_mean)
                normal.running_var.copy_(child.running_var)
            setattr(module, name, normal)
        else:
            unfreeze_batch_norm(child)


class GazeformerFullFT(GazeformerISP):
    def __init__(self, args, manifest, pretrained=True):
        super().__init__(args, manifest, pretrained=pretrained)
        # Detection COCO backbones use FrozenBatchNorm buffers, not learnable affine parameters.
        unfreeze_batch_norm(self.backbone)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state_dict = dict(state_dict)
        if any(key.startswith("backbone.") for key in state_dict):
            for key, value in self.named_buffers():
                if key.startswith("backbone.") and key.endswith("num_batches_tracked"):
                    state_dict.setdefault(key, value.clone())
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def train(self, mode=True):
        torch.nn.Module.train(self, mode)
        return self

    def forward(self, images, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        features = self.backbone(images)  # Autograd enabled: backbone is part of FT.
        if features.shape[1:] != (2048, *FEATURE_GRID):
            raise ValueError("Expected 1024x768 images producing 24x32 features")
        features = features.flatten(2).transpose(1, 2)
        return self.forward_features(features, subjects, attention_maps, task_tokens, task_mask, task_embeddings)
