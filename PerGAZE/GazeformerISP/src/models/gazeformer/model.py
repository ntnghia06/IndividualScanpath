"""Original ISP transformer/head with online image and PerGAZE guidance inputs."""
import torch
from torch import nn
from torch.nn import functional as F

from models.resnet import resnet50
from models.gazeformer.transformer import Transformer
from models.gazeformer.heads import CrossAttentionPredictor, attention_module
from models.gazeformer.positional_encodings import PositionEmbeddingSine2d
from geometry import FEATURE_GRID, ACTION_GRID, ACTION_COUNT


class GazeformerISP(nn.Module):
    def __init__(self, args, manifest, pretrained=True):
        super().__init__()
        self.args = args
        self.grid = FEATURE_GRID
        if args.backbone_weights == "coco":
            if pretrained:
                from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
                body = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body
            else:
                from torchvision.models import resnet50 as torchvision_resnet50
                body = torchvision_resnet50(weights=None)
            self.backbone = nn.Sequential(*[getattr(body, key) for key in (
                "conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")])
        else:
            self.backbone = nn.Sequential(*list(resnet50(pretrained=pretrained).children())[:-2])
        self.freeze_backbone = not args.train_backbone
        if self.freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        args.im_h, args.im_w = self.grid
        args.subject_feature_dim = args.embedding_dim
        self.transformer = Transformer(d_model=args.hidden_dim, img_hidden_dim=2048,
            subject_feature_dim=args.embedding_dim, lm_dmodel=args.text_dim, nhead=args.nhead,
            num_encoder_layers=args.num_encoder, num_decoder_layers=args.num_decoder,
            dim_feedforward=args.hidden_dim, encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout, device="cpu", args=args)
        self.subject_embed = nn.Embedding(len(manifest["subjects"]), args.embedding_dim)
        self.word_embed = nn.Embedding(len(manifest["text_vocabulary"]), args.text_dim, padding_idx=0)
        self.query_embed = nn.Embedding(args.max_length, args.hidden_dim)
        self.patch_embed = PositionEmbeddingSine2d(self.grid, hidden_dim=args.hidden_dim, normalize=True, device="cpu")
        self.heads = nn.ModuleList([CrossAttentionPredictor(args.nhead, args.dropout, args.hidden_dim)
                                   for _ in range(args.action_map_num)])
        self.prioritization = attention_module(args.hidden_dim, args.embedding_dim, args.hidden_dim // 2)
        self.stop = nn.Linear(args.hidden_dim, args.action_map_num)
        self.duration_mu = nn.Linear(args.hidden_dim, 1)
        self.duration_logvar = nn.Linear(args.hidden_dim, 1)
        self.dropout = nn.Dropout(args.dropout)
        for module in (self.stop, self.duration_mu, self.duration_logvar):
            nn.init.normal_(module.weight, std=.01)
            nn.init.zeros_(module.bias)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def encode_text(self, tokens, mask, embeddings=None):
        if embeddings is None:
            valid = (tokens != 0).unsqueeze(-1)
            embeddings = (self.word_embed(tokens) * valid).sum(1) / valid.sum(1).clamp_min(1)
        return embeddings * mask[:, None]

    def forward(self, images, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        features = self.backbone(images)
        if features.shape[-2:] != self.grid:
            features = F.adaptive_avg_pool2d(features, self.grid)
        features = features.flatten(2).transpose(1, 2)
        return self.forward_features(features, subjects, attention_maps, task_tokens, task_mask, task_embeddings)

    def forward_features(self, features, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        text = self.encode_text(task_tokens, task_mask, task_embeddings)
        observer = self.subject_embed(subjects)
        guidance = F.interpolate(attention_maps, size=self.grid, mode="area").flatten(1)
        queries = features.new_zeros(self.args.max_length, features.shape[0], self.args.hidden_dim)
        positions = self.query_embed.weight[:, None]
        memory, output = self.transformer(src=features, tgt=queries, subjects=observer, task=text,
                                         querypos_embed=positions, patchpos_embed=self.patch_embed,
                                         guidance=guidance, task_mask=task_mask)
        output = self.dropout(output)
        maps = torch.stack([head(output, memory, querypos_embed=positions, patchpos_embed=self.patch_embed)
                            for head in self.heads], dim=-2)
        weights = self.prioritization(memory, maps, observer)
        tokens = self.stop(output).permute(1, 0, 2)
        logits = torch.cat([tokens.unsqueeze(-1), maps], -1)
        logits = (logits * weights.unsqueeze(-1)).sum(2)
        # Image tokens use 24x32; fixation cells use 22x32 on a 352x512 frame.
        spatial = F.interpolate(logits[:, :, 1:].reshape(-1, 1, *self.grid), size=ACTION_GRID,
                                mode="bilinear", align_corners=False).reshape(features.shape[0], self.args.max_length, ACTION_COUNT - 1)
        logits = torch.cat([logits[:, :, :1], spatial], -1)
        return {"actions" if self.training else "all_actions_prob": logits if self.training else logits.softmax(-1),
                "log_normal_mu": self.duration_mu(output).permute(1, 0, 2).squeeze(-1),
                "log_normal_sigma2": self.duration_logvar(output).clamp(-10, 10).exp().permute(1, 0, 2).squeeze(-1)}
