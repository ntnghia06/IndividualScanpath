"""ISP transformer/head using frozen COCO features and sentence embeddings."""
import torch
from torch import nn

from models.gazeformer.transformer import Transformer
from models.gazeformer.heads import CrossAttentionPredictor, attention_module
from models.gazeformer.positional_encodings import PositionEmbeddingSine2d
from geometry import FEATURE_GRID


class GazeformerISP(nn.Module):
    def __init__(self, args, manifest, pretrained=True):
        super().__init__()
        self.args = args
        self.grid = FEATURE_GRID
        args.im_h, args.im_w = self.grid
        args.subject_feature_dim = args.embedding_dim
        self.transformer = Transformer(d_model=args.hidden_dim, img_hidden_dim=2048,
            subject_feature_dim=args.embedding_dim, lm_dmodel=args.text_dim, nhead=args.nhead,
            num_encoder_layers=args.num_encoder, num_decoder_layers=args.num_decoder,
            dim_feedforward=args.hidden_dim, encoder_dropout=args.encoder_dropout,
            decoder_dropout=args.decoder_dropout, device="cpu", args=args)
        self.subject_embed = nn.Embedding(len(manifest["subjects"]), args.embedding_dim)
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

    def encode_text(self, tokens, mask, embeddings=None):
        if embeddings is None:
            raise ValueError("Precomputed SentenceTransformer embeddings are required")
        return embeddings

    def forward(self, features, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        if features.ndim == 4:
            if features.shape[1:] != (2048, *self.grid):
                raise ValueError("Expected cached COCO features [batch, 2048, 24, 32]")
            features = features.flatten(2).transpose(1, 2)
        if features.shape[1:] != (self.grid[0] * self.grid[1], 2048):
            raise ValueError("Invalid cached image feature shape")
        return self.forward_features(features, subjects, attention_maps, task_tokens, task_mask, task_embeddings)

    def forward_features(self, features, subjects, attention_maps, task_tokens, task_mask, task_embeddings=None):
        text = self.encode_text(task_tokens, task_mask, task_embeddings)
        observer = self.subject_embed(subjects)
        queries = features.new_zeros(self.args.max_length, features.shape[0], self.args.hidden_dim)
        positions = self.query_embed.weight[:, None]
        memory, output = self.transformer(src=features, tgt=queries, subjects=observer, task=text,
                                         querypos_embed=positions, patchpos_embed=self.patch_embed)
        output = self.dropout(output)
        maps = torch.stack([head(output, memory, querypos_embed=positions, patchpos_embed=self.patch_embed)
                            for head in self.heads], dim=-2)
        weights = self.prioritization(memory, maps, observer)
        tokens = self.stop(output).permute(1, 0, 2)
        logits = torch.cat([tokens.unsqueeze(-1), maps], -1)
        logits = (logits * weights.unsqueeze(-1)).sum(2)
        return {"actions" if self.training else "all_actions_prob": logits if self.training else logits.softmax(-1),
                "log_normal_mu": self.duration_mu(output).permute(1, 0, 2).squeeze(-1),
                "log_normal_sigma2": self.duration_logvar(output).clamp(-10, 10).exp().permute(1, 0, 2).squeeze(-1)}
