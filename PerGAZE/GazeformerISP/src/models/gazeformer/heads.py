import torch
import torch.nn.functional as F
from torch import nn, Tensor

import math
from typing import Optional

eps = 1e-16

class attention_module(nn.Module):
    def __init__(self, feature_size, subject_embedding_size, project_size):
        super(attention_module, self).__init__()
        self.feature_size = feature_size
        self.project_size = project_size

        self.proj_feature = nn.Linear(feature_size, project_size, bias=True)
        self.proj_subject = nn.Linear(subject_embedding_size, project_size, bias=True)
        self.attention = nn.Linear(project_size, 1, bias=True)

        self.init_weights()

    def forward(self, features, attention_map, subject_embedding):
        features = features.permute(1, 2, 0)
        aggr_feature = features.unsqueeze(1).unsqueeze(1) * attention_map.unsqueeze(-2)
        aggr_feature = aggr_feature.mean(-1)

        proj_aggr_feature = self.proj_feature(aggr_feature)
        proj_subject_embedding = self.proj_subject(subject_embedding)
        attention_weights = F.softmax(
            self.attention(torch.tanh(proj_aggr_feature + proj_subject_embedding.unsqueeze(1).unsqueeze(1))), 2)
        attention_weights = attention_weights.squeeze(-1)
        return attention_weights

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

class CrossAttentionPredictor(nn.Module):
    def __init__(self, nhead = 8, dropout=0.4, d_model = 512):
        super(CrossAttentionPredictor, self).__init__()
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self._reset_parameters(self.self_attn)
        self._reset_parameters(self.multihead_attn)

    def _reset_parameters(self, mod):
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else pos + tensor

    def forward(self, tgt, memory, tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                    querypos_embed: Optional[Tensor] = None,
                    patchpos_embed: Optional[Tensor] = None):
        q = k = v = self.with_pos_embed(tgt, querypos_embed)
        tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        att = self.multihead_attn(query=self.with_pos_embed(tgt, querypos_embed),
                                   key=patchpos_embed(memory),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[1]
        att_logit = torch.log(att + eps)

        return att_logit
