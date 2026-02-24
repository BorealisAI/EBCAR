# Copyright (c) 2025-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x,
        attn_mask=None,
        key_padding_mask=None,
        causal_mask=False,
        return_attention=False,
    ):
        B, T, C = x.size()

        Q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim**0.5)

        if causal_mask:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn_logits = attn_logits.masked_fill(
                mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )

        if key_padding_mask is not None:
            padding_mask = key_padding_mask[:, None, None, :].to(torch.bool)
            attn_logits = attn_logits.masked_fill(padding_mask, float("-inf"))

        if attn_mask is not None:
            attn_logits += attn_mask

        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights_for_return = attn_weights.clone() if return_attention else None
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)

        if return_attention:
            return self.out_proj(attn_output), attn_weights_for_return
        return self.out_proj(attn_output)


class TransformerEncoderLayerHybridAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()

        # Shared attention (no mask)
        self.shared_attn = MultiheadAttention(embed_dim, num_heads, dropout)

        # Dedicated attention (with mask)
        self.dedicated_attn = MultiheadAttention(embed_dim, num_heads, dropout)

        # Feed Forward
        self.linear1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.linear2 = nn.Linear(ff_hidden_dim, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src,
        attn_mask=None,
        key_padding_mask=None,
        causal_mask=False,
        return_attention=False,
    ):
        src2 = self.norm1(src)

        # Shared full attention (no mask)
        if return_attention:
            shared_out, shared_attn_weights = self.shared_attn(
                src2, None, key_padding_mask, causal_mask, return_attention=True
            )  # Always have no attention mask so that the model can attend to all the passages

            # Dedicated masked attention
            dedicated_out, dedicated_attn_weights = self.dedicated_attn(
                src2, attn_mask, key_padding_mask, causal_mask, return_attention=True
            )  # Only attend to the passages that are from the same document
        else:
            shared_out = self.shared_attn(
                src2, None, key_padding_mask, causal_mask
            )  # Always have no attention mask so that the model can attend to all the passages

            # Dedicated masked attention
            dedicated_out = self.dedicated_attn(
                src2, attn_mask, key_padding_mask, causal_mask
            )  # Only attend to the passages that are from the same document

        # Combine both attentions
        src = src + self.dropout1(shared_out + dedicated_out)

        # Feed forward layer
        src2 = self.linear2(self.dropout(F.relu(self.linear1(self.norm2(src)))))
        src = src + self.dropout2(src2)

        if return_attention:
            return src, (shared_attn_weights, dedicated_attn_weights)
        return src


class TransformerEncoderWithHybridAttention(nn.Module):
    def __init__(
        self, num_layers, embed_dim, num_heads, ff_hidden_dim=2048, dropout=0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerHybridAttention(
                    embed_dim, num_heads, ff_hidden_dim, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        src,
        attn_mask=None,
        key_padding_mask=None,
        causal_mask=False,
        return_attention=False,
    ):
        all_attentions = [] if return_attention else None
        for layer in self.layers:
            if return_attention:
                src, attn_weights = layer(
                    src, attn_mask, key_padding_mask, causal_mask, return_attention=True
                )
                all_attentions.append(attn_weights)
            else:
                src = layer(src, attn_mask, key_padding_mask, causal_mask)

        if return_attention:
            return src, all_attentions
        return src
