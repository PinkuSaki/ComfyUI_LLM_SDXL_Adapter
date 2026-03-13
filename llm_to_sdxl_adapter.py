import torch
import torch.nn as nn
import logging

logger = logging.getLogger("LLM-SDXL-Adapter")


def pad_to_length(tensor, target_length, dim=1, value=0):
    """Universal function for padding tensors"""
    current_length = tensor.size(dim)

    if current_length >= target_length:
        return tensor.narrow(dim, 0, target_length)

    pad_size = list(tensor.shape)
    pad_size[dim] = target_length - current_length

    padding = torch.full(
        pad_size,
        value,
        device=tensor.device,
        dtype=tensor.dtype
    )

    return torch.cat([tensor, padding], dim=dim)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads, batch_first=True, dropout=dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, mask=None):
        # Self-attention
        normed = self.norm1(x)

        # Use key_padding_mask instead of attn_mask
        if mask is not None:
            # key_padding_mask: True means "ignore this token"
            # Our mask: 1 = real token, 0 = padding
            # So we invert
            key_padding_mask = ~mask.bool()
        else:
            key_padding_mask = None

        attn_out, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x


class LLMToSDXLAdapter(nn.Module):
    """
    Universal adapter for converting any LLM embeddings to SDXL format
    Supports various LLM architectures (Gemma, Llama, Mistral, etc.)
    """
    def __init__(self,
                 llm_dim=1152,              # Changed from gemma_dim to llm_dim
                 sdxl_seq_dim=2048,
                 sdxl_pooled_dim=1280,
                 max_input_len=512,
                 target_seq_len=308,
                 n_wide_blocks=3,        # Blocks BEFORE compression
                 n_narrow_blocks=3,      # Blocks AFTER compression
                 num_heads=16,
                 dropout=0):
        super().__init__()

        self.max_input_len = max_input_len
        self.target_seq_len = target_seq_len
        self.num_heads = num_heads

        # Projections
        self.seq_projection = None
        if llm_dim != sdxl_seq_dim:
            self.seq_projection = nn.Linear(llm_dim, sdxl_seq_dim)

        # Positional embeddings for full sequence
        self.input_position_embeddings = nn.Parameter(
            torch.randn(1, max_input_len, sdxl_seq_dim)
        )
        # Positional embeddings for compressed sequence
        self.output_position_embeddings = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        
        # Wide blocks - processing full sequence (512 tokens)
        self.wide_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_wide_blocks)
        ])

        # Compression: Cross-attention with learnable queries
        self.compression_queries = nn.Parameter(
            torch.randn(1, target_seq_len, sdxl_seq_dim)
        )
        self.compression_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )
        # Norm layer after compression for stability
        self.compression_norm = nn.LayerNorm(sdxl_seq_dim)
        # Optional gate mechanism for weighting information
        self.compression_gate = nn.Sequential(
            nn.Linear(sdxl_seq_dim * 2, sdxl_seq_dim),
            nn.Sigmoid()
        )

        # Narrow blocks - processing compressed sequence (308 tokens)
        self.narrow_attention_blocks = nn.ModuleList([
            TransformerBlock(sdxl_seq_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(n_narrow_blocks)
        ])
        
        # Pooling head - now works with processed sequence
        self.pooling_attention = nn.MultiheadAttention(
            embed_dim=sdxl_seq_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # Learnable [CLS]-like token for pooling
        self.pooling_token = nn.Parameter(torch.randn(1, 1, sdxl_seq_dim))

        # Final projection for pooled embeddings
        self.pooled_projection = nn.Sequential(
            nn.Linear(sdxl_seq_dim, sdxl_seq_dim),
            nn.LayerNorm(sdxl_seq_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(sdxl_seq_dim, sdxl_pooled_dim)
        )

    def _run_compression_attention(self, queries, hidden_states, attention_mask, need_weights):
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None

        return self.compression_attention(
            queries,
            hidden_states,
            hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )

    def _compute_query_weights(self, compression_weights, token_weights, attention_mask, dtype):
        """把 source token 权重汇总成 compressed token 级别的插值权重。"""
        if compression_weights is None or token_weights is None:
            return None

        attn = compression_weights.mean(dim=1)
        safe_weights = token_weights.to(dtype=dtype).clamp(1e-3, 4.0)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=dtype)
            valid_weights = torch.where(mask > 0, safe_weights, torch.ones_like(safe_weights))
            mean_weight = (valid_weights * mask).sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        else:
            valid_weights = safe_weights
            mean_weight = valid_weights.mean(dim=-1, keepdim=True)

        normalized_weights = valid_weights / mean_weight.clamp(min=1e-3)
        query_weights = torch.matmul(attn, normalized_weights.unsqueeze(-1)).squeeze(-1)

        return query_weights.clamp(1e-3, 4.0)

    def _should_apply_weighted_prompt(self, token_weights, attention_mask, dtype):
        if token_weights is None:
            return False

        weights = token_weights.to(dtype=dtype)
        if attention_mask is not None:
            weights = weights[attention_mask > 0]
        else:
            weights = weights.reshape(-1)

        if weights.numel() == 0:
            return False

        return not torch.allclose(
            weights,
            torch.ones_like(weights),
            atol=1e-6,
            rtol=0.0,
        )

    def _prepare_sequence_inputs(self, llm_hidden_states, attention_mask=None, token_weights=None):
        batch_size, seq_len, _ = llm_hidden_states.shape

        if self.seq_projection:
            hidden_states = self.seq_projection(llm_hidden_states)
        else:
            hidden_states = llm_hidden_states

        if seq_len > self.max_input_len:
            hidden_states = hidden_states[:, :self.max_input_len, :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_input_len]
            if token_weights is not None:
                token_weights = token_weights[:, :self.max_input_len]
        elif seq_len < self.max_input_len:
            hidden_states = pad_to_length(hidden_states, self.max_input_len, dim=1)
            if attention_mask is not None:
                attention_mask = pad_to_length(attention_mask, self.max_input_len, dim=1, value=0)
            else:
                attention_mask = torch.ones(batch_size, self.max_input_len, device=hidden_states.device)
                attention_mask[:, seq_len:] = 0
            if token_weights is not None:
                token_weights = pad_to_length(token_weights, self.max_input_len, dim=1, value=1.0)

        if token_weights is not None and attention_mask is None:
            attention_mask = torch.ones(batch_size, hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)

        return hidden_states, attention_mask, token_weights

    def _encode_input_sequence(self, hidden_states, attention_mask):
        hidden_states = hidden_states + self.input_position_embeddings

        for block in self.wide_attention_blocks:
            hidden_states = block(hidden_states, attention_mask)

        return hidden_states

    def _apply_output_stack(self, sequence):
        """共享 gate 之后到 pooled 之前的后处理路径。"""
        sequence = self.compression_norm(sequence)
        sequence = sequence + self.output_position_embeddings

        for block in self.narrow_attention_blocks:
            sequence = block(sequence)

        return sequence

    def _encode_compressed_sequence(self, hidden_states, attention_mask, need_weights):
        batch_size = hidden_states.shape[0]
        queries = self.compression_queries.expand(batch_size, -1, -1)
        compressed_sequence, compression_weights = self._run_compression_attention(
            queries,
            hidden_states,
            attention_mask,
            need_weights=need_weights,
        )

        gate_input = torch.cat([queries, compressed_sequence], dim=-1)
        gate_weights = self.compression_gate(gate_input)
        compressed_sequence = gate_weights * compressed_sequence + (1 - gate_weights) * queries
        compressed_sequence = self._apply_output_stack(compressed_sequence)

        return compressed_sequence, compression_weights

    def forward(
        self,
        llm_hidden_states,
        attention_mask=None,
        token_weights=None,
        empty_prompt_hidden_states=None,
        empty_prompt_attention_mask=None,
    ):
        hidden_states, attention_mask, token_weights = self._prepare_sequence_inputs(
            llm_hidden_states,
            attention_mask=attention_mask,
            token_weights=token_weights,
        )
        hidden_states = self._encode_input_sequence(hidden_states, attention_mask)

        use_weighted_prompt = self._should_apply_weighted_prompt(
            token_weights,
            attention_mask,
            hidden_states.dtype,
        )
        compressed_sequence, compression_weights = self._encode_compressed_sequence(
            hidden_states,
            attention_mask,
            need_weights=use_weighted_prompt,
        )

        if use_weighted_prompt:
            query_weights = self._compute_query_weights(
                compression_weights,
                token_weights,
                attention_mask,
                hidden_states.dtype,
            )
            if empty_prompt_hidden_states is None or empty_prompt_attention_mask is None:
                raise ValueError(
                    "token_weights 已提供，但缺少 empty_prompt_hidden_states 或 "
                    "empty_prompt_attention_mask。请将 T5GEMMATextEncoder 的空提示输出连接到 "
                    "Apply T5Gemma LLM to Adapter。"
                )

            neutral_hidden_states, neutral_attention_mask, _ = self._prepare_sequence_inputs(
                empty_prompt_hidden_states,
                attention_mask=empty_prompt_attention_mask,
            )
            neutral_hidden_states = self._encode_input_sequence(
                neutral_hidden_states,
                neutral_attention_mask,
            )
            neutral_sequence, _ = self._encode_compressed_sequence(
                neutral_hidden_states,
                neutral_attention_mask,
                need_weights=False,
            )
            compressed_sequence = neutral_sequence + query_weights.unsqueeze(-1) * (compressed_sequence - neutral_sequence)

        # ===== STAGE 4: Pooling for Vector Embeddings =====
        # Pool the compressed sequence for vector embeddings
        batch_size = compressed_sequence.shape[0]
        pooling_tokens = self.pooling_token.expand(batch_size, -1, -1)
        pooled_output, _ = self.pooling_attention(
            pooling_tokens,
            compressed_sequence,
            compressed_sequence,
            need_weights=False
        )
        pooled_output = pooled_output.squeeze(1)  # Remove sequence dimension

        # Final projection for pooled embeddings
        pooled_output = self.pooled_projection(pooled_output)

        return compressed_sequence, pooled_output 
