import torch
import torch.nn as nn
import math

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class SpecDirectEmbed(nn.Module):
    def __init__(self, d_model=512, src_vocab=1000) -> None:
        super(SpecDirectEmbed, self).__init__()
        self.d_model = d_model
        self.embed = Embeddings(d_model, src_vocab)
    def forward(self, spec):
        return self.embed(spec.to(torch.int)).squeeze(1) #[batch_size, spec_len, d_model]       


class EmbedPatchAttention(nn.Module):
    def __init__(self, spec_len=3200, patch_len=8, d_model=512, src_vocab=1000) -> None:
        super(EmbedPatchAttention, self).__init__()
        assert spec_len % patch_len == 0, "Patch length {} doesn't match spectra length {}".format(patch_len, spec_len)
        self.patch_len = patch_len
        self.d_model = d_model
        self.embed = Embeddings(d_model, src_vocab)
        self.patch = nn.Linear(patch_len*d_model, d_model)
        self.attention = MultiHeadedAttention(h=8, d_model=512)
        
    def forward(self, spec):
        batch_size = spec.shape[0]  # spec: (B, spec_len)
        spec = self.embed(spec.to(torch.int)) # (batch_size, spec_len, d_model)
        # 分 patch
        num_patches = spec.shape[1] // self.patch_len
        spec = spec.view(batch_size, num_patches, self.patch_len, self.d_model) # (batch_size, num_patches, patch_size, d_model)
        # Attention 在 patch 内部执行（token 间）
        spec = spec.view(batch_size * num_patches, self.patch_len, self.d_model) # (batch_size * num_patches, patch_size, d_model)
        spec = self.attention(spec, spec, spec)
        spec = spec.view(batch_size, num_patches, self.patch_len * self.d_model) # (batch_size, num_patches, patch_size * d_model)
        
        return self.patch(spec) # (batch_size, num_patches, d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query, key, value

        return self.linears[-1](x)
 
    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))
    

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, d_model=512, d_ff=2048, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(h, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayers = nn.ModuleList(
            [SublayerConnection(d_model, dropout) for _ in range(2)]
        )
        self.d_model = d_model

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)
    

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding = EmbedPatchAttention(
            spec_len=config.spec_len, 
            patch_len=config.patch_len, 
            d_model=config.d_model, 
            src_vocab=1000
        )
        self.pos_encoding = PositionalEncoding(config.d_model, config.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(config.d_model, config.d_ff, config.num_heads, config.dropout) for _ in range(config.num_layers)]
        )
        self.norm = LayerNorm(config.d_model)
        self.quant_conv = nn.Linear(config.d_model, 2 * config.d_model, bias=False)

    def forward(self, x, mask=None, return_states=False):
        "Pass the input (and mask) through each layer in turn."
        """
        Args:
            x:                (B, spec_len)，整数化光谱
            mask:             (B, N) 可选的 attention mask（True=有效）
            return_states:   True  -> 返回所有层的输出状态
                              False -> 只返回最后一层的输出

        Returns:
            return_states=True:  (B, N, D) 的张量，包含所有层的输出状态
            return_states=False:  (B, N, D) 的张量，只包含最后一层的输出
        """
        # Embedding + Positional Encoding: (B, N, D)
        x = self.embedding(x)   # (batch_size, spec_len/patch_size, d_model)
        x = self.pos_encoding(x)

        # Transformer Encoder layers
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.norm(x)
        x = self.quant_conv(x)  # (B, N, 2*D) 

        if return_states:
            mu, logvar = x.chunk(2, dim=-1)  # (B, N, D), (B, N, D)
            logvar = torch.clamp(logvar, -30.0, 20.0)
            return mu, logvar