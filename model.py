import torch
import math
from torch import nn

# nn.Embedding: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html


class InputEmbeddings(nn.Module):
    """Some Information about InputEmbeddings"""

    def __init__(self, vocab_size: int, d_model: int):
        super(InputEmbeddings, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # input x : (batch, seq_len) consisting of token ids
        x = self.embedding(x) + math.sqrt(self.d_model)
        return x


class Postional_Encoding(nn.Module):
    """Some Information about Postional_Encoding"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(Postional_Encoding, self).__init__()

        # self.dropout = nn.Dropout(p=dropout)

        # matrix pos_emb (max_len, d_model)
        pos_emb = torch.zeros(max_len, d_model)

        # positions matrix(max_len, 1): useful to multiply each index in max_len
        position = torch.arange(max_len).unsqueeze(1)

        # formulae: denominator is taken in exponent form for simpler calculation
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # formulae: sin at even pos and cos at odd pos
        pos_emb[:, 0::2] = torch.sin(position * div_term)
        pos_emb[:, 1::2] = torch.cos(position * div_term)

        # accomodate batch_size
        pos_emb = pos_emb.unsqueeze(0)

        # save pos_emb in buffer to avoid updation with parameters.
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x):
        x = x + self.pos_emb[:, : x.shape[1], :]
        return x


class TransformerEmbedding(nn.Module):
    """Some Information about TransformerEmbedding"""

    def __init__(self, vocab_size: int, max_len: int, d_model: int, dropout: float):
        super(TransformerEmbedding, self).__init__()

        self.embedding = InputEmbeddings(vocab_size, d_model)
        self.pos_embedding = Postional_Encoding(d_model, max_len)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        emb = self.embedding(x)
        pos_emb = self.pos_embedding(emb)
        return self.dropout(emb + pos_emb)


# https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html


class LayerNormalisation(nn.Module):
    """Some Information about LayerNormalisation"""

    def __init__(self, d_model: int, eps: float = 1e-12):
        super(LayerNormalisation, self).__init__()

        self.eps = eps
        # nn.Parameter: A kind of Tensor that is to be considered a module parameter.

        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # input x: (batch, seq_len, d_model/hidden_size)

        # mean : https://pytorch.org/docs/stable/generated/torch.mean.html
        mean = x.mean(dim=-1, keepdim=True)  # dim (batch, seq_len, 1)

        # std : https://pytorch.org/docs/stable/generated/torch.std.html
        std = x.std(dim=-1, keepdim=True)  # dim (batch, seq_len, 1)

        # layernorm formula
        output = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return output


# nn.Linear: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html


class FeedForwardLayer(nn.Module):
    """Some Information about FeedForwardLayer"""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1):
        super(FeedForwardLayer, self).__init__()

        # FFN(x) = max(0, xW1 + b1)W2 + b2
        self.linear1 = nn.Linear(d_model, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden, d_model)

    def forward(self, x):
        # input x: (batch, seq_len, d_model)
        x = self.linear1(x)
        # x : (batch, seq_len, d_ff)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # x : (batch, seq_len, d_model)

        return x


class MultiHeadAttentionBlock(nn.Module):
    """Some Information about MultiHeadAttentionBlock"""

    def __init__(self, d_model: int, h: int):
        super(MultiHeadAttentionBlock, self).__init__()

        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "model dimension not divisible by heads"
        # key, query, value and O matrix (d_model, d_model)

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def attention(query, key, value, mask):
        # query, key, value, mask dim (batch, head, seq_len, d_k)

        d_k = query.shape[-1]  # shape of the d_model

        # attention formula
        # key.transpose(-2, -1): (batch, head, d_k, seq_len)
        # attention_score: (batch, head, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # fill the remaining spaces with mask
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # q, k, v, mask : (batch, seq_len, d_model)
        # query, key, value : (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # split the input to heads -> d_model = h*d_k
        # final shape (batch, seq_len, head, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k)

        # transpose  (batch, seq_len, head, d_k) ->  (batch, head, seq_len, d_k)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask
        )

        # combine back all the head: (batch, h, seq_len, d_k) --> (batch, seq_len, d_model)
        # contiguous: https://pytorch.org/docs/stable/generated/torch.Tensor.contiguous.html
        # contiguous: https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
        x = x.transpose(1, 2).contiguous()  # (batch, seq_len, h, d_k)
        x = x.view(x.shape[0], -1, self.h * self.d_k)  # (batch, seq_len, d_model)

        x = self.w_o(x)  # (batch, seq_len, d_model)

        return x


class EncoderBlock(nn.Module):
    """Some Information about EncoderBlock"""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float):
        super(EncoderBlock, self).__init__()

        # architecture of a encoder layer
        self.self_attention_layer = MultiHeadAttentionBlock(d_model, n_head)
        self.layer_norm_1 = LayerNormalisation(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)

        self.feed_forward_layer = FeedForwardLayer(d_model, ffn_hidden, dropout)
        self.layer_norm_2 = LayerNormalisation(d_model)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        _x = x

        # compute self-attention
        x = self.self_attention_layer(x, x, x, src_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x + _x)

        # feed forward layer
        _x = x
        x = self.feed_forward_layer(x)
        x = self.dropout_2(x)
        x = self.layer_norm_2(x + _x)

        return x


class DecoderBlock(nn.Module):
    """Some Information about DecoderBlock"""

    def __init__(self, d_model: int, ffn_hidden: int, n_head: int, dropout: float):
        super(DecoderBlock, self).__init__()

        # decoder architecture
        self.self_attention = MultiHeadAttentionBlock(d_model, n_head)
        self.layer_norm_1 = LayerNormalisation(d_model)
        self.dropout_1 = nn.Dropout(p=dropout)

        self.cross_attention = MultiHeadAttentionBlock(d_model, n_head)
        self.layer_norm_2 = LayerNormalisation(d_model)
        self.dropout_2 = nn.Dropout(p=dropout)

        self.feed_forward_layer = FeedForwardLayer(d_model, ffn_hidden, dropout)
        self.layer_norm_3 = LayerNormalisation(d_model)
        self.dropout_3 = nn.Dropout(p=dropout)

    def forward(self, dec, enc, trg_mask, src_mask):
        _x = dec

        x = self.self_attention(dec, dec, dec, trg_mask)
        x = self.dropout_1(x)
        x = self.layer_norm_1(x + _x)

        if enc is not None:
            _x = x
            x = self.cross_attention(dec, enc, enc, src_mask)
            x = self.dropout_2(x)
            x = self.layer_norm_2(x + _x)

        _x = x
        x = self.feed_forward_layer(x)
        x = self.dropout_3(x)
        x = self.layer_norm_3(x + _x)

        return x


class Encoder(nn.Module):
    """Some Information about Encoder"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        n_layer: int,
        d_model: int,
        n_head: int,
        ffn_hidden: int,
        dropout: float,
    ):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, max_len, d_model, dropout)

        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layer)]
        )

    def forward(self, x, src_mask):
        # Input x : (batch, seq_len)
        # src/trg_mask : (seq_len, seq_len)

        x = self.embedding(x)  # (batch, seq_len, d_model)

        for layer in self.layers:
            x = layer(x, src_mask)  # (batch, seq_len, d_model)

        return x


class Decoder(nn.Module):
    """Some Information about Decoder"""

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        n_layer: int,
        d_model: int,
        n_head: int,
        ffn_hidden: int,
        dropout: float,
    ):
        super(Decoder, self).__init__()

        self.embedding = TransformerEmbedding(vocab_size, max_len, d_model, dropout)

        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, ffn_hidden, n_head, dropout) for _ in range(n_layer)]
        )

        # self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, trg, enc_out, src_mask, trg_mask):
        # trg : (batch, seq_len)
        # enc_out : (batch, seq_len, d_model)
        # src/trg_mask : (seq_len, seq_len)

        x = self.embedding(trg)

        for layer in self.layers:
            x = layer(trg, enc_out, src_mask, trg_mask)

        # x = self.linear(x)  # (batch, vocab_size)

        return x


class ProjectionLayer(nn.Module):
    """Some Information about ProjectionLayer"""

    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        # self.softmax = torch.log_softmax()

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    """Some Information about Transformer"""

    def __init__(
        self,
        src_vocab_size: int,
        trg_vocab_size: int,
        src_seq_len: int,
        trg_seq_len: int,
        n_encoder: int,
        n_decoder: int,
        d_model: int,
        n_head: int,
        d_ffn: int,
        dropout: float,
    ):
        super(Transformer, self).__init__()
        self.src_vs = src_vocab_size
        self.trg_vs = trg_vocab_size

        # encoders
        self.encoder = Encoder(
            src_vocab_size, src_seq_len, n_encoder, d_model, n_head, d_ffn, dropout
        )

        # decoder
        self.decoder = Decoder(
            trg_vocab_size, trg_seq_len, n_decoder, d_model, n_head, d_ffn, dropout
        )

        # projection layer
        self.projection = ProjectionLayer(d_model, trg_vocab_size)

    def encode(self, x, src_mask):
        return self.encoder(x, src_mask)

    def decode(self, trg, enc_out, src_mask, tgt_mask):
        return self.decode(trg, enc_out, src_mask, tgt_mask)

    def project(self, x):
        return self.projection(x)

    def forward(self, src_in, tgt_in, src_mask, tgt_mask):
        # input x: (batch, src_seq_len)
        enc_out = self.encode(src_in, src_mask)
        dec_out = self.decode(tgt_in, enc_out, src_mask, tgt_mask)
        x = self.project(dec_out)
        return x
