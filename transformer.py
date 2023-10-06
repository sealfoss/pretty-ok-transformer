import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from torch import Tensor

#Idea for write up. Follow data path of inference. String >> Tokens >> MatMuls Loop >> Output >> String

# Attention(Q, K, V) = softmax( (Q * K^T)/sqrt(dk) ) * V
def Attention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    numerator = Q.bmm(K.transpose(1,2))
    denomenator = math.sqrt(Q.size(-1))
    quotient = numerator / denomenator
    attention = quotient.bmm(V)
    return attention

#PE(pos, 2_i) = sin(pos/10000^(2i/d_model))
#PE(pos, 2_i+1) = cos(pos/10000^(2i/d_model))
def PositionEncoding(
        seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")
) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device = device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (10000 ** (dim / dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

"""
Each of the layers in our encoder and decoder contains a fully connected feed-forward network, 
which … consists of two linear transformations with a ReLU activation in between. The dimensionality 
of input and output is 512, and the inner-layer has dimensionality 2048.
"""
def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input)
    )

# MultiHead(Q,K,V) = Concat(head_1, head_2, ... , head_h) * W^O 
# Where head_i = Attention(Q * W^Q_i, K * W^K_i, V * W^V_i)
class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init()
        # Y = X dot W^T + b, where X is the input tensor, W is teh weight matrix,
        #b is the bias vector and Y is the output tensor.
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key:Tensor, value: Tensor) -> Tensor:
        return Attention(self.q(query), self.k(key), self.v(value))
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        # create list of Attention Head modules of length num_heads, with given values
        # for dim_in, dim_q, dim_k
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for x in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

        def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
            return self.linear(
                torch.cat(
                    [h(query, key, value) for h in self.heads],
                    dim = -1
                )
            )
        
class Residual(nn.Module):
    def __init(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assumes query tensor is given first so risdual can be computed.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))
    
class EncoderLayer(nn.Module):
    def __init__(
            self,
            dim_model: int = 512,
            num_heads: int = 6,
            dim_feedforward: int = 2048,
            dropout: float = 0.1
    ):
        
        super().__init__()

        dim_q = dim_k = max(dim_model // num_heads, 1)

        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k),
            dimension=dim_model,
            dropout=dropout
        )

        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            droput=dropout
        )

    def forward(self, src: Tensor) -> Tensor:
        src = self.attention(src, src, src)
        return self.feed_forward(src)

class Encoder(nn.Module):
    def __inti__(
        self,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init()
        self.layers == nn.ModuleList(
            [
                EncoderLayer(dim_model, num_heads, dim_feedforward, dropout)
                for x in range(num_layers)
            ]
        )
    
    def forward(self, src: Tensor) -> Tensor:
        seq_len, dimension = src.size(1), src.size(2)
        src += PositionEncoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)
