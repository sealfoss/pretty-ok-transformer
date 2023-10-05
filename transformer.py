import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy

from torch import Tensor

#Idea for write up. Follow data path of inference. String >> Tokens >> MatMuls Loop >> Output >> String

# Attention(Q, K, V) = softmax( (QK^T)/sqrt(dk) ) * V
def ScaledDotProductAttention(Q: Tensor, K: Tensor, V: Tensor) -> Tensor:
    numerator = Q.bmm(K.transpose(1,2))
    denomenator = math.sqrt(Q.size(-1))
    quotient = numerator / denomenator
    attention = quotient.bmm(V)
    return attention
