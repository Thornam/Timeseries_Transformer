import torch
import torch.nn as nn 
import math
from torch import nn, Tensor

class PositionalEncoder(nn.Module):
    """
    Positional Encoder from the original paper
    (Vaswani et al, 2017)
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
        """
        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)

        #print(f'PE dimensions: {pe.size()}')
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)


class T2V(nn.Module):
    """
    The Time2Vector positional Encoder
    """
    def __init__(self, input_length, batch_size, device, dropout=0.1):
        super(T2V, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.input_length = input_length
        self.batch_size = batch_size
        
        self.tau  = torch.arange(start= 0, end=self.input_length, device=device)
        self.tau  = self.tau.expand(self.batch_size, self.input_length)

        self.w0   = nn.Parameter(torch.rand((1), device=device))
        self.p0   = nn.Parameter(torch.rand((1), device=device))
        self.w    = nn.Parameter(torch.rand((self.input_length -1), device=device))
        self.p    = nn.Parameter(torch.rand((self.input_length -1), device=device))
        self.f    = torch.sin
        self.double()

    def forward(self, x):
        v0    = self.w0 * self.tau[:, 0] + self.p0
        v1    = self.f( torch.add( torch.mul(self.tau[:, 1:], self.w.reshape(-1,1).t()), self.p.reshape(-1,1).t()))
        t2v   = torch.cat((v0.reshape(-1,1), v1), axis=1).float()

        return x + self.dropout(t2v)