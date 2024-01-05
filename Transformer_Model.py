
import torch
import torch.nn as nn 
from torch import nn, Tensor
import positional_encoder as pe
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TimeSeriesTransformer(nn.Module):

    def __init__(self, 
        input_size: int,
        dec_seq_len: int,
        batch_first: bool,
        batch_size: int=128,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=2048,
        dim_feedforward_decoder: int=2048,
        num_predicted_features: int=1,
        PE = 'original',
        device=device
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        self.dec_seq_len = dec_seq_len

        # Creating the linear layers for the Encoder input, Decoder input, and the overall output
        self.encoder_input_layer = nn.Linear(
                                                in_features=input_size, 
                                                out_features=dim_val 
                                                )

        self.decoder_input_layer = nn.Linear(
                                                in_features=num_predicted_features,
                                                out_features=dim_val
                                                )  
        
        self.linear_mapping = nn.Linear(
                                            in_features=dim_val, 
                                            out_features=num_predicted_features
                                            )

        # Create positional encoder
        if PE == 'T2V':
            self.positional_encoding_layer = pe.T2V(
                                                        input_length=dim_val,
                                                        batch_size=batch_size,
                                                        device = device,
                                                        dropout=dropout_pos_enc
                                                        )
        elif PE == 'original':
            self.positional_encoding_layer = pe.PositionalEncoder(
                                                                    d_model=dim_val,
                                                                    dropout=dropout_pos_enc
                                                                    )

        # Creating the Encoder block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
            )

        # Stacking Encoder block
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
            )

        # Create the Decoder block
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
            )

        # Stacking Decoder blocks
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
            )

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None) -> Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """

        # Pass through Encoder input embedding
        src = self.encoder_input_layer(src) 

        # Pass through Positional Encoding
        src = self.positional_encoding_layer(src) 

       # Pass through the stack of Encoder blocks
        src = self.encoder(src=src)

        # Pass through Decoder input embedding
        decoder_output = self.decoder_input_layer(tgt) 

        # Pass through the stack of Decoder blocks
        decoder_output = self.decoder(
                                        tgt=decoder_output,
                                        memory=src,
                                        tgt_mask=tgt_mask,
                                        memory_mask=src_mask
                                        )

        # Pass through linear layer
        decoder_output = self.linear_mapping(decoder_output) 

        return decoder_output
