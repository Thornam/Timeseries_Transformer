
import numpy as np
from torch import nn, Tensor
from typing import Optional, Any, Union, Callable, Tuple
import torch
import pandas as pd
from pathlib import Path
import random
import math


def generate_square_subsequent_mask(dim1: int, dim2: int) -> Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def dataset(
    data: pd.DataFrame,
    enc_seq_len: int,
    trg_seq_len: int,
    batch_size: int,
    device):

  # Resets index in data
  data = data.reset_index(drop=True)

  # Creating list
  index_list = list(range( len(data) ))

  # Shuffle list
  random.shuffle(index_list)

  ## Creates the batches, by looping over the index_list with the batch_size

  # Number of loops
  max_loop = math.floor(len(data)/ batch_size)

  # Creates a manual dropout
  max_loop -= 0 # Leaves a portion of the train dataset out of the batch creation

  # Creates a collection list for each batch
  Batches = []

  # Start of batch 1
  batch_start = 0

  # Perform the loop
  for b in range(max_loop):

    # End of the current batch
    batch_end = (1 + b) * batch_size

    # Subsets the train_df based on the indeces from the shuffle index_list with the length of the batch size
    batch_data = data[ data.index.isin(index_list[batch_start : batch_end]) ].iloc[:, 6:]

    # Saves the mean and std to be able to revert the standardization
    mn = torch.tensor(data[ data.index.isin(index_list[batch_start : batch_end]) ]['Mean'].values)
    std = torch.tensor(data[ data.index.isin(index_list[batch_start : batch_end]) ]['Std'].values)

    # Creating srg (Encoder input)
    srg = torch.tensor(batch_data.iloc[:, :enc_seq_len].values).float()
    srg = srg.unsqueeze(2)

    # Creating trg
    if trg_seq_len == 1:
       trg = torch.tensor(batch_data.iloc[:, enc_seq_len - 1].values).float()
       trg = trg.unsqueeze(1)
    else:
      trg = torch.tensor(batch_data.iloc[:, enc_seq_len - 1 : (enc_seq_len + trg_seq_len)-1].values).float()
    trg = trg.unsqueeze(2)

    # Creating trg_y, which it the sequence the model will compute the loss function on
    if trg_seq_len == 1:
       trg_y = torch.tensor(data[ data.index.isin(index_list[batch_start : batch_end]) ]['Target'].values).float()
       trg_y = trg_y.unsqueeze(1)
    else:
      trg_y = torch.tensor(batch_data.iloc[:, - trg_seq_len :].values).float()

    # Transfer the tensors to GPU
    srg = srg.to(device)
    trg = trg.to(device)
    trg_y = trg_y.to(device)
    mn = mn.to(device)
    std = std.to(device)

    # Create a list of the current batches
    batch = list([srg, trg, trg_y, mn, std])

    # Append the collection list
    Batches.append(batch)

    # We want to start next batch at this point in the index_list
    batch_start = batch_end

  #print(f'Dropout procentage: {((len(data) - batch_size * len(Batches)) / len(data) * 100):.1f} %')

  return Batches

def change_dim(
        src,
        trg,
        trg_y,
        batch_first):

    # Change dimensions from [batch size, seq len, num features] to [seq len, batch size, num features]
    if batch_first == False:

        src = src.permute(1, 0, 2)

        trg = trg.permute(1, 0, 2)

        trg_y = trg_y.permute(1, 0)

    return src, trg, trg_y

def run_encoder_decoder_inference(
    model: nn.Module,
    src: torch.Tensor,
    forecast_window: int,
    batch_size: int,
    device,
    batch_first: bool=False
    ) -> torch.Tensor:

    """

    This function is for encoder-decoder type models in which the decoder requires
    an input, tgt, which - during training - is the target sequence. During inference,
    the values of tgt are unknown, and the values therefore have to be generated
    iteratively.

    This function returns a prediction of length forecast_window for each batch in src

    NB! If you want the inference to be done without gradient calculation,
    make sure to call this function inside the context manager torch.no_grad like:
    with torch.no_grad:
        run_encoder_decoder_inference()

    The context manager is intentionally not called inside this function to make
    it usable in cases where the function is used to compute loss that must be
    backpropagated during training and gradient calculation hence is required.

    If use_predicted_tgt = True:
    To begin with, tgt is equal to the last value of src. Then, the last element
    in the model's prediction is iteratively concatenated with tgt, such that
    at each step in the for-loop, tgt's size increases by 1. Finally, tgt will
    have the correct length (target sequence length) and the final prediction
    will be produced and returned.

    Args:
        model: An encoder-decoder type model where the decoder requires
               target values as input. Should be set to evaluation mode before
               passed to this function.

        src: The input to the model

        forecast_horizon: The desired length of the model's output, e.g. 58 if you
                         want to predict the next 58 hours of FCR prices.

        batch_size: batch size

        batch_first: If true, the shape of the model input should be
                     [batch size, input sequence length, number of features].
                     If false, [input sequence length, batch size, number of features]

    """

    # Dimension of a batched model input that contains the target sequence values
    target_seq_dim = 0 if batch_first == False else 1

    # Take the last value of the target variable in all batches in src and make it tgt
    # as per the Influenza paper
    tgt = src[-1, :, 0] if batch_first == False else src[:, -1, 0] # shape [1, batch_size, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_size == 1 and batch_first == False:
        tgt = tgt.unsqueeze(0).unsqueeze(0) # change from [1] to [1, 1, 1]

    # Change shape from [batch_size] to [1, batch_size, 1]
    if batch_first == False and batch_size > 1:
        tgt = tgt.unsqueeze(0).unsqueeze(-1)

    # Iteratively concatenate tgt with the first element in the prediction
    for _ in range(forecast_window-1):

        # Create masks
        dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

        dim_b = src.shape[1] if batch_first == True else src.shape[0]

        tgt_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_a,
            ).cuda()

        src_mask = generate_square_subsequent_mask(
            dim1=dim_a,
            dim2=dim_b,
            ).cuda()

        # Make prediction
        prediction = model(src, tgt, src_mask, tgt_mask)

        # If statement simply makes sure that the predicted value is
        # extracted and reshaped correctly
        if batch_first == False:

            # Obtain the predicted value at t+1 where t is the last time step
            # represented in tgt
            last_predicted_value = prediction[-1, :, :]

            # Reshape from [batch_size, 1] --> [1, batch_size, 1]
            last_predicted_value = last_predicted_value.unsqueeze(0)

        else:

            # Obtain predicted value
            last_predicted_value = prediction[:, -1, :]

            # Reshape from [batch_size, 1] --> [batch_size, 1, 1]
            last_predicted_value = last_predicted_value.unsqueeze(-1)

        # Detach the predicted element from the graph and concatenate with
        # tgt in dimension 1 or 0
        tgt = torch.cat((tgt, last_predicted_value.detach()), target_seq_dim)

    # Create masks
    dim_a = tgt.shape[1] if batch_first == True else tgt.shape[0]

    dim_b = src.shape[1] if batch_first == True else src.shape[0]

    tgt_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_a,
        ).cuda()

    src_mask = generate_square_subsequent_mask(
        dim1=dim_a,
        dim2=dim_b,
        ).cuda()

    # Transfer the tensors to GPU
    src = src.to(device)
    tgt = tgt.to(device)

    # Make final prediction
    final_prediction = model(src, tgt, src_mask, tgt_mask)

    return final_prediction

class AdjMSELoss2(nn.Module):
    def __init__(self):
        super(AdjMSELoss2, self).__init__()

    def loss(self, True_vector, Pred_vector):

        beta = 2.5
        loss = (Pred_vector - True_vector)**2
        adj_loss = beta - (beta - 0.5) / (1 + torch.exp(2 * torch.mul(Pred_vector, True_vector)))
        loss = beta * loss /(1+adj_loss)
        return torch.mean(loss)

class WMSE(nn.Module):
  def __init__(self):
    super(WMSE, self).__init__()

  def loss(self, True_vector, Pred_vector):

    SE = torch.square( torch.sub(True_vector, Pred_vector) )
    WSE = torch.mul( torch.add( torch.abs(True_vector), 1) , SE)

    return torch.mean(WSE)

class MSECorr(nn.Module):
  def __init__(self):
    super(MSECorr, self).__init__()
    self.MSE = nn.MSELoss().cuda()

  def loss(self, True_vector, Pred_vector):

    mse = self.MSE(Pred_vector, True_vector)

    NegCorr = - torch.corrcoef(torch.vstack((True_vector, Pred_vector)))[0][1]

    return mse + NegCorr
  
class MSE_c(nn.Module):
  def __init__(self):
    super(MSE_c, self).__init__()
    self.MSE = nn.MSELoss().cuda()

  def loss(self, True_vector, Pred_vector):

    mse = self.MSE(Pred_vector, True_vector)

    return mse

class NegCorr(nn.Module):
  def __init__(self):
    super(NegCorr, self).__init__()

  def loss(self, True_vector, Pred_vector):

    corr = torch.corrcoef(torch.vstack((True_vector, Pred_vector)))[0][1]

    return - corr
  
class EIntensiveLoss(nn.Module):
  def __init__(self, epsilon):
    super(EIntensiveLoss, self).__init__()
    self.epsilon = float(epsilon)

  def loss(self, True_vector, Pred_vector):
    epsilon_tube = torch.sub( torch.abs(True_vector), self.epsilon)
    error2 = torch.square( torch.sub(True_vector, Pred_vector) )
    error2[ epsilon_tube < 0 ] = 0

    return torch.mean(error2)
