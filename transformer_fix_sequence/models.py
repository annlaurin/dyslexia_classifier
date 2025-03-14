import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import math
from data import EyetrackingDataset
from typing import TextIO, Callable, Collection, Dict, Iterator, List, Tuple, Type, TypeVar
from functools import partial, reduce
T = TypeVar("T", bound="EyetrackingClassifier")

NUM_FEATURES = 14
NUM_FIX = 30 

def mse_loss(target, input, mask):
    out = (input[mask]-target[mask])**2
    return out.mean()


def mask_with_tokens_3D(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    reduced = torch.any(mask, dim=-1, keepdim=True)
    expanded = reduced.expand_as(mask)
    return expanded


def get_mask_subset_with_prob_3D(mask, prob):
    batch, num_fix, num_features, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * num_fix)

    num_tokens = mask.sum(dim=-2, keepdim=True)
    mask_excess = (mask.cumsum(dim=-2)[:,:,0] > (num_tokens[:,:,0] * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, num_fix, num_features), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-2)
    sampled_indices = (sampled_indices[:,:,0] + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, num_fix + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    new_mask = new_mask[:, 1:].bool()
    
    return new_mask.unsqueeze_(2).expand(-1,-1, num_features)
    
    
class AnnasPositionalEncoding(nn.Module):
    def __init__(self, fixations = int, features = int, device = None):
        super(AnnasPositionalEncoding, self).__init__()
        
        # Initialize a learnable positional encoding matrix for fixations
        self.fix_encoding = nn.Parameter(torch.zeros(fixations, 1)).to(device)
        nn.init.xavier_uniform_(self.fix_encoding)  # Xavier initialization for better training stability
        self.fix_encoding = self.fix_encoding.expand(-1, features)
        
        # Initialize a learnable positional encoding matrix for features
        self.feat_encoding = nn.Parameter(torch.zeros(1, features)).to(device)
        nn.init.xavier_uniform_(self.feat_encoding)  # Xavier initialization for better training stability
        self.feat_encoding = self.feat_encoding.expand(fixations, -1)
        
        self.encoding = self.fix_encoding + self.feat_encoding
        
    def forward(self, x, mask = None):
        if mask is not None:
            # Apply the mask to ignore padded positions
            pos_encoding = self.encoding  * mask
        else:
            pos_encoding = self.encoding
        return x + pos_encoding


def getmeansd(dataset, batch: bool = False):
    if batch:
        # Anna added preprocessing from ndarray to torch
        tensors = [X for X, _, _, _ in dataset]  
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=(1,2)) != 0]   
        # remove rows of 0s from the computation
        sentences, timesteps, features = tensors.size()
        subset = tensors.sum(dim=(2)) != 0
        subset = subset.view(sentences, timesteps, 1)
        subset = subset.expand(sentences, timesteps, features)
        result = tensors[subset].view(-1, features) 
        
        means = torch.mean(result, dim=(0))
        sd = torch.std(result, dim=(0))
        return means, sd
    else:
        tensors = [torch.from_numpy(X).float() for X, _, _, _ in dataset] 
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=1) != 0]
        means = torch.mean(tensors, 0)
        sd = torch.std(tensors, 0)
        return means, sd
    
    
class EncoderLayer(nn.Module):  
    def __init__(self,
                dim_upscale = int,
                inner_dim_upscale = int,
                num_heads = int, 
                num_layers = int, 
                dropout = 0
                ):

        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout
        self.dim_upscale = dim_upscale
        self.inner_dim_upscale = inner_dim_upscale
        
        # layer norm for multi-head attention
        self.attn_layer_norm = nn.LayerNorm(self.dim_upscale)
        # layer norm for feedforward network
        self.ffn_layer_norm = nn.LayerNorm(self.dim_upscale)
        
        self.attention = nn.MultiheadAttention(embed_dim = self.dim_upscale,  
                                                 num_heads = self.num_heads, 
                                                 bias = True,
                                                 batch_first = True)
        # feed forward
        self.ff = nn.Sequential(
            nn.Linear(self.dim_upscale, self.inner_dim_upscale, bias = True),
            nn.LayerNorm(self.inner_dim_upscale),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.inner_dim_upscale, self.dim_upscale, bias = True)
        )
        

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):
        # pass embeddings through multi-head attention
        x, attn_probs = self.attention(src, src, src, src_mask)

        # residual add and norm
        first_out = self.attn_layer_norm(x + src)

        # position-wise feed-forward network
        x2 = self.ff(first_out)

        # residual add and norm
        second_out = self.ffn_layer_norm(x2 + first_out)

        return second_out, attn_probs


class Encoder(nn.Module):
    def __init__(self, 
                dim_upscale = int,
                inner_dim_upscale = int,
                num_heads = int, 
                num_layers = int, 
                dropout = 0):

        super().__init__()
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.dim_upscale = dim_upscale
        self.inner_dim_upscale = inner_dim_upscale
        
        # create n_layers encoders 
        self.layers = nn.ModuleList([EncoderLayer(
                                    dim_upscale = self.dim_upscale,
                                    num_heads = self.num_heads, 
                                    inner_dim_upscale = self.inner_dim_upscale,
                                    dropout = self.dropout)
                                     for layer in range(self.num_layers)])


    def forward(self, src: torch.Tensor, src_mask: torch.Tensor):

        # pass the sequences through each encoder
        for layer in self.layers:
            src, attn_probs = layer(src, src_mask)

        self.attn_probs = attn_probs

        return src
        
        
class EyetrackingClassifier(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.initialize_model(input_size, config)
        self.config = config

    def initialize_model(self, input_size: int, config):
        raise NotImplementedError()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _predict(self, X: torch.Tensor, identity: bool = False, subj_mean: bool = False, pretrain: bool = False) -> torch.Tensor:
        if pretrain:
            y_preds, labels, mask = self(X, identity, pretrain=True)
            return y_preds, labels, mask
        else:
            y_preds, labels, mask = self(X, identity, pretrain)
            return y_preds, labels, mask
        
    @classmethod
    def train_model(
        cls: Type[T],
        data: EyetrackingDataset,
        min_epochs: int = 15,
        max_epochs: int = 300,
        dev_data: EyetrackingDataset = None,
        device: str = "cuda",
        config = None,
        patience = 20,
        pretrained_model: T = None,
        **kwargs,
    ) -> Tuple[T, int]:
        model = pretrained_model or cls(data.num_features, config, **kwargs) 
        model.to(device)
        model.train()
        optimizer = Adam(model.parameters(), lr=config["lr"])
        epoch_count = 0
        best_losses = [float("inf")] * patience
        for epoch in range(max_epochs):
            # reshuffle data in each epoch
            loader = torch.utils.data.DataLoader(
                data,
                batch_size=config["batch_size"],
                shuffle=True,
            )
            epoch_count += 1
            epoch_loss = 0
            for X, _, _, _ in loader:
                X = X.to(device)
                train_preds, labels, mask = model._predict(X, identity = False) 
                loss = mse_loss(
                    labels,
                    train_preds,
                    mask
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            #avg_loss = epoch_loss/math.ceil(len(loader.dataset)/BATCH_SIZE)  # TODO: delete later
            #print(f"Epoch {epoch} done. Loss: {avg_loss}")
            
            if dev_data is not None:
                dev_loss = model.evaluate(dev_data, metric="loss", device=device, batch_size = config["batch_size"])
                model.train()
                # print(f"Dev loss: {dev_accuracy} in Epoch {epoch}")
                if epoch > min_epochs and all(dev_loss > i for i in best_losses):
                    epoch_count -= patience - best_losses.index(min(best_losses))
                    break
                else:
                    best_losses.pop(0)
                    best_losses.append(dev_loss)
        return model

    def evaluate(
        self,
        data: EyetrackingDataset,
        metric: str = "loss",
        device: str = "cuda",
        batch_size = None,
    ) -> Tuple[float, float, float, float]:
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data, batch_size = batch_size)
        loss = 0
        for X, _, _, _ in loader:
            X = X.to(device)
            dev_preds, labels, mask = self._predict(X, identity = False)
            dloss = mse_loss(
                labels,
                dev_preds,
                mask
            )
            loss += dloss.item() 
        avg_loss = loss /math.ceil(len(loader.dataset)/batch_size) 
        if metric == "loss":
            return avg_loss
        else:
            raise ValueError(f"Unknown metric '{metric}'")


class TransformerClassifier(EyetrackingClassifier):
    def initialize_model(
        self, 
        input_size: int, 
        config,
        embed_dim = NUM_FIX,
        d_model = NUM_FEATURES,
        dim_upscale = 128,
        inner_dim_upscale = 4*128,
        num_heads = 1, 
        num_layers = 1, 
        dropout = 0.15,
        mask_prob = 0.2,
        pad_token_id = -5,
        mask_ignore_token_ids = [],
        device: str = "cuda"
        ):
        
        self.embed_dim = embed_dim
        self.d_model = d_model
        self.mask_prob = config["mask_prob"]
        self.num_heads = config["num_heads"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]
        self.dim_upscale = config["upscale_dim"]
        self.inner_dim_upscale = config["inner_dim"]
        self.device = device

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, self.pad_token_id])
        
        self.positional_encoding = AnnasPositionalEncoding(fixations = self.embed_dim, 
                                                           features = self.d_model,
                                                           device = self.device)
        
        self.upscale = nn.Linear(self.d_model, self.dim_upscale, bias = True)
        self.downscale = nn.Linear(self.dim_upscale, self.d_model, bias = True)
        
        self.encoder = Encoder(dim_upscale = self.dim_upscale, 
                               num_heads = self.num_heads, 
                               num_layers = self.num_layers,
                               inner_dim_upscale = self.inner_dim_upscale, 
                               dropout = self.dropout)

    def forward(self, input: torch.Tensor, identity = False, pretrain: bool = False):
        
        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens_3D(input, self.mask_ignore_token_ids) 
        mask = get_mask_subset_with_prob_3D(~no_mask, self.mask_prob)
        hidden = no_mask + mask # all elements that the model will not attend to

        masked_seq = input.clone().detach().to(self.device)
        
        #  positional encoding
        masked_seq_pos = self.positional_encoding(masked_seq, mask = None) #for hidden fixations ~ no_mask

        # derive labels to predict
        labels = input.masked_fill(~mask, self.pad_token_id)
    
        if identity:
            attn_mask = no_mask[:,:,0]
        else:
            attn_mask = hidden[:,:,0]
        
        # Upscaling
        masked_seq_upscaled = self.upscale(masked_seq_pos)
        
        # Encoder
        out = self.encoder(masked_seq_upscaled, 
                           attn_mask)  
        out = self.downscale(out)

        return out, labels, mask


class LSTMClassifier(EyetrackingClassifier):
    def initialize_model(self, input_size: int, config):
        self.lstm = nn.LSTM(input_size-(NUM_DEMOGR_FEATURES), config["lstm_hidden_size"], 
                            batch_first=True, bidirectional=True) 
        self.linear1 = nn.Linear(config["lstm_hidden_size"] + (NUM_DEMOGR_FEATURES), 20)
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 1)
        self.dropout = nn.Dropout(0.20) 

    def forward(self, input: torch.Tensor, pretrain: bool = False) -> torch.Tensor:
        demo = input[:, 0, -(NUM_DEMOGR_FEATURES):] # only one time step is needed, demography is the same across time steps
        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(input[:,:,:-(NUM_DEMOGR_FEATURES)])
        lstm_hidden = lstm_hidden.mean(0)
        linear_input = torch.cat((lstm_hidden, demo), dim=1) # adding demographic features back
        linear1_output = self.linear1(linear_input)
        linear2_output = self.linear2(linear1_output)
        if pretrain:
            linear_output = linear1_output.squeeze(1)
        else:
            linear_output = self.linear3(linear2_output) 
        return linear_output
        
