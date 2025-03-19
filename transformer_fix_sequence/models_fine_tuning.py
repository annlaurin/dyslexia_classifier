import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import math
from sklearn import metrics
from data import EyetrackingDataset
from typing import TextIO, Callable, Collection, Dict, Iterator, List, Tuple, Type, TypeVar
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score, precision_score, recall_score, f1_score
from functools import partial, reduce
import torch.nn.functional as F
T = TypeVar("T", bound="EyetrackingClassifier")

NUM_FEATURES = 14
NUM_FIX = 30 
BATCH_SUBJECTS = True
NUM_DEMOGR_FEATURES = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

loss_fn = nn.BCEWithLogitsLoss()

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


def aggregate_per_subject(subjs, y_preds, y_preds_class, y_trues):
    y_preds = np.array(y_preds)
    y_preds_class = np.array(y_preds_class)
    y_trues = np.array(y_trues)
    subjs = np.array(subjs).flatten()
    y_preds_subj = []
    y_preds_class_subj = []
    y_trues_subj = []
    subjs_subj = np.unique(subjs)
    for subj in subjs_subj:
        subj = subj.item()
        y_pred_class_subj = y_preds_class[subjs == subj]
        y_pred_subj = y_preds[subjs == subj]
        y_true_subj = y_trues[subjs == subj]
        assert len(np.unique(y_true_subj)) == 1, f"No unique label: subj={subj}"
        y_trues_subj.append(np.unique(y_true_subj).item())
        y_preds_subj.append(np.mean(y_pred_subj).item())
        if sum(y_pred_class_subj) >= (len(y_pred_class_subj) / 2):
            y_preds_class_subj.append(1)
        else:
            y_preds_class_subj.append(0)
    return subjs_subj, y_preds_subj, y_preds_class_subj, y_trues_subj

    
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


class EyetrackingClassifierBinary(nn.Module):
    def __init__(self, input_size: int, config, pretrained_model):
        super().__init__()
        self.initialize_model(input_size, config, pretrained_model)
        self.config = config

    def initialize_model(self, input_size: int, config, pretrained_model):
        raise NotImplementedError()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
        
    def _predict_bin(self, X: torch.Tensor) -> torch.Tensor:
        y_preds = self(X)
        return y_preds
        
    
    @classmethod
    def train_model_bin(
        cls: Type[T],
        data: EyetrackingDataset,
        min_epochs: int = 15,
        max_epochs: int = 200,
        dev_data: EyetrackingDataset = None,
        device: str = "cuda",
        config = None,
        patience = 10,
        pretrained_model = None,
        **kwargs,
    ) -> Tuple[T, int]:
        model = cls(input_size = data.num_features, 
                    config = config, 
                    pretrained_model = pretrained_model, **kwargs) 
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
                # drop_last=True
            )
            epoch_count += 1
            epoch_loss = 0
            for X, y, _, _ in loader:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
               
                y_logits = model._predict_bin(X).squeeze() 
                y_pred = torch.round(torch.sigmoid(y_logits))
                loss = loss_fn(y_logits, y.squeeze()) 
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if dev_data is not None:
                dev_accuracy = model.evaluate_bin(dev_data, metric="loss", 
                                                  device=device, 
                                                  config = config)
                model.train()
                if epoch > min_epochs and all(dev_accuracy > i for i in best_losses):
                    epoch_count -= patience - best_losses.index(min(best_losses))
                    break
                else:
                    best_losses.pop(0)
                    best_losses.append(dev_accuracy)
        return model
    
    def predict_probs(
        self,
        data: EyetrackingDataset,
        device: str = "cuda",
        config = None,
        per_subj: bool = True,
    ):
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data, batch_size=config["batch_size"])
        y_preds_class = []
        y_preds = []
        y_trues = []
        subjs = []
        for X, y, subj, _ in loader:
            X = X.to(device)
            y = y.to(device)
            y_logits = self._predict_bin(X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits)) 
            if len(y) == 1:
                y_preds.append(torch.sigmoid(y_logits).item())
                y_preds_class.append([1 if y_pred >= 0.5 else 0][0])
                y_trues.append(y.item())
            else:
                y_preds.extend([i.item() for i in torch.sigmoid(y_logits)])
                y_preds_class.extend([1 if i else 0 for i in y_pred >= 0.5])
                y_trues.extend([i.item() for i in y])
            subjs.extend([i for i in subj])
        if per_subj:
            subjs, y_preds, y_preds_class, y_trues = aggregate_per_subject(
                subjs, y_preds, y_preds_class, y_trues
            )
        return y_preds, y_trues, subjs
            
    def evaluate_bin(
        self,
        data: EyetrackingDataset,
        print_report: bool = False,
        metric: str = "loss",
        device: str = "cuda",
        config = None,
        per_subj: bool = True,
    ) -> Tuple[float, float, float, float]:
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data, batch_size=config["batch_size"])
        y_preds_class = []
        y_preds = []
        y_trues = []
        subjs = []
        loss = 0
        for X, y, subj, _ in loader:
            X = X.to(device)
            y = y.to(device)
            y_logits = self._predict_bin(X).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits)) 
            loss += loss_fn(y_logits, y.squeeze()).item() 
            if len(y) == 1:
                y_preds.append(torch.sigmoid(y_logits).item())
                y_preds_class.append([1 if y_pred >= 0.5 else 0][0])
                y_trues.append(y.item())
            else:
                y_preds.extend([i.item() for i in torch.sigmoid(y_logits)])
                y_preds_class.extend([1 if i else 0 for i in y_pred >= 0.5])
                y_trues.extend([i.item() for i in y])
            subjs.extend([i for i in subj])
        avg_loss = loss /math.ceil(len(loader.dataset)/config['batch_size']) 
        if per_subj:
            subjs, y_preds, y_preds_class, y_trues = aggregate_per_subject(
                subjs, y_preds, y_preds_class, y_trues
            )
        if print_report:
            print(
                classification_report(y_trues, y_preds_class, zero_division=0)
            )
        if metric == "accuracy":
            return accuracy_score(y_trues, y_preds_class)
        elif metric == "loss":
            return avg_loss
        elif metric == "auc":
            return roc_auc_score(y_trues, y_preds)
        elif metric == "all":
            return (
                avg_loss,
                accuracy_score(y_trues, y_preds_class),
                precision_score(y_trues, y_preds_class, zero_division=np.nan), 
                recall_score(y_trues, y_preds_class, zero_division=np.nan),  
                f1_score(y_trues, y_preds_class, zero_division=np.nan),  
                roc_auc_score(y_trues, y_preds)
            )
        else:
            raise ValueError(f"Unknown metric '{metric}'")
            

            
class BinaryTransformerClassifier(EyetrackingClassifierBinary):
    def initialize_model(
        self, 
        input_size: int, 
        config,
        pretrained_model,
        dim_upscale = 256,
        pad_token_id = -5,
        mask_ignore_token_ids = []
        ):
        
        self.dim_upscale = dim_upscale

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, self.pad_token_id])
        
        # pre-trained layers
        self.positional_encoding = pretrained_model.positional_encoding
        self.upscale = pretrained_model.upscale
        self.encoder = pretrained_model.encoder
        
        # new trainable layers
        self.tuning = nn.Sequential(
            nn.Linear(self.dim_upscale + NUM_DEMOGR_FEATURES, config["hidden_size"], bias = True),
            nn.ReLU(inplace=True),
            nn.Linear(config["hidden_size"], 16, bias = True),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1, bias = True),
        )
        

    def forward(self, input: torch.Tensor):
        demo = input[:, 0, -(NUM_DEMOGR_FEATURES):]
        no_demo_input = input[:,:,:-(NUM_DEMOGR_FEATURES)]
        no_mask = mask_with_tokens_3D(no_demo_input, self.mask_ignore_token_ids) 
        masked_seq = no_demo_input.clone().detach().to(device)
        
        #  positional encoding
        masked_seq_pos = self.positional_encoding(masked_seq, mask = None) 
    
        attn_mask = no_mask[:,:,0]
        upscale_mask = ~attn_mask.unsqueeze(2).expand(-1,-1, self.dim_upscale)
        # Upscaling
        masked_seq_upscaled = self.upscale(masked_seq_pos)
        
        # Encoder
        out = self.encoder(masked_seq_upscaled, 
                           attn_mask) 
        
        # Trainable level
        mean_out = (out*upscale_mask).sum(dim=1)/upscale_mask.sum(dim=1)
        linear_input = torch.cat((mean_out, demo), dim=1) # adding demographic features back
#        mean_out = out[:, 0, :]# torch.mean(out, dim=-2) 
#        first_out = out[:, 0, :]
#        last_out = out[:, 29, :]
        pred = self.tuning(linear_input)

        return pred
        
