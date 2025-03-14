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
BATCH_SUBJECTS = True

loss_fn = nn.MSELoss()

def aggregate_speed_per_subject(subjs, y_preds, y_trues):
    y_preds = np.array(y_preds)
    #y_preds_class = np.array(y_preds_class)
    y_trues = np.array(y_trues)
    subjs = np.array(subjs).flatten()
    y_preds_subj = []
    y_trues_subj = []
    subjs_subj = np.unique(subjs)
    for subj in subjs_subj:
        subj = subj.item()
        y_pred_subj = y_preds[subjs == subj]
        y_true_subj = y_trues[subjs == subj]
        assert len(np.unique(y_true_subj)) == 1, f"No unique label: subj={subj}"
        y_trues_subj.append(np.unique(y_true_subj).item())
        y_preds_subj.append(np.mean(y_pred_subj).item())

    return subjs_subj, y_preds_subj, y_trues_subj
    
    
class RSClassifier(nn.Module):
    def __init__(self, input_size: int, config):
        super().__init__()
        self.initialize_model(input_size, config)
        self.config = config

    def initialize_model(self, input_size: int, config):
        raise NotImplementedError()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def _predict(self, X: torch.Tensor, subj_mean: bool = False, pretrain: bool = False) -> torch.Tensor:
        if pretrain:
            y_preds = self(X, pretrain=True)
            # return predictions for all 12 features
            return y_preds
        if subj_mean:
            # X = (batch_size, sentences, time_steps, features)
            X_flat = X.view(X.size(0) * X.size(1), X.size(2), X.size(3))
            # X_flat = (batch_size*sentences, time_steps, features)
            y_pred_flat = self(X_flat, pretrain)
            # y_pred_flat = (batch_size*sentences)
            y_pred = y_pred_flat.view(X.size(0), X.size(1))
            # y_pred = (batch_size, sentences)
            final_predictions = []
            for batch_X, batch_y_pred in zip(X, y_pred):
                batch_predictions = []
                for sentence_X, sentence_y_pred in zip(batch_X, batch_y_pred):
                    if (
                        sentence_X.count_nonzero() > 0
                    ):  # Ignore the padding sentences
                        batch_predictions.append(sentence_y_pred)
                final_predictions.append(torch.mean(torch.stack(batch_predictions)))
            final_y_pred = torch.stack(final_predictions)
            return final_y_pred
        else:
            y_pred = self(X, pretrain)
            return y_pred
 

    @classmethod
    def train_model(
        cls: Type[T],
        data: EyetrackingDataset,
        min_epochs: int = 15,
        max_epochs: int = 200,
        dev_data: EyetrackingDataset = None,
        device: str = "cuda",
        config=None,
        patience=10,
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
            for X, _, _, y in loader:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model._predict(X, subj_mean=BATCH_SUBJECTS).squeeze()
                loss = loss_fn(y_pred, y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()/len(X)  
            mean_loss = epoch_loss/len(loader)
            if dev_data is not None:
                dev_accuracy = model.evaluate(dev_data, metric="loss", device=device)
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
        per_subj: bool = True,
    ):
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data)
        y_preds = []
        y_trues = []
        subjs = []
        for X, _, subj, y in loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = self._predict(X, subj_mean=data.batch_subjects).squeeze()               
            y_preds.append(y_pred.item())
            y_trues.append(y.item())
            subjs.append(subj)
        if per_subj:
            subjs, y_preds, y_trues = aggregate_speed_per_subject(
                subjs, y_preds, y_trues
            )
        return y_preds, y_trues, subjs

    def evaluate(
        self,
        data: EyetrackingDataset,
        metric: str = "loss",
        print_report: bool = False,
        save_errors: TextIO = None,
        per_subj: bool = False,
        device: str = "cuda",
    ) -> Tuple[float, float, float, float]:
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data)
        y_preds = []
        y_trues = []
        subjs = []
        loss = 0
        for X, _, subj, y in loader:
            X = X.to(device)
            y = y.to(device)
            y_pred = self._predict(X, subj_mean=BATCH_SUBJECTS).squeeze()
            batch_loss = loss_fn(y_pred, y.squeeze())
            loss += batch_loss.item()/len(X)     
            
            y_preds.append(y_pred.item())  
            y_trues.append(y.item()) 
            subjs.append(subj) 
        mean_loss = loss/len(loader)
        if per_subj:
            subjs, y_preds, y_trues = aggregate_speed_per_subject(
                subjs, y_preds, y_trues
            )
        if metric == "loss":
            return mean_loss
        else:
            raise ValueError(f"Unknown metric '{metric}'")
            
            
    
class LSTMClassifier_RS(RSClassifier):
    def initialize_model(self, input_size: int, config):
        self.lstm = nn.LSTM(input_size, config["lstm_hidden_size"], batch_first=True, bidirectional=True)  
        self.linear1 = nn.Linear(config["lstm_hidden_size"]*2, 10)  
        self.linear2 = nn.Linear(10, 5)
        self.linear3 = nn.Linear(5, 1)

    def forward(self, input: torch.Tensor, pretrain: bool = False) -> torch.Tensor:
        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(input)
        h_1, h_2 = lstm_hidden[0], lstm_hidden[1]
        final_hidden_state = torch.cat((h_1, h_2), 1)
        linear1_output = self.linear1(final_hidden_state)
        if pretrain:
            linear_output = linear1_output.squeeze(0)
        else:
            linear_output = self.linear3(self.linear2(linear1_output)).squeeze(0)  
        return linear_output
        
        