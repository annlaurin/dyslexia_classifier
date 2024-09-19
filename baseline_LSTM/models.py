from typing import TextIO, Tuple, Type, TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics

import numpy as np

from data import EyetrackingDataset

import matplotlib.pyplot as plt


T = TypeVar("T", bound="EyetrackingClassifier")

loss_fn = nn.BCEWithLogitsLoss()

class EyetrackingClassifier(nn.Module):
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
            y_pred_flat = self(X_flat, pretrain)
            # y_pred = (batch_size, sentences)
            y_pred = y_pred_flat.view(X.size(0), X.size(1))
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
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])
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
            for X, y, _ in loader:
                X = X.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_logits = model._predict(X, subj_mean=data.batch_subjects).squeeze()
                y_pred = torch.round(torch.sigmoid(y_logits))
                loss = loss_fn(y_logits, y.squeeze())
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
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
        y_preds_class = []
        y_preds = []
        y_trues = []
        subjs = []
        for X, y, subj in loader:
            X = X.to(device)
            y = y.to(device)
            y_logits = self._predict(X, subj_mean=data.batch_subjects).squeeze() 
            y_pred = torch.round(torch.sigmoid(y_logits)).squeeze()
            y_preds.append(torch.sigmoid(y_logits).item())  
            
            y_preds_class.append(round(y_pred.item()))
            y_trues.append(int(y.item()))
            subjs.append(subj)
        if per_subj:
            subjs, y_preds, y_preds_class, y_trues = aggregate_per_subject(
                subjs, y_preds, y_preds_class, y_trues
            )
        return y_preds, y_trues, subjs

    def evaluate(
        self,
        data: EyetrackingDataset,
        metric: str = "accuracy",
        print_report: bool = False,
        save_errors: TextIO = None,
        per_subj: bool = False,
        device: str = "cuda",
    ) -> Tuple[float, float, float, float]:
        self.to(device)
        self.eval()
        loader = torch.utils.data.DataLoader(data)
        y_preds_class = []
        y_preds = []
        y_trues = []
        subjs = []
        loss = 0
        for X, y, subj in loader:
            X = X.to(device)
            y = y.to(device)
            y_logits = self._predict(X, subj_mean=data.batch_subjects).squeeze()
            y_pred = torch.round(torch.sigmoid(y_logits)) 
            loss += loss_fn(y_logits, y.squeeze()).item() 
            
            y_preds.append(torch.sigmoid(y_logits).item())  
            y_preds_class.append(1 if y_pred.item() >= self.config['decision_boundary'] else 0)
            y_trues.append(int(y.item())) 
            subjs.append(subj) 
        if per_subj:
            subjs, y_preds, y_preds_class, y_trues = aggregate_per_subject(
                subjs, y_preds, y_preds_class, y_trues
            )
        if print_report:
            print(
                metrics.classification_report(y_trues, y_preds_class, zero_division=0)
            )
        if save_errors is not None:
            for subj, y_pred, y_pred_class, y_true in zip(subjs, y_preds, y_preds_class, y_trues):
                if y_pred_class != y_true:
                    save_errors.write(f"{subj},{y_pred},{y_true}\n")
        if metric == "accuracy":
            return metrics.accuracy_score(y_trues, y_preds_class)
        elif metric == "loss":
            return loss
        elif metric == "f1":
            return metrics.f1_score(y_trues, y_preds_class, zero_division= np.nan) 
        elif metric == "auc": 
            return metrics.roc_auc_score(y_trues, y_preds_class) 
        elif metric == "all":
            return (
                metrics.accuracy_score(y_trues, y_preds_class),
                metrics.precision_score(y_trues, y_preds_class, zero_division=np.nan), 
                metrics.recall_score(y_trues, y_preds_class, zero_division=np.nan),  
                metrics.f1_score(y_trues, y_preds_class, zero_division=np.nan),  
            )
        else:
            raise ValueError(f"Unknown metric '{metric}'")
            
   
class LSTMClassifier(EyetrackingClassifier):
    def initialize_model(self, input_size: int, config):
        self.lstm = nn.LSTM(input_size, config["lstm_hidden_size"], batch_first=True, bidirectional=True)  
        self.linear1 = nn.Linear(config["lstm_hidden_size"], 20)  
        self.linear2 = nn.Linear(20, 10)
        self.linear3 = nn.Linear(10, 1)

    def forward(self, input: torch.Tensor, pretrain: bool = False) -> torch.Tensor:
        lstm_output, (lstm_hidden, lstm_cell) = self.lstm(input)
        lstm_hidden = lstm_hidden.mean(0)
        linear1_output = self.linear1(lstm_hidden)
        if pretrain:
            linear_output = linear1_output.squeeze(1)
        else:
            linear_output = self.linear3(self.linear2(linear1_output)) 
        return linear_output  

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

        