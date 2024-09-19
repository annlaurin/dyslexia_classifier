import random
from typing import Callable, Collection, Dict, Iterator, List, Tuple, Type

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from math import ceil

def apply_standardization(x, m, sd):
    nonzero_sd = sd.clone()
    nonzero_sd[sd == 0] = 1
    x = torch.from_numpy(x).float()
    res = (x - m.unsqueeze(0)) / nonzero_sd.unsqueeze(0)
    return res
    
    
class EyetrackingDataPreprocessor(Dataset):
    """Dataset with the long-format sequence of fixations made during reading by dyslexic 
    and normally-developing Russian-speaking monolingual children."""

    def __init__(
        self, 
        csv_file, 
        transform=None, 
        target_transform=None, 
        dropPhonologyFeatures = True, 
        dropPhonologySubjects = True,     
        num_folds: float = 10,
        ablation = False
        ):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_transform (callable, optional): Optional transform to be applied
                on a label.
        """
        data = pd.read_csv(csv_file)
        
        # changing dyslexia labels to 0 and 1
        data['group'] = data['group'] + 0.5
        
        # drop reading speed, the main basis of classification
        data = data.drop(columns = ['Reading_speed'])
        
        if ablation == True:
            data = data.drop(columns = ['next_fix_dist', 'sac_ampl', 'sac_angle', 'sac_vel', 'direction'])
            convert_columns = ['Grade']
        else:
            convert_columns = ['Grade', 'direction']
        
        if dropPhonologyFeatures == True:
            data = data.drop(columns = ['IQ', 'Sound_detection', 'Sound_change'])
        
        for column in convert_columns:
            prefix = column + '_dummy'
            data = pd.concat([data, pd.get_dummies(data[column], 
                                    prefix=prefix)], axis=1)
            data = data.drop(columns = column)
            
        if dropPhonologySubjects == True:
            # Drop subjects
            data.dropna(axis = 0, how = 'any', inplace = True)
        else:
            # Drop columns
            data.dropna(axis = 1, how = 'any', inplace = True)
        
        # Record features that are used for prediction
        self._features = [i for i in data.columns if i not in ['group', 'item', 'subj']]

        self._data = pd.DataFrame()
        # Add sentence IDs and subject IDs
        self._data["sn"] = data["item"]
        self._data["subj"] = data["subj"]
        # Add labels
        self._data["group"] = data["group"]
        
        # Add features used for prediction
        for feature in self._features:
            self._data[feature] = data[feature]

        # Distribute subjects across stratified folds
        self._num_folds = num_folds
        self._folds = [[] for _ in range(num_folds)]
        dyslexic_subjects = self._data[self._data["group"] == 1]["subj"].unique()
        control_subjects = self._data[self._data["group"] == 0]["subj"].unique()
        random.shuffle(dyslexic_subjects)
        random.shuffle(control_subjects)
        for i, subj in enumerate(dyslexic_subjects):
            self._folds[i % num_folds].append(subj)
        for i, subj in enumerate(control_subjects):
            self._folds[num_folds - 1 - i % num_folds].append(subj)
        for fold in self._folds:
            random.shuffle(fold)

    def _iter_trials(self, folds: Collection[int]) -> Iterator[pd.DataFrame]:
        # Iterate over all folds
        for fold in folds:
            # Iterate over all subjects in the fold
            for subj in self._folds[fold]:       
                subj_data = self._data[self._data["subj"] == subj]
                # Iterate over all sentences this subject read
                for sn in subj_data["sn"].unique():
                    trial_data = subj_data[subj_data["sn"] == sn]
                    yield trial_data
                    
                    
    def iter_folds(
        self, folds: Collection[int]) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        for trial_data in self._iter_trials(folds):
            predictors = trial_data[self._features].to_numpy()
            label = trial_data["group"].unique().item()
            subj = trial_data["subj"].unique().item()
            #  X = (time_steps, features)
            X = predictors
            y = torch.tensor(label, dtype=torch.float)
            yield X, y, subj
                    

    @property
    def num_features(self) -> int:
        """Number of features per word (excluding word vector dimensions)."""
        return len(self._features)
    

    @property
    def max_number_of_sentences(self):
        data_copy = self._data.copy()
        max_s_count = data_copy.groupby(by="subj").sn.unique()
        return max([len(x) for x in max_s_count])

        
class EyetrackingDataset(Dataset):
    def __init__(
        self,
        preprocessor: EyetrackingDataPreprocessor,
        folds: Collection[int],
        batch_subjects: bool = False,
    ):
        self.sentences = list(preprocessor.iter_folds(folds))
        self._subjects = list(np.unique([subj for _, _, subj in self.sentences]))
        self.num_features = preprocessor.num_features
        self.batch_subjects = batch_subjects
        self.max_number_of_sentences = preprocessor.max_number_of_sentences

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.batch_subjects:
            subject = self._subjects[index]
            subject_sentences = [
                (X, y, subj) for X, y, subj in self.sentences if subj == subject
            ]
            X = torch.stack([torch.FloatTensor(X) for X, _, _ in subject_sentences])
            y = torch.stack([y for _, y, _ in subject_sentences]).unique().squeeze() 
            return X, y, subject

        else:
            X, y, subj = self.sentences[index]
            return X, y, subj

    def __len__(self) -> int:
        if self.batch_subjects:
            return len(self._subjects)
        else:
            return len(self.sentences)

    def standardize(self, mean: torch.Tensor, sd: torch.Tensor):
        self.sentences = [
            (apply_standardization(X, mean, sd), y, subj)
            for X, y, subj in self.sentences
        ]
