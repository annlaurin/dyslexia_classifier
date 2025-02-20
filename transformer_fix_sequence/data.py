import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import TextIO, Callable, Collection, Dict, Iterator, List, Tuple, Type, TypeVar
            
def apply_standardization(x, m, sd):
    nonzero_sd = sd.clone()
    nonzero_sd[sd == 0] = 1
    x = torch.from_numpy(x).float()
    x_zeros = x[x.sum(dim=(1)) == 0]
    x_zeros[x_zeros==0] = -5
    x_non_zeros = x[x.sum(dim=(1)) != 0]
    x_non_zeros = (x_non_zeros - m.unsqueeze(0)) / nonzero_sd.unsqueeze(0)
    res = torch.cat((x_non_zeros, x_zeros), axis =0)
    return res


class EyetrackingDataPreprocessor(Dataset):
    """Dataset with the long-format sequence of fixations made during reading by dyslexic 
    and normally-developing Russian-speaking monolingual children."""

    def __init__(
        self, 
        csv_file, 
        transform=None, 
        target_transform=None,  
        num_folds: float = 10,
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
        if {'group'}.issubset(data.columns):   # not the case for pretrain dataset
            data['group'] = data['group'] + 0.5
        
        # log-transforming frequency
        to_transform = ['frequency', 'predictability', 'fix_dur'] #
        for column in to_transform:
            data[column] = data[column].apply(lambda x: np.log(x) if x > 0 else 0) 
        
        # drop columns we don't use
        data = data.drop(columns = ['fix_x', 'fix_y', 'fix_index'])  
        
        # center reading sopeed in case we need to predict it
        if {'Reading_speed'}.issubset(data.columns):
            data['Reading_speed'] = (data['Reading_speed'] - data['Reading_speed'].mean())/data['Reading_speed'].std(ddof=0)
        
        if {'sex', 'Grade'}.issubset(data.columns):
            data = data.drop(columns = ['sex', 'Grade'])
            
        convert_columns = ['direction']
        
        if {'IQ', 'Sound_detection', 'Sound_change'}.issubset(data.columns):
            data = data.drop(columns = ['IQ', 'Sound_detection', 'Sound_change'])
        
        for column in convert_columns:
            prefix = column + '_dummy'
            data = pd.concat([data, pd.get_dummies(data[column], 
                                    prefix=prefix)], axis=1)
            data = data.drop(columns = column)

        data.dropna(axis = 0, how = 'any', inplace = True)

            
        # rearrange columns (I need demogrpahic information to come last)
#         cols = ['item', 'subj', 'group', 'Reading_speed', 'fix_dur', 'landing', 'word_length',
#                  'predictability', 'frequency', 'number.morphemes', 'next_fix_dist',
#                  'sac_ampl', 'sac_angle', 'sac_vel', 'rel.position', 'direction_dummy_DOWN',
#                  'direction_dummy_LEFT', 'direction_dummy_RIGHT', 'direction_dummy_UP',
#                  'sex', 'Age', 'Grade_dummy_1', 'Grade_dummy_2', 'Grade_dummy_3', 'Grade_dummy_4',
#                  'Grade_dummy_5', 'Grade_dummy_6']
        if {'Reading_speed'}.issubset(data.columns):
            cols = ['item', 'subj', 'group', 'Reading_speed', 'fix_dur',
                   'landing', 'word_length', 'predictability', 'frequency', 
                    'number.morphemes', 'next_fix_dist', 'sac_ampl', 'sac_angle', 
                    'sac_vel', 'rel.position', 'direction_dummy_LEFT', 
                    'direction_dummy_RIGHT', 'direction_dummy_DOWN'] # temporary
        else:
            cols = ['item', 'subj', 'fix_dur',
                   'landing', 'word_length', 'predictability', 'frequency', 
                    'number.morphemes', 'next_fix_dist', 'sac_ampl', 'sac_angle', 
                    'sac_vel', 'rel.position', 'direction_dummy_LEFT', 
                    'direction_dummy_RIGHT', 'direction_dummy_DOWN'] # temporary
        data = data[cols]
        
        # Record features that are used for prediction
        if {'Reading_speed'}.issubset(data.columns):
            self._features = [i for i in data.columns if i not in ['group', 'item', 'subj', 'Reading_speed']]
        else:
            self._features = [i for i in data.columns if i not in ['item', 'subj']]
        self._data = pd.DataFrame()
        # Add sentence IDs and subject IDs
        self._data["sn"] = data["item"]
        self._data["subj"] = data["subj"]
        # Add labels
        if {'Reading_speed'}.issubset(data.columns):
            self._data["group"] = data["group"]
            self._data["reading_speed"] = data["Reading_speed"]
        else:
            self._data["group"] = -1
            self._data["reading_speed"] = -1
        
        # Add features used for prediction
        for feature in self._features:
            self._data[feature] = data[feature]

#       # Distribute subjects across stratified folds
        self._num_folds = num_folds
        self._folds = [[] for _ in range(num_folds)]
        just_subjects = self._data["subj"].unique()
        random.shuffle(just_subjects)
        for i, subj in enumerate(just_subjects):
            self._folds[i % num_folds].append(subj)
#         dyslexic_subjects = self._data[self._data["group"] == 1]["subj"].unique()
#         control_subjects = self._data[self._data["group"] == 0]["subj"].unique()
#         random.shuffle(dyslexic_subjects)
#         random.shuffle(control_subjects)
#         for i, subj in enumerate(dyslexic_subjects):
#             self._folds[i % num_folds].append(subj)
#         for i, subj in enumerate(control_subjects):
#             self._folds[num_folds - 1 - i % num_folds].append(subj)
        for fold in self._folds:
            random.shuffle(fold)

    def _iter_trials(self, folds: Collection[int]) -> Iterator[pd.DataFrame]:
        # Iterate over all folds
        for fold in folds:
            # Iterate over all subjects in the fold
            for subj in self._folds[fold]:       # Anna: subj in fold?
                subj_data = self._data[self._data["subj"] == subj]
                # Iterate over all sentences this subject read
                for sn in subj_data["sn"].unique():
                    trial_data = subj_data[subj_data["sn"] == sn]
                    yield trial_data
                    
                    
    def iter_folds(
        self, folds: Collection[int]) -> Iterator[Tuple[torch.Tensor, torch.Tensor, int]]:
        for trial_data in self._iter_trials(folds):
            predictors = trial_data[self._features].to_numpy()
            #predictors = np.reshape(predictors, (int(len(predictors)/278), 278, predictors.shape[1]))
            label = trial_data["group"].unique().item()
            subj = trial_data["subj"].unique().item()
            reading_speed = trial_data["reading_speed"].unique().item()
            #  X = (time_steps, features)
            X = predictors
            y = torch.tensor(label, dtype=torch.float)
            rs = torch.tensor(reading_speed , dtype=torch.float)
            yield X, y, subj, rs
                    

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
       # word_vector_model: WordVectorModel,
        folds: Collection[int],
        batch_subjects: bool = False,
    ):
        self.sentences = list(preprocessor.iter_folds(folds))
        self._subjects = list(np.unique([subj for _, _, subj, _ in self.sentences]))
        self.num_features = preprocessor.num_features# + word_vector_model.dimensions()
        self.batch_subjects = batch_subjects
        #self.max_sentence_length = preprocessor.max_sentence_length
        self.max_number_of_sentences = preprocessor.max_number_of_sentences

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        if self.batch_subjects:
            subject = self._subjects[index]
            subject_sentences = [
                (X, y, subj, rs) for X, y, subj, rs in self.sentences if subj == subject
            ]
            X = torch.stack([torch.FloatTensor(X) for X, _, _, _ in subject_sentences]) #[X for X, _, _ in subject_sentences] #torch.FloatTensor([X for X, _, _ in subject_sentences])
            y = torch.stack([y for _, y, _, _ in subject_sentences]).unique().squeeze() 
            rs = torch.stack([rs for _, _, _, rs in subject_sentences]).unique().squeeze()
            return X, y, subject, rs

        else:
            X, y, subj, rs = self.sentences[index] 
            return X, y, subj, rs

    def __len__(self) -> int:
        if self.batch_subjects:
            return len(self._subjects)
        else:
            return len(self.sentences)

    def standardize(self, mean: torch.Tensor, sd: torch.Tensor):
        self.sentences = [
            (apply_standardization(X, mean, sd), y, subj, rs)
            for X, y, subj, rs in self.sentences
        ]