import random
import numpy as np
import argparse
import copy
import pickle
import os


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import Subset 
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
import torch.optim as optim

BATCH_SUBJECTS = True
dropPhonologyFeatures = True
ablation = False

## Import custom parts of the project
from models import EyetrackingClassifier, LSTMClassifier
from constants import hyperparameter_space, default_params, features
import data
from data import EyetrackingDataset, apply_standardization, EyetrackingDataPreprocessor
from roc import ROC

from typing import TextIO, Callable, Collection, Dict, Iterator, List, Tuple, Type, TypeVar



T = TypeVar("T", bound="EyetrackingClassifier")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def getmeansd(dataset, batch: bool = False):
    if batch:
        tensors = [X for X, _, _ in dataset]  
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=(1, 2)) != 0]
        means = torch.mean(tensors, dim=(0, 1))
        sd = torch.std(tensors, dim=(0, 1))
        return means, sd
    else:
        tensors = [torch.from_numpy(X).float() for X, _, _ in dataset] 
        tensors = torch.cat(tensors, axis=0)
        # remove padded tensors
        tensors = tensors[tensors.sum(dim=1) != 0]
        means = torch.mean(tensors, 0)
        sd = torch.std(tensors, 0)
        return means, sd
    
    
def get_params(paramdict) -> dict:
    selected_pars = dict()
    for k in paramdict:
        selected_pars[k] = random.sample(list(paramdict[k]), 1)[0]
    return selected_pars
    
    
# default: --tune none --model lstm --subjpred
parser = argparse.ArgumentParser(description="Run Russian Dyslexia Experiments")
parser.add_argument("--model", dest="model")
parser.add_argument("--roc", dest="roc", action="store_true")
parser.add_argument("--no-roc", dest="roc", action="store_false")
parser.add_argument("--tunesets", type=int, default=2)
parser.add_argument("--tune", dest="tune", action="store_true")
parser.add_argument("--no-tune", dest="tune", action="store_false")
parser.add_argument("--pretrain", dest="pretrain", action="store_true")
parser.add_argument("--subjpred", dest="batch_subjects", action="store_false")
parser.add_argument("--textpred", dest="batch_subjects", action="store_true")
parser.add_argument("--save-errors", dest="save_errors", type=argparse.FileType("w"))
parser.add_argument("--seed", dest="seed", type=int, default=42)
parser.add_argument("--cuda", dest="cudaid", default=0)
parser.set_defaults(tune=True) 
parser.set_defaults(roc=True)
parser.set_defaults(batch_subjects=True) 
parser.set_defaults(model = "lstm")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if args.model == "lstm":
    MODEL_CLASS = LSTMClassifier 
    
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device = torch.device(f'cuda:{args.cudaid}')
    
NUM_FOLDS = 3
NUM_TUNE_SETS = args.tunesets
BATCH_SUBJECTS = args.batch_subjects
tune = args.tune

# Prepare ROC Curves
if args.roc:
    Roc = ROC(args.model, args.tune)

if args.save_errors is not None:
    args.save_errors.write("subj,y_pred,y_true\n")

if tune:
    used_test_params = []
    parameter_sample = [
        get_params(hyperparameter_space[args.model]) for _ in range(NUM_TUNE_SETS)
    ]

tprs_folds = {}
   

# load and preprocess data for training
if BATCH_SUBJECTS:
    file = 'data/fixation_dataset_long_no_word_padded.csv'
    setting = "reader"
    folder = "reader_prediction_results/"
else:
    file = 'data/fixation_dataset_long_no_word_new.csv'
    setting = "sentence"
    folder = "sentence_prediction_results/"
    
dirExist = os.path.exists(folder)
if not dirExist:
    os.makedirs(folder)

def main():
    try:
        preprocessor = EyetrackingDataPreprocessor(
            csv_file = file,
            num_folds = NUM_FOLDS
        )
        
        test_accuracies = []
        for test_fold in range(NUM_FOLDS):
            print("test fold ", test_fold)
            parameter_evaluations = np.zeros(shape=(NUM_FOLDS, NUM_TUNE_SETS))
            if tune:
                # Normal training / fine-tuning
                for dev_fold in range(NUM_FOLDS):
                    if args.pretrain:
                        pretrained_models = next(pretrained_model_generator)
                    if dev_fold == test_fold:
                        continue
                    print(f'dev fold {dev_fold}')
                    train_folds = [
                        fold
                        for fold in range(NUM_FOLDS)
                        if fold != test_fold and fold != dev_fold
                    ]
                    train_dataset = EyetrackingDataset(
                        preprocessor,
                        folds=train_folds,
                        batch_subjects=BATCH_SUBJECTS,
                    )
                    mean, sd = getmeansd(train_dataset, batch=BATCH_SUBJECTS)
                    train_dataset.standardize(mean, sd)
                    dev_dataset = EyetrackingDataset(
                        preprocessor,
                        folds=[dev_fold],
                        batch_subjects=BATCH_SUBJECTS,
                    )
                    dev_dataset.standardize(mean, sd)
                    for tune_set in range(NUM_TUNE_SETS):
                        running_model = copy.deepcopy(MODEL_CLASS)
                        if tune_set%20 == 0:
                            print(f'tune set {tune_set}')
                        if args.pretrain:
                            pretrained_model = next(pretrained_models)
                        else:
                            pretrained_model = None
                        model = running_model.train_model(
                            train_dataset,
                            min_epochs=15,
                            max_epochs=200,
                            dev_data=dev_dataset,
                            pretrained_model=pretrained_model,
                            device=device,
                            config=parameter_sample[tune_set],
                        )
                        tune_accuracy = model.evaluate(
                            data=dev_dataset,
                            device=device,
                            metric="auc",
                            per_subj=BATCH_SUBJECTS,
                        )
                        parameter_evaluations[dev_fold, tune_set] = tune_accuracy
                # Select best parameter set
                mean_dev_accuracies = np.mean(parameter_evaluations, axis=0)
                best_parameter_set = np.argmax(mean_dev_accuracies)
                params_test = parameter_sample[best_parameter_set]
                used_test_params.append(params_test)
                if args.pretrain:
                    pretrained_model = copy.deepcopy(MODEL_CLASS)
                    best_pretrained_model = pretrained_model.pretrain_model(
                                pretrain_dataset,
                                epochs=100,
                                device=device,
                                config=params_test,
                            )
                else:
                    best_pretrained_model = None
            else:  # (not tuning)
                params_test = default_params[args.model]
                best_pretrained_model = None
            # If tune: train using best feature set over dev sets, else: train using default parameters
            # Use fold next to test fold for early stopping
            running_model = copy.deepcopy(MODEL_CLASS)
            dev_fold = (test_fold + 1) % NUM_FOLDS
            train_folds = [
                fold for fold in range(NUM_FOLDS) if fold != test_fold and fold != dev_fold
            ]
            train_dataset = EyetrackingDataset(
                preprocessor,
                folds=train_folds,
                batch_subjects=BATCH_SUBJECTS,
            )
            mean, sd = getmeansd(train_dataset, batch=BATCH_SUBJECTS)
            train_dataset.standardize(mean, sd)
            dev_dataset = EyetrackingDataset(
                preprocessor,
                folds=[dev_fold],
                batch_subjects=BATCH_SUBJECTS
            )
            dev_dataset.standardize(mean, sd)
            test_dataset = EyetrackingDataset(
                preprocessor,
                folds=[test_fold],
                batch_subjects=BATCH_SUBJECTS,
            )
            test_dataset.standardize(mean, sd)
            model = running_model.train_model(
                train_dataset,
                min_epochs=15,
                max_epochs=200,
                dev_data=dev_dataset,
                pretrained_model=best_pretrained_model,
                device=device,
                config=params_test,
            )
            print(f'test accuraccy fold ', test_fold)
            test_accuracy = model.evaluate(
                test_dataset,
                device=device,
                metric="all",
                print_report=True,
                per_subj=BATCH_SUBJECTS,
                save_errors=args.save_errors,
            )
            test_accuracies.append(test_accuracy)
            if args.roc:
                y_preds, y_trues, subjs = model.predict_probs(
                    test_dataset,
                    device=device,
                    per_subj=BATCH_SUBJECTS,
                )
                tprs_folds[str(test_fold)] = (y_trues, y_preds, subjs)
                Roc.get_tprs_aucs(y_trues, y_preds, test_fold)
        
        if tune:
            print("used test params: ", used_test_params)
        print(
            "mean:",
            np.mean(test_accuracies, axis=0),
            "std:",
            np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS),
        )
        
        if args.roc:
            Roc.plot()
            Roc.save()
            print("auc: ", Roc.mean_auc, "+-", Roc.std_auc)
        pred_level = "subjectpred" if BATCH_SUBJECTS else "sentpred"
        final_scores_mean = np.mean(test_accuracies, axis=0)
        final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS)
        final_scores_mean = np.insert(final_scores_mean, 0, Roc.mean_auc, axis=0)
        final_scores_std = np.insert(final_scores_std, 0, Roc.std_auc, axis=0)
        out_str = ""
        with open(f"{args.model}_scores_{pred_level}.txt", "w") as f:
            for i in range(len(final_scores_mean)):
                out_str += f"${round(final_scores_mean[i],2):1.2f}\db{{{round(final_scores_std[i],2):1.2f}}}$"
                if i < len(final_scores_mean) - 1:
                    out_str += " & "
                else:
                    out_str += " \\\\ "
            f.write(out_str)
           
            
        with open(f'{folder}folds_{setting}.pickle', 'wb') as handle:
            pickle.dump(preprocessor._folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'{folder}tpr_folds_{setting}.pickle', 'wb') as handle:
            pickle.dump(tprs_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'{folder}Roc_tprs_{setting}.pickle', 'wb') as handle:
            pickle.dump(Roc.tprs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(f'{folder}Roc_aucs_{setting}.pickle', 'wb') as handle:
            pickle.dump(Roc.aucs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'{folder}test_accs_{setting}.pickle', 'wb') as handle:
            pickle.dump(test_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open(f'{folder}test_params_{setting}.pickle', 'wb') as handle: 
            pickle.dump(used_test_params, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    raise SystemExit(main())


