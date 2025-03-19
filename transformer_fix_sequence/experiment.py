import os
import numpy as np
import pandas as pd
import random
from functools import partial, reduce
import argparse
import copy
import time
import gc

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

## Import custom parts of the project
from models import mse_loss, mask_with_tokens_3D, get_mask_subset_with_prob_3D, AnnasPositionalEncoding, getmeansd, EncoderLayer, Encoder, EyetrackingClassifier, TransformerClassifier
from constants import hyperparameter_space, nested_space, get_params_nested
import data
from data import EyetrackingDataset, apply_standardization, EyetrackingDataPreprocessor


parser = argparse.ArgumentParser(description="Run Russian Eye-Movement Pretraining")
parser.add_argument("--model", dest="model")
parser.add_argument("--tunesets", type=int, default=1000)
parser.add_argument("--tune", dest="tune", action="store_true")
parser.add_argument("--no-tune", dest="tune", action="store_false")
parser.add_argument("--pretrain", dest="pretrain", action="store_true")
parser.add_argument("--seed", dest="seed", type=int, default=76) 
parser.add_argument("--cuda", dest="cudaid", default=0)
parser.set_defaults(tune=True) 
parser.set_defaults(model = "transformer")
args = parser.parse_args()

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()

if args.model == "transformer":
    MODEL_CLASS = TransformerClassifier 
    
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device = torch.device(f'cuda:{args.cudaid}')
    
NUM_FOLDS = 10
NUM_TUNE_SETS = args.tunesets
tune = args.tune

BATCH_SUBJECTS = False


# Create folders to store results
folder = "results/pretraining/"
os.makedirs(folder, exist_ok=True) 

# Delete previous results
for file in [folder+"hyperparameter_selection.csv", folder+'evaluation.csv']:
    try:
        os.remove(file)
    except OSError:
        pass

# Hyperparameter selection
header = "Dev fold; Tune set; Parameters; Tune loss\n"
with open(folder+"hyperparameter_selection.csv", 'w') as file:
    file.write(header)

# Evaluation
header = "Test fold; Parameters; Loss\n"
with open(folder+'evaluation.csv', 'w') as file:
    file.write(header)



# Combined dataset: data from children (293) and adults (114), total N = 407
file = "data/30fixations_RSC_and_children.csv" 

if tune:
    used_test_params = []
    parameter_sample = [
        get_params_nested(hyperparameter_space["transformer"], nested_space) for _ in range(NUM_TUNE_SETS)
    ]

tprs_folds = {}
    

# load and preprocess data for training
preprocessor = EyetrackingDataPreprocessor(
    csv_file = file, 
   num_folds = NUM_FOLDS
)

test_accuracies = []

dev_fold = 2   # dev fold stays the same
parameter_evaluations = np.zeros(shape=(NUM_FOLDS, NUM_TUNE_SETS))
if tune:
    train_folds = [
        fold
        for fold in range(NUM_FOLDS)
        if fold != dev_fold
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
        if tune_set%100 == 0:
            print(f'tune set {tune_set}')
        if args.pretrain:
            pretrained_model = next(pretrained_models)
        else:
            pretrained_model = None
        model = None
        model = running_model.train_model(
            train_dataset,
            min_epochs=15,
            max_epochs=300,
            dev_data=dev_dataset,
            pretrained_model=pretrained_model,
            device=device,
            config=parameter_sample[tune_set],
        )
        tune_accuracy = model.evaluate(
            data=dev_dataset,
            metric="loss",
            device=device,
	    batch_size = parameter_sample[tune_set]["batch_size"]
        )
        parameter_evaluations[dev_fold, tune_set] = tune_accuracy
        print(tune_accuracy)
        out_str = f"{dev_fold}; {tune_set}; {parameter_sample[tune_set]}; {round(tune_accuracy,4):1.4f}\n"
        with open(folder+"hyperparameter_selection.csv", 'a') as file:
            file.write(out_str)
        del running_model, model
        gc.collect()
        torch.cuda.empty_cache()
    # Select best parameter set
    mean_dev_loss = np.mean(parameter_evaluations, axis=0)
    best_parameter_set = np.argmin(mean_dev_loss)
    params_test = parameter_sample[best_parameter_set]
    print(f'best performing parameter:", {params_test}')
    used_test_params.append(params_test)
    best_pretrained_model = None
    
# Evaluation    
for test_fold in range(NUM_FOLDS):
    if dev_fold == test_fold:
        continue
    print(f'test fold {test_fold}')
    running_model = copy.deepcopy(MODEL_CLASS)

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
        max_epochs=300,
        dev_data=dev_dataset,
        pretrained_model=best_pretrained_model,
        device=device,
        config=params_test,
    )
    test_loss = model.evaluate(
        test_dataset,
        device=device,
        metric="loss",
	batch_size = params_test["batch_size"]
    )
    print("test loss fold", test_fold, ":", test_loss)
    line = f"{test_fold}; {params_test}; {round(test_loss,4):1.4f}\n"
    with open(folder+'evaluation.csv', 'a') as file:
        file.write(line)
    test_accuracies.append(test_loss)

if tune:
    print("used test params: ", used_test_params)
print(
    "mean:",
    np.mean(test_accuracies, axis=0),
    "std:",
    np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS-1),  # -1 because dev_fold is never used for testing
)

final_scores_mean = np.mean(test_accuracies, axis=0)
final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS-1) # -1 because dev_fold is never used 
time_elapsed = time.time() - start_time 
out_str = ""
out_str += "Loss & Accuracy & Precision & Recall & F1 & ROC AUC\\\\"
with open(f"results.txt", "w") as f:
    out_str += f"${round(final_scores_mean,6):1.6f}\db{{{round(final_scores_std,6):1.6f}}}$"
    out_str += "\n"
    out_str += f"Time: {time_elapsed:7.2f} seconds.\n" 
    out_str += f"Test losses: {test_accuracies}\n"
    out_str += f"Used test params: {used_test_params}"
    f.write(out_str)
