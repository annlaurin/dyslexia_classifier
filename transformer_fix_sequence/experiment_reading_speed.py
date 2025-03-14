import os
import os.path
import numpy as np
import pandas as pd
import random
import pickle
from functools import partial, reduce
import argparse
import copy
import time
from collections import Counter
import json
import gc
import itertools

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

## Import custom parts of the project
from models import getmeansd
from models_reading_speed import LSTMClassifier_RS, RSClassifier, aggregate_speed_per_subject
from constants import hyperparameter_space, get_params
import data
from data import EyetrackingDataset, apply_standardization, EyetrackingDataPreprocessor


parser = argparse.ArgumentParser(description="Predict reading speed and evaluate loss")
parser.add_argument("--model", dest="model")
parser.add_argument("--no-roc", dest="roc", action="store_false")
parser.add_argument("--tunesets", type=int, default=55)
parser.add_argument("--tune", dest="tune", action="store_true")
parser.add_argument("--no-tune", dest="tune", action="store_false")
parser.add_argument("--wordvectors", type=str, default="none")
parser.add_argument("--pretrain", dest="pretrain", action="store_true")
parser.add_argument("--subjpred", dest="batch_subjects", action="store_false")
parser.add_argument("--seed", dest="seed", type=int, default=43)
parser.add_argument("--cuda", dest="cudaid", default=0)
parser.set_defaults(tune=True) #True
parser.set_defaults(roc=True)
parser.set_defaults(batch_subjects=True) #True
parser.set_defaults(model = "lstm")
args = parser.parse_args()


# Ensure that all operations are deterministic on GPU (if used) for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()

if args.model == "lstm":
    MODEL_CLASS = LSTMClassifier_RS
    
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device = torch.device(f'cuda:{args.cudaid}')
    
NUM_FOLDS = 10
#NUM_TUNE_SETS = args.tunesets
BATCH_SUBJECTS = args.batch_subjects
tune = args.tune

# Create folders to store results
folder = "results/predicting_reading_speed/"
os.makedirs(folder, exist_ok=True) 
os.makedirs(folder+'saved_models/', exist_ok=True)

if tune:
#    used_test_params = []
#    parameter_sample = [
#        get_params(hyperparameter_space[args.model]) for _ in range(NUM_TUNE_SETS)
#    ]
    keys, values = zip(*hyperparameter_space[args.model].items())
    parameter_sample = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(parameter_sample)
NUM_TUNE_SETS = len(parameter_sample)
    
tprs_folds = {}


# load and preprocess data for training
preprocessor = EyetrackingDataPreprocessor(
    csv_file = 'data/fixation_dataset_long_no_word_padded_word_pos.csv', 
    num_folds = NUM_FOLDS,
    drop_demographics = True
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
                print(f'tune set {tune_set}')
                if args.pretrain:
                    pretrained_model = next(pretrained_models)
                else:
                    pretrained_model = None
                model = None
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
                    metric="loss",
                    per_subj=BATCH_SUBJECTS,
                )
                parameter_evaluations[dev_fold, tune_set] = tune_accuracy
                print("Tune loss:", tune_accuracy)
                del running_model, model
                gc.collect()
                torch.cuda.empty_cache()
        # Select best parameter set
        mean_dev_loss = np.mean(parameter_evaluations, axis=0)
        best_parameter_set = np.argmin(mean_dev_loss)
        params_test = parameter_sample[best_parameter_set]
        # print(f'best performing parameter for fold ', test_fold, ": ", params_test)
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
    test_accuracy = model.evaluate(
        test_dataset,
        device=device,
        metric="loss",
        print_report=True,
        per_subj=BATCH_SUBJECTS,
        save_errors=args.save_errors,
    )
    print("test loss fold ", test_fold, " : ", test_accuracy)
    test_accuracies.append(test_accuracy)
    torch.save(model.state_dict(), f'results/predicting_reading_speed/saved_models/test_fold_{test_fold}')
    if True:
        y_preds, y_trues, subjs = model.predict_probs(
            test_dataset,
            device=device,
            per_subj=BATCH_SUBJECTS,
        )
        tprs_folds[str(test_fold)] = (y_trues, y_preds, subjs)
    del running_model, model
    gc.collect()
    torch.cuda.empty_cache()
if tune:
    print("used test params: ", used_test_params)
print(
    "mean:",
    np.mean(test_accuracies, axis=0),
    "std:",
    np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS),
)

final_scores_mean = np.mean(test_accuracies, axis=0)
final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS)

# Find if there is a winning combination.
json_dicts = [json.dumps(d, sort_keys=True) for d in used_test_params]
# Count duplicates
counts = Counter(json_dicts)
# There is:
max_entry = max(counts.items(), key=lambda x: x[1])


folds = []
subjects = []

for fold in range(len(preprocessor._folds)):
    fold_column = np.full(len(preprocessor._folds[fold]), fold)
    folds.extend(fold_column)
    subjects.extend(preprocessor._folds[fold])
subj_folds = pd.DataFrame({'fold':folds, 'subject':subjects})


folds = []
groups = []
pred_probs = []
subjs = []

for fold in tprs_folds:
    fold_column = np.full(len(tprs_folds[fold][0]), int(fold))
    folds.extend(fold_column)
    groups.extend(tprs_folds[fold][0])
    pred_probs.extend(tprs_folds[fold][1])
    subjs.extend(tprs_folds[fold][2])

pred_folds = pd.DataFrame({'fold':folds, 'group':groups, 'pred_speed':pred_probs, 'subject':subjs})

df = subj_folds.merge(pred_folds, on='subject')
df = df.drop(['fold_x'], axis=1)
df = df.rename(columns={'fold_y': "fold"})

# load demographics
demo = pd.read_csv('data/demo_filtered_centered.csv', decimal=",")
demo = demo.rename(columns={'subj_demo': "subject"})

# add predicted reading speed to the demographic information
final = df.merge(demo, on =['subject'])
final.to_csv('data/rs_predictions_full_info.csv', index=False)

with open(f'{folder}folds_best_hyperparameters.pickle', 'wb') as handle:
    pickle.dump(max_entry, handle, protocol=pickle.HIGHEST_PROTOCOL) 

with open(f'{folder}folds.pickle', 'wb') as handle:
    pickle.dump(preprocessor._folds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{folder}tpr_folds.pickle', 'wb') as handle:
    pickle.dump(tprs_folds, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{folder}final_scores_mean.pickle', 'wb') as handle:
    pickle.dump(final_scores_mean, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(f'{folder}test_accs.pickle', 'wb') as handle:
    pickle.dump(test_accuracies, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(f'{folder}test_params.pickle', 'wb') as handle: 
    pickle.dump(used_test_params, handle, protocol=pickle.HIGHEST_PROTOCOL)