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

## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

## Import custom parts of the project
from models import getmeansd
from models_reading_speed import LSTMClassifier_RS, RSClassifier, EyetrackingClassifierReadingSpeed, LSTMClassifier, aggregate_speed_per_subject
from constants import hyperparameter_space, get_params
import data
from data import EyetrackingDataset, apply_standardization, EyetrackingDataPreprocessor
from roc import ROC

parser = argparse.ArgumentParser(description="Run enhanced LSTM with predicted reading speed as a feature")
parser.add_argument("--model", dest="model")
parser.add_argument("--roc", dest="roc", action="store_true")
parser.add_argument("--no-roc", dest="roc", action="store_false")
parser.add_argument("--tunesets", type=int, default=1)
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


# Create folders to store results
folder = "results/enhanced_lstm/"
os.makedirs(folder, exist_ok=True) 
os.makedirs(folder+'saved_models/', exist_ok=True)

# Delete previous results
for file in [folder+"hyperparameter_selection.csv", folder+'evaluation.csv']:
    try:
        os.remove(file)
    except OSError:
        pass
    
# Hyperparameter selection
header = "Test fold; Dev fold; Tune set; Parameters; Tune loss\n"
with open(folder+"hyperparameter_selection.csv", 'w') as file:
    file.write(header)

# Evaluation
header = "Test fold; Parameters; Loss\n"
with open(folder+'evaluation.csv', 'w') as file:
    file.write(header)

if tune:
    used_test_params = []
    parameter_sample = [
        get_params(hyperparameter_space[args.model]) for _ in range(NUM_TUNE_SETS)
    ]
    
tprs_folds = {}

# load and preprocess data for training
preprocessor = EyetrackingDataPreprocessor(
    csv_file = 'data/fixation_dataset_long_no_word_padded_word_pos.csv', 
    num_folds = NUM_FOLDS,
    drop_demographics = True
)

######## Getting reading speed predictions ########
tprs_folds_rs = {}
params_test_rs = {"batch_size": 32, "decision_boundary": 0.5, "lr": 0.001, "lstm_hidden_size": 40}
loss_fn = nn.MSELoss()

if os.path.exists(f'{folder}rs_predictions.pickle'):
    print("Loading reading speed predictions")
    with open(f'{folder}rs_predictions.pickle', 'rb') as handle:
        reading_speed = pickle.load(handle)
else:
    print("Generating reading speed predictions")
    for test_fold in range(NUM_FOLDS):
        print("test fold ", test_fold)
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
        running_model = copy.deepcopy(LSTMClassifier_RS)
        model = running_model.train_model(
            train_dataset,
            min_epochs=15,
            max_epochs=200,
            dev_data=dev_dataset,
            device=device,
            config=params_test_rs
        )
        y_preds, y_trues, subjs = model.predict_probs(
            test_dataset,
            device=device,
            per_subj=BATCH_SUBJECTS,
        )
        tprs_folds_rs[str(test_fold)] = (y_trues, y_preds, subjs)
        del running_model, model
        gc.collect()
        torch.cuda.empty_cache()

    pred_speed = []
    subjs = []

    for fold in tprs_folds_rs:
        pred_speed.extend(tprs_folds_rs[fold][1])
        subjs.extend(tprs_folds_rs[fold][2])

    reading_speed = pd.DataFrame({'Reading_speed':pred_speed, 'subj':subjs})
    reading_speed = reading_speed.set_index('subj')

    with open(f'{folder}rs_predictions.pickle', 'wb') as handle:
        pickle.dump(reading_speed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
print("Reading speed predictions are ready")
########################################################
# 		Main training and evaluation
########################################################
preprocessor = EyetrackingDataPreprocessor(
    csv_file = 'data/fixation_dataset_long_no_word_padded_word_pos.csv', 
    num_folds = NUM_FOLDS,
    drop_demographics = False
)
loss_fn = nn.BCEWithLogitsLoss()
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
                    reading_speed = reading_speed
                )
                tune_accuracy = model.evaluate(
                    data=dev_dataset,
                    device=device,
                    metric="loss",
                    per_subj=BATCH_SUBJECTS,
                    reading_speed = reading_speed
                )
                parameter_evaluations[dev_fold, tune_set] = tune_accuracy
                print("Loss:", tune_accuracy)
                out_str = f"{test_fold}; {dev_fold}; {tune_set}; {parameter_sample[tune_set]}; {round(tune_accuracy,4):1.4f}\n"
                with open(folder+"hyperparameter_selection.csv", 'a') as file:
                    file.write(out_str)
                del running_model, model
                gc.collect()
                torch.cuda.empty_cache()
        mean_dev_accuracies = np.mean(parameter_evaluations, axis=0)
        best_parameter_set = np.argmin(mean_dev_accuracies)
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
        reading_speed = reading_speed,
    )
    print(f'test accuraccy fold ', test_fold)
    test_accuracy = model.evaluate(
        test_dataset,
        device=device,
        metric="all",
        print_report=True,
        per_subj=BATCH_SUBJECTS,
        reading_speed = reading_speed
    )
    test_accuracies.append(test_accuracy)
    line = f"{test_fold}; {params_test}; {round(test_accuracy[0],4):1.4f}\n"
    with open(folder+'evaluation.csv', 'a') as file:
        file.write(line)
    torch.save(model.state_dict(), f'{folder}saved_models/test_fold_{test_fold}_model_weights.pth')

    if args.roc:
        y_preds, y_trues, subjs = model.predict_probs(
            test_dataset,
            device=device,
            per_subj=BATCH_SUBJECTS,
            reading_speed = reading_speed
        )
        tprs_folds[str(test_fold)] = (y_trues, y_preds, subjs)
        Roc.get_tprs_aucs(y_trues, y_preds, test_fold) 
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

if args.roc:
    Roc.plot()
    Roc.save()
    print("auc: ", Roc.mean_auc, "+-", Roc.std_auc)

final_scores_mean = np.mean(test_accuracies, axis=0)
final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS)
final_scores_mean = np.insert(final_scores_mean, 0, Roc.mean_auc, axis=0)
final_scores_std = np.insert(final_scores_std, 0, Roc.std_auc, axis=0)
out_str = ""
out_str += "Loss & Accuracy & Precision & Recall & F1 & ROC AUC\\\\"
with open(f"{folder}{args.model}_scores.txt", "w") as f:
    for i in range(len(final_scores_mean)):
        out_str += f"${round(final_scores_mean[i],2):1.2f}\db{{{round(final_scores_std[i],2):1.2f}}}$"
        if i < len(final_scores_mean) - 1:
            out_str += " & "
        else:
            out_str += " \\\\ "
    f.write(out_str)
