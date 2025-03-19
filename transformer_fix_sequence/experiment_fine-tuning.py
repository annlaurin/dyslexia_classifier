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
import gc


## PyTorch
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

## Import custom parts of the project
from models import getmeansd, EyetrackingClassifier, TransformerClassifier, Encoder, EncoderLayer, mask_with_tokens_3D, get_mask_subset_with_prob_3D
from models_fine_tuning import EyetrackingClassifierBinary, BinaryTransformerClassifier, AnnasPositionalEncoding, aggregate_per_subject
from constants import hyperparameter_space, get_params
import data
from data import EyetrackingDataset, apply_standardization, EyetrackingDataPreprocessor
from roc import ROC

parser = argparse.ArgumentParser(description="Run tansformer fine-tuning")
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
parser.set_defaults(tune=True) 
parser.set_defaults(roc=True)
parser.set_defaults(batch_subjects=False) 
parser.set_defaults(model = "transformer_tuning_slow")
args = parser.parse_args()

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    device = torch.device(f'cuda:{args.cudaid}')
  
NUM_FOLDS = 3
NUM_TUNE_SETS = args.tunesets
BATCH_SUBJECTS = args.batch_subjects
CHECKPOINT_PATH = "results/pretraining/"
MODEL_CLASS = BinaryTransformerClassifier
tune = args.tune

# Prepare ROC Curves
if args.roc:
    Roc = ROC(args.model, args.tune)


# Create folders to store results
folder = "results/"+f'{args.model}/'
os.makedirs(folder, exist_ok=True) 


# Delete previous results
for file in [folder+"hyperparameter_selection.csv", folder+'evaluation.csv']:
    try:
        os.remove(file)
    except OSError:
        pass

# Hyperparameter selection results
header = "Test fold; Dev fold; Tune set; Parameters; Tune loss\n"
with open(folder+"hyperparameter_selection.csv", 'w') as file:
    file.write(header)

# Evaluation results
header = "Test fold; Parameters; Loss\n"
with open(folder+'evaluation.csv', 'w') as file:
    file.write(header)

if tune:
    used_test_params = []
    parameter_sample = [
        get_params(hyperparameter_space[args.model]) for _ in range(NUM_TUNE_SETS)
    ]
    
tprs_folds = {}
loss_fn = nn.BCEWithLogitsLoss()

# load and preprocess data for training
preprocessor = EyetrackingDataPreprocessor(
    csv_file = 'data/30fixations_no_padding_sentence_word_pos.csv', 
    num_folds = NUM_FOLDS,
    drop_demographics = False
)

test_accuracies = []

    
def reload_model(freezing):
    model = torch.load(CHECKPOINT_PATH+"final_full_model.pth", weights_only=False)
    layers = [model.encoder.layers[0].attn_layer_norm, 
              model.encoder.layers[0].attention, 
              model.encoder.layers[0].ffn_layer_norm,
              model.encoder.layers[0].ff[0],  # this
              model.encoder.layers[0].ff[1],
              model.encoder.layers[0].ff[2],
              model.encoder.layers[0].ff[3],
              model.encoder.layers[0].ff[4]   # and this
             ]
    if freezing == "transformer_tuning_frozen":
        frozen_layers = layers
    elif freezing == "transformer_tuning_slow":
        frozen_layers = np.delete(layers, [3, 7])
    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False
    return model



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
                if tune_set%100 == 0:
                    print(f'tune set {tune_set}')
                model = None
                pretrained_mod = reload_model(args.model)
                model = running_model.train_model_bin(
                            train_dataset,
                            min_epochs=15,
                            max_epochs=300,
                            dev_data=dev_dataset,
                            pretrained_model=pretrained_mod,
                            device=device,
                            config= parameter_sample[tune_set]
                        )
                tune_accuracy = model.evaluate_bin(
                    data=dev_dataset,
                    device=device,
                    metric="loss",
                    config = parameter_sample[tune_set]
                )
                parameter_evaluations[dev_fold, tune_set] = tune_accuracy
                print(tune_accuracy)
                out_str = f"{test_fold}; {dev_fold}; {tune_set}; {parameter_sample[tune_set]}; {round(tune_accuracy,4):1.4f}\n"
                with open(folder+"hyperparameter_selection.csv", 'a') as file:
                    file.write(out_str)
                del running_model, model
                gc.collect()
                torch.cuda.empty_cache()

            # Select best parameter set
        mean_dev_loss = np.mean(parameter_evaluations, axis=0)
        best_parameter_set = np.argmin(mean_dev_loss)
        params_test = parameter_sample[best_parameter_set]
        print(f'best performing parameter for test fold ', test_fold, ": ", params_test)
        used_test_params.append(params_test)
        best_pretrained_model = None
    pretrained_mod = reload_model(args.model)    
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
    model = running_model.train_model_bin(
        train_dataset,
        min_epochs=15,
        max_epochs=200,
        dev_data=dev_dataset,
        pretrained_model=pretrained_mod,
        device=device,
        config=params_test,
    )
    print(f'test accuraccy fold ', test_fold)
    test_accuracy = model.evaluate_bin(
        test_dataset,
        device=device,
        metric="all",
        print_report=True,
        per_subj=BATCH_SUBJECTS,
        config = params_test
    )
    line = f"{test_fold}; {params_test}; {round(test_accuracy[0],4):1.4f}\n"
    with open(folder+'evaluation.csv', 'a') as file:
        file.write(line)
    test_accuracies.append(test_accuracy)
    if args.roc:
        y_preds, y_trues, subjs = model.predict_probs(
            test_dataset,
            device=device,
            per_subj=BATCH_SUBJECTS,
            config=params_test
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
pred_level = "subjectpred" if BATCH_SUBJECTS else "textpred"
final_scores_mean = np.mean(test_accuracies, axis=0)
final_scores_std = np.std(test_accuracies, axis=0) / np.sqrt(NUM_FOLDS)
final_scores_mean = np.insert(final_scores_mean, 0, Roc.mean_auc, axis=0)
final_scores_std = np.insert(final_scores_std, 0, Roc.std_auc, axis=0)
time_elapsed = time.time() - start_time 
out_str = ""
out_str += "Loss & Accuracy & Precision & Recall & F1 & ROC AUC\\\\"
with open(f"{folder}{args.model}_scores.txt", "w") as f:
    for i in range(len(final_scores_mean)):
        out_str += f"${round(final_scores_mean[i],2):1.2f}\db{{{round(final_scores_std[i],2):1.2f}}}$"
        if i < len(final_scores_mean) - 1:
            out_str += " & "
        else:
            out_str += " \\\\ "
    out_str += "\n"
    out_str += f"Time: {time_elapsed:7.2f} seconds.\n" 
    f.write(out_str)