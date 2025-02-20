import numpy as np
import random

hyperparameter_space = {
    "transformer": {
        "lr": np.linspace(1e-03, 1e-05, num=15), 
        "num_layers": [1, 2, 4, 8, 16],      
        "dropout": np.linspace(0, 0.5, num=15),
        "mask_prob": np.linspace(0, 0.5, num=15),
        "upscale_dim": [14, 32, 64, 128, 256, 512, 1024],
        "batch_size": [256],
        # "num_heads" is specified in get_params_nested()
        # "inner_dim" is specified in get_params_nested()
    }
}

nested_space = {
    "14": [1, 2, 7, 14],
    "32": [1, 4, 8, 16, 32],
    "64": [1, 8, 16, 32, 64],
    "128": [1, 8, 32, 64, 128],
    "256": [1, 32, 64, 128, 256],
    "512": [32, 64, 128, 256, 512],
    "1024": [32, 64, 128, 256, 512], 
}


def get_params_nested(paramdict, nested) -> dict:
    selected_pars = dict()
    for k in paramdict:
        selected_pars[k] = random.sample(list(paramdict[k]), 1)[0] 
    factor = [1, 2, 4, 8]
    dimension = selected_pars["upscale_dim"]
    selected_pars["num_heads"] = random.sample(list(nested[f"{dimension}"]), 1)[0]
    selected_pars["inner_dim"] = random.sample([dimension*x for x in factor], 1)[0]
        
    return selected_pars