import numpy as np

## Constants
hyperparameter_space = {
    "lstm": {
        "batch_size": [8, 16, 32, 64],           
        "lr": np.linspace(1e-5, 1e-1, num=15),    
        "lstm_hidden_size": [30, 40, 50, 60, 70],       
        "decision_boundary": [0.5, 0.5]      
    }
}

default_params = {
    "lstm": {
        "epochs": 40,
        "batch_size": 32,
        "lr": 0.001,
        "lstm_hidden_size": 50,
        "decision_boundary": 0.5
    }
}

features = [
    "age",
    "sex",
    "fix_dur",
    "fix_x",
    "fix_y",
    "landing",
    "word_length",
    'predictability', 
    'frequency',
    'number.morphemes', 
    'next_fix_dist', 
    'sac_ampl', 
    'sac_angle', 
    'sac_vel',
    'dummy_1',
    'dummy_2', 
    'dummy_3', 
    'dummy_4', 
    'dummy_5', 
    'dummy_6',
    'dummy_Grade_0', 
    'dummy_dir_DOWN', 
    'dummy_dir_LEFT', 
    'dummy_dir_RIGHT', 
    'dummy_dir_UP',
    'dummy_dir_0'
    ]

num_features = len(features)