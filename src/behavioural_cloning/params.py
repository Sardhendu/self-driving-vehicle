
params = {
    "backbone": "xception",
    
    "train_log_steps": 50,
    "batch_size": 32,
    "epochs": 100,
    
    "eval_log_steps": 500,
    "eval_batch_cnt": 10000,
    "snapshot_log_steps": 500,
    # "tensorboard_dir": "/opt/xception1/summary",
    # "snapshot_dir": "/opt/xception1/snapshot",
    "tensorboard_dir": "./data/xception1/summary",
    "snapshot_dir": "./data/xception1/snapshot",
    
    "poly_decay": {
        "learning_rate": 0.001,
        "end_learning_rate": 0,
        "learning_power": 0.9,
        "learning_rate_min": 0.000001
    },
    
    "weight_decay": None,#0.0001,
    "model_weight_path": "/Users/sam/Downloads/xception_train_step_4500.h5"
}
