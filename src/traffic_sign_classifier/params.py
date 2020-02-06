
params = {
    "num_classes": 43,
    "batch_size": 256,
    "poly_decay_schedular": {
        "learning_rate": 0.01,
        "learning_power": 0.9,
        "learning_rate_min": 0.00001,
        "end_learning_rate": 0
    },
    "optimizer_learning_momentum": 0.9,
    "epochs": 10,
    "train_steps": 34799*10*256,
    "eval_steps": 100,
    "eval_data_cnt": 17,
    "save_checkpoint": 1000,
    "save_summary_steps": 100,
    "model_dir": "./data/model"
}