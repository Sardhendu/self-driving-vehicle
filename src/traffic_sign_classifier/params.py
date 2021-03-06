
params = {
    "img_height": 32,
    "img_width": 32,
    "num_classes": 43,
    "batch_size": 256,
    "weight_deacay": 0.0001,
    "poly_decay_schedular": {
        "learning_rate": 0.01,
        "learning_power": 0.9,
        "learning_rate_min": 0.00001,
        "end_learning_rate": 0.000001
    },
    "poly_cosine_schedular": {
        "learning_rate": 0.01,
        "poly_power": 0.9,
        "end_learning_rate": 0.00001
    },
    "optimizer_learning_momentum": 0.9,
    "epochs": 400,
    "train_steps": int(34799/256),
    "eval_steps": 200,
    "eval_data_cnt": 17,
    "save_checkpoint": 1000,
    "save_summary_steps": 100,
    "model_dir": "./data/model",
    "summary_dir": "./data/model/summary"
}

